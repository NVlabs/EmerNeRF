from typing import Callable, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import logging

import torch
from nerfacc.data_specs import RayIntervals
from nerfacc.estimators.base import AbstractEstimator
from nerfacc.pdf import importance_sampling, searchsorted
from nerfacc.volrend import render_transmittance_from_density
from torch import Tensor

# acknowledgement: this file is mostly adpated from nerfacc

logger = logging.getLogger()


def blur_stepfun(x, y, r):
    # taken and modified from https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/stepfun.py
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (
        torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1)
        - torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
    ) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum(
        (xr[..., 1:] - xr[..., :-1]) * torch.cumsum(y2, dim=-1), dim=-1
    ).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr


def sorted_interp_quad(x, xp, fpdf, fcdf):
    # taken and modified from https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/stepfun.py
    """interp in quadratic"""
    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x, return_idx=False):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, x0_idx = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, x1_idx = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        if return_idx:
            return x0, x1, x0_idx, x1_idx
        return x0, x1

    fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
    fpdf0 = fpdf.take_along_dim(fcdf0_idx, dim=-1)
    fpdf1 = fpdf.take_along_dim(fcdf1_idx, dim=-1)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret


class PropNetEstimator(AbstractEstimator):
    """Proposal network transmittance estimator.

    References: "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields."

    Args:
        optimizer: The optimizer to use for the proposal networks.
        scheduler: The learning rate scheduler to use for the proposal networks.
    """

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        enable_anti_aliasing_loss: Optional[bool] = True,
        anti_aliasing_pulse_width: Optional[List[float]] = [0.03, 0.003],
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prop_cache: List = []
        self.enable_anti_aliasing_loss = enable_anti_aliasing_loss
        self.pulse_width = anti_aliasing_pulse_width
        if self.enable_anti_aliasing_loss:
            logger.info("Enable anti-aliasing loss, pulse width: %s", self.pulse_width)

    @torch.no_grad()
    def sampling(
        self,
        prop_sigma_fns: List[Callable],
        prop_samples: List[int],
        num_samples: int,
        # rendering options
        n_rays: int,
        near_plane: float,
        far_plane: float,
        sampling_type: Literal[
            "uniform", "lindisp", "sqrt", "log", "uniform_lindisp"
        ] = "uniform_lindisp",
        # training options
        stratified: bool = False,
        requires_grad: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Sampling with CDFs from proposal networks.

        Note:
            When `requires_grad` is `True`, the gradients are allowed to flow
            through the proposal networks, and the outputs of the proposal
            networks are cached to update them later when calling `update_every_n_steps()`

        Args:
            prop_sigma_fns: Proposal network evaluate functions. It should be a list
                of functions that take in samples {t_starts (n_rays, n_samples),
                t_ends (n_rays, n_samples)} and returns the post-activation densities
                (n_rays, n_samples).
            prop_samples: Number of samples to draw from each proposal network. Should
                be the same length as `prop_sigma_fns`.
            num_samples: Number of samples to draw in the end.
            n_rays: Number of rays.
            near_plane: Near plane.
            far_plane: Far plane.
            sampling_type: Sampling type. Either "uniform" or "lindisp". Default to
                "lindisp".
            stratified: Whether to use stratified sampling. Default to `False`.
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.

        Returns:
            A tuple of {Tensor, Tensor}:

            - **t_starts**: The starts of the samples. Shape (n_rays, num_samples).
            - **t_ends**: The ends of the samples. Shape (n_rays, num_samples).

        """
        assert len(prop_sigma_fns) == len(prop_samples), (
            "The number of proposal networks and the number of samples "
            "should be the same."
        )
        cdfs = torch.cat(
            [
                torch.zeros((n_rays, 1), device=self.device),
                torch.ones((n_rays, 1), device=self.device),
            ],
            dim=-1,
        )
        intervals = RayIntervals(vals=cdfs)

        for i, (level_fn, level_samples) in enumerate(
            zip(prop_sigma_fns, prop_samples)
        ):
            intervals, _ = importance_sampling(
                intervals, cdfs, level_samples, stratified
            )
            t_vals = _transform_stot(
                sampling_type, intervals.vals, near_plane, far_plane
            )
            t_starts = t_vals[..., :-1]
            t_ends = t_vals[..., 1:]

            with torch.set_grad_enabled(requires_grad):
                sigmas = level_fn(t_starts, t_ends)["density"].squeeze(-1)
                assert sigmas.shape == t_starts.shape
                trans, _ = render_transmittance_from_density(t_starts, t_ends, sigmas)
                cdfs = 1.0 - torch.cat(
                    [trans, torch.zeros_like(trans[..., :1])], dim=-1
                )
                if requires_grad:
                    self.prop_cache.append((intervals, cdfs, i))

        intervals, _ = importance_sampling(intervals, cdfs, num_samples, stratified)
        t_vals = _transform_stot(sampling_type, intervals.vals, near_plane, far_plane)
        t_starts = t_vals[..., :-1]
        t_ends = t_vals[..., 1:]
        if requires_grad:
            self.prop_cache.append((intervals, None, None))

        return t_starts, t_ends

    @torch.enable_grad()
    def compute_loss(self, trans: Tensor, loss_scaler: float = 1.0) -> Tensor:
        """Compute the loss for the proposal networks.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            loss_scaler: The loss scaler. Default to 1.0.

        Returns:
            The loss for the proposal networks.
        """
        if len(self.prop_cache) == 0:
            return torch.zeros((), device=self.device)

        intervals, _, _ = self.prop_cache.pop()
        # get cdfs at all edges of intervals
        cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[..., :1])], dim=-1)
        cdfs = cdfs.detach()
        loss = 0.0

        if self.enable_anti_aliasing_loss:
            w_normalize = (cdfs[..., 1:] - cdfs[..., :-1]) / (
                intervals.vals[..., 1:] - intervals.vals[..., :-1]
            )
            c1, w1 = blur_stepfun(intervals.vals, w_normalize, self.pulse_width[0])
            c2, w2 = blur_stepfun(intervals.vals, w_normalize, self.pulse_width[1])
            area1 = 0.5 * (w1[..., 1:] + w1[..., :-1]) * (c1[..., 1:] - c1[..., :-1])
            area2 = 0.5 * (w2[..., 1:] + w2[..., :-1]) * (c2[..., 1:] - c2[..., :-1])
            cdfs1 = torch.cat(
                [
                    torch.zeros_like(area1[..., :1]),
                    torch.cumsum(area1, dim=-1),
                ],
                dim=-1,
            )
            cdfs2 = torch.cat(
                [
                    torch.zeros_like(area2[..., :1]),
                    torch.cumsum(area2, dim=-1),
                ],
                dim=-1,
            )
            cs = [c1, c2]
            ws = [w1, w2]
            _cdfs = [cdfs1, cdfs2]
            while self.prop_cache:
                prop_intervals, prop_cdfs, prop_id = self.prop_cache.pop()
                wp = prop_cdfs[..., 1:] - prop_cdfs[..., :-1]
                cdf_interp = sorted_interp_quad(
                    prop_intervals.vals, cs[prop_id], ws[prop_id], _cdfs[prop_id]
                )
                w_s = torch.diff(cdf_interp, dim=-1)
                loss += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()
        else:
            while self.prop_cache:
                prop_intervals, prop_cdfs, _ = self.prop_cache.pop()
                loss += _pdf_loss(intervals, cdfs, prop_intervals, prop_cdfs).mean()
        return loss * loss_scaler

    @torch.enable_grad()
    def update_every_n_steps(
        self,
        trans: Tensor,
        requires_grad: bool = False,
        loss_scaler: float = 1.0,
    ) -> float:
        """Update the estimator every n steps during training.

        Args:
            trans: The transmittance of all samples. Shape (n_rays, num_samples).
            requires_grad: Whether to allow gradients to flow through the proposal
                networks. Default to `False`.
            loss_scaler: The loss scaler to use. Default to 1.0.

        Returns:
            The loss of the proposal networks for logging (a float scalar).
        """
        if requires_grad:
            return self._update(trans=trans, loss_scaler=loss_scaler)
        else:
            if self.scheduler is not None:
                self.scheduler.step()
            return 0.0

    @torch.enable_grad()
    def _update(self, trans: Tensor, loss_scaler: float = 1.0) -> float:
        assert len(self.prop_cache) > 0
        assert self.optimizer is not None, "No optimizer is provided."

        loss = self.compute_loss(trans, loss_scaler)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()


def get_proposal_requires_grad_fn(
    target: float = 5.0, num_steps: int = 1000
) -> Callable:
    schedule = lambda s: min(s / num_steps, 1.0) * target

    steps_since_last_grad = 0

    def proposal_requires_grad_fn(step: int) -> bool:
        nonlocal steps_since_last_grad
        target_steps_since_last_grad = schedule(step)
        requires_grad = steps_since_last_grad > target_steps_since_last_grad
        if requires_grad:
            steps_since_last_grad = 0
        steps_since_last_grad += 1
        return requires_grad

    return proposal_requires_grad_fn


TRANSFROM_DICT = {
    "uniform": (lambda x: x, lambda x: x),
    "lindisp": (lambda x: 1 / x, lambda x: 1 / x),
    "sqrt": (lambda x: torch.sqrt(x), lambda x: x**2),
    "log": (lambda x: torch.log(x), lambda x: torch.exp(x)),
    "uniform_lindisp": (
        # lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
        # lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
        lambda x: torch.where(x < 200, x / 400, 1 - 1 / (2 * x / 200)),
        lambda x: torch.where(x < 0.5, x * 400, 200 / (2 - 2 * x)),
    ),
    "uniform_lindisp_0": (
        lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
        lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
    ),
}


def _transform_stot(
    transform_type: Literal[
        "uniform", "lindisp", "sqrt", "log", "uniform_lindisp", "uniform_lindisp_0"
    ],
    s_vals: Tensor,
    t_min: Tensor,
    t_max: Tensor,
) -> Tensor:
    if isinstance(t_min, float):
        t_min = torch.tensor(t_min, device=s_vals.device)
    if t_min.dim() > 0:
        t_min = t_min[:, None]
    if isinstance(t_max, float):
        t_max = torch.tensor(t_max, device=s_vals.device)
    if t_max.dim() > 0:
        t_max = t_max[:, None]
    if transform_type in TRANSFROM_DICT:
        _contract_fn, _icontract_fn = TRANSFROM_DICT[transform_type]
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    s_min, s_max = _contract_fn(t_min), _contract_fn(t_max)
    icontract_fn = lambda s: _icontract_fn(s * s_max + (1 - s) * s_min)
    return icontract_fn(s_vals)


def _pdf_loss(
    segments_query: RayIntervals,
    cdfs_query: Tensor,
    segments_key: RayIntervals,
    cdfs_key: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    ids_left, ids_right = searchsorted(segments_key, segments_query)
    if segments_query.vals.dim() > 1:
        w = cdfs_query[..., 1:] - cdfs_query[..., :-1]
        ids_left = ids_left[..., :-1]
        ids_right = ids_right[..., 1:]
    else:
        assert segments_query.is_left is not None
        assert segments_query.is_right is not None
        w = cdfs_query[segments_query.is_right] - cdfs_query[segments_query.is_left]
        ids_left = ids_left[segments_query.is_left]
        ids_right = ids_right[segments_query.is_right]

    w_outer = cdfs_key.gather(-1, ids_right) - cdfs_key.gather(-1, ids_left)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)
