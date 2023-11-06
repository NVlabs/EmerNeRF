from typing import Callable, Dict, List, Optional, Tuple

import torch
from nerfacc import (
    accumulate_along_rays,
    render_transmittance_from_density,
    render_weight_from_density,
)
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from radiance_fields import DensityField, RadianceField
from third_party.nerfacc_prop_net import PropNetEstimator

# acknowledgement: this code is inspired by the code from nerfacc


def render_weights_opacity_depth_from_density(
    t_starts: Tensor,
    t_ends: Tensor,
    density: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Renders weights, opacities, and depths from density along rays.

    Args:
        t_starts (torch.Tensor): Starting points of rays.
        t_ends (torch.Tensor): Ending points of rays.
        density (torch.Tensor): Density values along rays.

    Returns:
        Tuple of torch.Tensor: weights, opacities, and depths.
    """
    weights, _, _ = render_weight_from_density(t_starts, t_ends, density)
    opacities = accumulate_along_rays(
        weights,
        values=None,
    ).clamp(1e-6, 1.0)
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
    )
    depths = depths / opacities
    return weights, opacities, depths


def rendering(
    t_starts: Tensor,
    t_ends: Tensor,
    query_fn: Optional[Callable] = None,
    return_decomposition: bool = False,
) -> Dict[str, Tensor]:
    """
    Renders the scene given the start and end points of the rays, and a query function to retrieve information about the scene along the rays.
    Args:
        t_starts (Tensor): The start points of the rays.
        t_ends (Tensor): The end points of the rays.
        query_fn (Optional[Callable], optional):
            A function that takes in the start and end points of the rays and
            returns a dictionary of information about the scene along the rays.
        return_decomposition (bool, optional):
            Whether to return the decomposition of the scene into static and dynamic components.
            Defaults to False.

    Returns:
        Dict[str, Tensor]: A dictionary of rendered information about the scene, including density, depth, opacity, rgb values, and more.
    """
    # query the scene for density and other information along the rays
    results = query_fn(t_starts, t_ends)

    # calculate transmittance and alpha values for each point along the rays
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, results["density"].squeeze(-1)
    )
    # Calculate weights for each point along the rays based on the transmittance and alpha values
    weights = trans * alphas

    extras = {
        "weights": weights,
        # the transmittance of the intervals
        "trans": trans,
        # the midpoints of the intervals
        "t_vals": (t_starts + t_ends) / 2.0,
        # the lengths of the intervals
        "t_dist": (t_ends - t_starts),
    }

    for k in [
        # predicted forward flow
        "forward_flow",
        # predicted backward flow
        "backward_flow",
        # the predicted backward flow from the forward-warpped points
        "forward_pred_backward_flow",
        # the predicted forward flow from the backward-warpped points
        "backward_pred_forward_flow",
    ]:
        if k in results:
            extras[k] = results[k]

    # =============== Geometry ================ #
    opacities = accumulate_along_rays(weights, values=None).clamp(1e-6, 1.0)
    # expected depth
    depths = accumulate_along_rays(weights, values=(t_starts + t_ends)[..., None] / 2.0)
    depths = depths / opacities
    # median depth
    steps = (t_starts + t_ends)[..., None] / 2.0
    cumulative_weights = torch.cumsum(weights, dim=-1)  # [..., num_samples]
    # [..., 1]
    split = torch.ones((*weights.shape[:-1], 1), device=weights.device) * 0.5
    # [..., 1]
    median_index = torch.searchsorted(cumulative_weights, split, side="left")
    median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
    median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]

    results_dict = {
        "density": results["density"].squeeze(-1),
        "depth": depths,
        "opacity": opacities,
        "median_depth": median_depth,
    }

    # =========== Geometry Decomposition =========== #
    if "static_density" in results and "dynamic_density" in results:
        extras["static_density"] = results["static_density"]
        extras["dynamic_density"] = results["dynamic_density"]
        # blend static and dynamic densities
        static_ratio = results["static_density"] / (results["density"] + 1e-6)
        dynamic_ratio = results["dynamic_density"] / (results["density"] + 1e-6)
        if return_decomposition:
            (
                static_weights,
                static_opacities,
                static_depths,
            ) = render_weights_opacity_depth_from_density(
                t_starts,
                t_ends,
                results["static_density"],
            )
            results_dict["static_opacity"] = static_opacities
            results_dict["static_depth"] = static_depths

            (
                dynamic_weights,
                dynamic_opacities,
                dynamic_depths,
            ) = render_weights_opacity_depth_from_density(
                t_starts,
                t_ends,
                results["dynamic_density"],
            )
            results_dict["dynamic_opacity"] = dynamic_opacities
            results_dict["dynamic_depth"] = dynamic_depths

    # =========== RGB =========== #
    if "rgb" in results:
        # static-only scene
        results_dict["rgb"] = accumulate_along_rays(weights, values=results["rgb"])
    elif "static_rgb" in results and "dynamic_rgb" in results:
        # default to no shadow
        shadow_ratio = 0.0
        if "shadow_ratio" in results:
            shadow_ratio = results["shadow_ratio"]
            results_dict["shadow_ratio"] = accumulate_along_rays(
                weights,
                values=shadow_ratio.square(),
            )
        rgb = (
            static_ratio[..., None] * results["static_rgb"] * (1 - shadow_ratio)
            + dynamic_ratio[..., None] * results["dynamic_rgb"]
        )
        results_dict["rgb"] = accumulate_along_rays(weights, values=rgb)

        # =========== RGB Decomposition =========== #
        if return_decomposition:
            results_dict["static_rgb"] = accumulate_along_rays(
                static_weights,
                values=results["static_rgb"],
            )
            if "shadow_ratio" in results:
                # shadow reduced static rgb
                results_dict["shadow_reduced_static_rgb"] = accumulate_along_rays(
                    static_weights,
                    values=results["static_rgb"] * (1 - shadow_ratio),
                )
                # shadow-only rgb
                shadow_only_static_rgb = accumulate_along_rays(
                    static_weights,
                    values=results["static_rgb"] * shadow_ratio,
                )
                acc_shadow = accumulate_along_rays(weights, values=shadow_ratio)
                results_dict["shadow_only_static_rgb"] = shadow_only_static_rgb + (
                    1 - acc_shadow
                )
                results_dict["shadow"] = accumulate_along_rays(
                    weights, values=shadow_ratio
                )

            dynamic_rgb = accumulate_along_rays(
                dynamic_weights,
                values=results["dynamic_rgb"],
            )
            results_dict["dynamic_rgb"] = dynamic_rgb
            if "forward_flow" in results:
                # render 2D flows
                results_dict["forward_flow"] = accumulate_along_rays(
                    dynamic_weights,
                    values=results["forward_flow"],
                )
                results_dict["backward_flow"] = accumulate_along_rays(
                    dynamic_weights,
                    values=results["backward_flow"],
                )

    # Sky composition.
    if "rgb_sky" in results:
        results_dict["rgb"] = results_dict["rgb"] + results["rgb_sky"] * (
            1.0 - results_dict["opacity"]
        )
        if "static_rgb" in results_dict:
            # add sky to static rgb
            results_dict["static_rgb"] = results_dict["static_rgb"] + results[
                "rgb_sky"
            ] * (1.0 - results_dict["static_opacity"])

    # =========== features =========== #
    if "dino_feat" in results:
        results_dict["dino_feat"] = accumulate_along_rays(
            weights, values=results["dino_feat"]
        )
        if "dino_sky_feat" in results:
            # dino sky composition
            results_dict["dino_feat"] = results_dict["dino_feat"] + results[
                "dino_sky_feat"
            ] * (1.0 - results_dict["opacity"])
        if "dino_pe" in results:
            # dino positional embedding decomposition
            # pe_free volume-rendered dino features
            results_dict["dino_pe_free"] = results_dict["dino_feat"].clone()
            # pe features
            results_dict["dino_pe"] = results["dino_pe"]
            # add pe to volume-rendered dino features
            results_dict["dino_feat"] = results_dict["dino_feat"] + results["dino_pe"]
    elif "static_dino_feat" in results and "dynamic_dino_feat" in results:
        dino_feat = (
            static_ratio[..., None] * results["static_dino_feat"]
            + dynamic_ratio[..., None] * results["dynamic_dino_feat"]
        )
        results_dict["dino_feat"] = accumulate_along_rays(
            weights,
            values=dino_feat,
        )
        # dino sky
        if "dino_sky_feat" in results:
            # dino sky composition
            results_dict["dino_feat"] = results_dict["dino_feat"] + results[
                "dino_sky_feat"
            ] * (1.0 - results_dict["opacity"])
        if "dino_pe" in results:
            # dino positional embedding decomposition
            # pe_free volume-rendered dino features
            results_dict["dino_pe_free"] = results_dict["dino_feat"].clone()
            # pe features
            results_dict["dino_pe"] = results["dino_pe"]
            # add pe to volume-rendered dino features
            results_dict["dino_feat"] = results_dict["dino_feat"] + results["dino_pe"]

        if return_decomposition:
            results_dict["static_dino"] = accumulate_along_rays(
                static_weights,
                values=results["static_dino_feat"],
            )
            results_dict["dynamic_dino"] = accumulate_along_rays(
                dynamic_weights,
                values=results["dynamic_dino_feat"],
            )
            if "dino_sky_feat" in results:
                # add dino sky to static dino
                results_dict["static_dino"] = results_dict["static_dino"] + results[
                    "dino_sky_feat"
                ] * (1.0 - results_dict["opacity"])

    # also return "extras" for some supervision
    results_dict["extras"] = extras

    return results_dict


def render_rays(
    # scene
    radiance_field: RadianceField = None,
    proposal_estimator: PropNetEstimator = None,
    proposal_networks: Optional[List[DensityField]] = None,
    data_dict: Dict[str, Tensor] = None,
    cfg: OmegaConf = None,
    proposal_requires_grad: bool = False,
    return_decomposition: bool = False,
    prefix="",
) -> Dict[str, Tensor]:
    """Render some attributes of the scene along the rays."""
    # reshape data_dict to be (num_rays, ...)
    rays_shape = data_dict[prefix + "origins"].shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        reshaped_data_dict = {}
        for k, v in data_dict.items():
            reshaped_data_dict[k] = v.reshape(num_rays, -1).squeeze()
    else:
        num_rays, _ = rays_shape
        reshaped_data_dict = data_dict.copy()

    def prop_sigma_fn(t_starts, t_ends, proposal_network):
        # query propsal networks for density
        t_origins = chunk_data_dict[prefix + "origins"][..., None, :]
        t_dirs = chunk_data_dict[prefix + "viewdirs"][..., None, :]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        sub_dict = {
            k: v[..., None].repeat_interleave(t_starts.shape[-1], dim=-1)
            for k, v in chunk_data_dict.items()
            if "time" in k
        }
        return proposal_network(positions, sub_dict)

    def query_fn(t_starts, t_ends):
        # query the final nerf model for density and other information along the rays
        t_origins = chunk_data_dict[prefix + "origins"][..., None, :]
        t_dirs = chunk_data_dict[prefix + "viewdirs"][..., None, :].repeat_interleave(
            t_starts.shape[-1], dim=-2
        )
        sub_dict = {
            k: v[..., None].repeat_interleave(t_starts.shape[-1], dim=-1)
            for k, v in chunk_data_dict.items()
            if k not in [prefix + "viewdirs", prefix + "origins", "pixel_coords"]
        }
        sub_dict["t_starts"], sub_dict["t_ends"] = t_starts, t_ends
        if "pixel_coords" in chunk_data_dict:
            # use this for positional embedding decomposition
            sub_dict["pixel_coords"] = chunk_data_dict["pixel_coords"]
        positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        # return density only when rendering lidar, i.e., no rgb or sky or features are rendered
        results_dict: Dict[str, Tensor] = radiance_field(
            positions, t_dirs, sub_dict, return_density_only=(prefix == "lidar_")
        )
        results_dict["density"] = results_dict["density"].squeeze(-1)
        return results_dict

    results = []
    chunk = 2**24 if radiance_field.training else cfg.render.render_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_data_dict = {k: v[i : i + chunk] for k, v in reshaped_data_dict.items()}
        assert proposal_networks is not None, "proposal_networks is required."
        # obtain proposed intervals
        t_starts, t_ends = proposal_estimator.sampling(
            prop_sigma_fns=[
                lambda *args: prop_sigma_fn(*args, p) for p in proposal_networks
            ],
            num_samples=cfg.nerf.sampling.num_samples,
            prop_samples=cfg.nerf.propnet.num_samples_per_prop,
            n_rays=chunk_data_dict[prefix + "origins"].shape[0],
            near_plane=cfg.nerf.propnet.near_plane,
            far_plane=cfg.nerf.propnet.far_plane,
            sampling_type=cfg.nerf.propnet.sampling_type,
            stratified=radiance_field.training,
            requires_grad=proposal_requires_grad,
        )
        # render the scene
        chunk_results_dict = rendering(
            t_starts,
            t_ends,
            query_fn=query_fn,
            return_decomposition=return_decomposition,
        )
        extras = chunk_results_dict.pop("extras")
        results.append(chunk_results_dict)
    render_results = collate(
        results,
        collate_fn_map={
            **default_collate_fn_map,
            Tensor: lambda x, **_: torch.cat(x, 0),
        },
    )
    extras["density"] = render_results.pop("density")
    for k, v in render_results.items():
        # recover the original shape
        render_results[k] = v.reshape(list(rays_shape[:-1]) + list(v.shape[1:]))
    render_results["extras"] = extras
    return render_results
