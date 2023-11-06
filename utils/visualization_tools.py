import logging
import os
from collections import namedtuple
from itertools import accumulate
from typing import List, Optional, Union

import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from scipy import ndimage
from tqdm import tqdm
import json

from datasets import SceneDataset
from datasets.utils import voxel_coords_to_world_coords, world_coords_to_voxel_coords
from radiance_fields import DensityField, RadianceField
from radiance_fields.render_utils import render_rays
from third_party.nerfacc_prop_net import PropNetEstimator
from utils.misc import get_robust_pca
from utils.misc import NumpyEncoder

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

logger = logging.getLogger()
turbo_cmap = cm.get_cmap("turbo")


def to8b(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def resize_five_views(imgs: np.array):
    if len(imgs) != 5:
        return imgs
    for idx in [0, -1]:
        img = imgs[idx]
        new_shape = [int(img.shape[1] * 0.46), img.shape[1], 3]
        new_img = np.zeros_like(img)
        new_img[-new_shape[0] :, : new_shape[1], :] = ndimage.zoom(
            img, [new_shape[0] / img.shape[0], new_shape[1] / img.shape[1], 1]
        )
        # clip the image to 0-1
        new_img = np.clip(new_img, 0, 1)
        imgs[idx] = new_img
    return imgs


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.
    from mipnerf

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    if lo is None or hi is None:
        lo_auto, hi_auto = weighted_percentile(
            value, weight, [50 - percentile / 2, 50 + percentile / 2]
        )
        # If `lo` or `hi` are None, use the automatically-computed bounds above.
        eps = np.finfo(np.float32).eps
        lo = lo or (lo_auto - eps)
        hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
        )
    if weight is not None:
        value *= weight
    else:
        weight = np.ones_like(value)
    if colormap:
        colorized = colormap(value)[..., :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_depth(
    x, acc=None, lo=None, hi=None, depth_curve_fn=lambda x: -np.log(x + 1e-6)
):
    """Visualizes depth maps."""
    return visualize_cmap(
        x,
        acc,
        cm.get_cmap("turbo"),
        curve_fn=depth_curve_fn,
        lo=lo,
        hi=hi,
        matte_background=False,
    )


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> torch.Tensor:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return torch.FloatTensor(colorwheel)


WHEEL = _make_colorwheel()
N_COLS = len(WHEEL)
WHEEL = torch.vstack((WHEEL, WHEEL[0]))  # Make the wheel cyclic for interpolation


def scene_flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "dark",
) -> torch.Tensor:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)
    if flow_max_radius is None:
        # flow_max_radius = torch.max(radius)
        flow_max_radius = torch.quantile(radius, 0.99)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((N_COLS - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = (
        torch.fmod(angle, 1),
        angle.trunc(),
        torch.ceil(angle),
    )
    angle_fractional = angle_fractional.unsqueeze(-1)
    wheel = WHEEL.to(angle_floor.device)
    float_hue = (
        wheel[angle_floor.long()] * (1 - angle_fractional)
        + wheel[angle_ceil.long()] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs",
        ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"],
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors.unsqueeze(-1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - factors.unsqueeze(-1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, torch.FloatTensor([255, 255, 255])
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, torch.zeros(3)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    return colors / 255.0


def vis_occ_plotly(
    vis_aabb: List[Union[int, float]],
    coords: np.array = None,
    colors: np.array = None,
    dynamic_coords: List[np.array] = None,
    dynamic_colors: List[np.array] = None,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
    z_ratio: float = 0.125,
    size: int = 5,
    black_bg: bool = False,
    title: str = None,
) -> go.Figure:  # type: ignore
    fig = go.Figure()  # start with an empty figure

    if coords is not None:
        # Add static trace
        static_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=colors,
                symbol="square",
            ),
        )
        fig.add_trace(static_trace)

    # Add temporal traces
    if dynamic_coords is not None:
        for i in range(len(dynamic_coords)):
            fig.add_trace(
                go.Scatter3d(
                    x=dynamic_coords[i][:, 0],
                    y=dynamic_coords[i][:, 1],
                    z=dynamic_coords[i][:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=dynamic_colors[i],
                        symbol="diamond",
                    ),
                )
            )
        steps = []
        if coords is not None:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords) + 1),
                    ],  # Include the static trace
                    label=f"Second {i}",
                )
                step["args"][1][0] = True  # Make the static trace always visible
                step["args"][1][i + 1] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)
        else:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords)),
                    ],
                    label=f"Second {i}",
                )
                step["args"][1][i] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)

        sliders = [
            dict(
                active=0,
                pad={"t": 1},
                steps=steps,
                font=dict(color="white") if black_bg else {},  # Update for font color
            )
        ]
        fig.update_layout(sliders=sliders)
    title_font_color = "white" if black_bg else "black"
    if not black_bg:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            paper_bgcolor="black",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    eye = np.array([-1, 0, 0.5])
    eye = eye.tolist()
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        ),
    )
    return fig


def visualize_voxels(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_estimator: PropNetEstimator = None,
    proposal_networks: DensityField = None,
    dataset: SceneDataset = None,
    device: str = "cuda",
    save_html: bool = True,
    is_dynamic: bool = False,
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    if proposal_estimator is not None:
        proposal_estimator.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()

    vis_voxel_aabb = torch.tensor(model.aabb, device=device)
    # slightly expand the aabb to make sure all points are covered
    vis_voxel_aabb[1:3] -= 1
    vis_voxel_aabb[3:] += 1
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    aabb_length = aabb_max - aabb_min

    # compute the voxel resolution for visualization
    static_voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.render.vis_voxel_size
    ).long()
    empty_static_voxels = torch.zeros(*static_voxel_resolution, device=device)
    if is_dynamic:
        # use a slightly smaller voxel size for dynamic voxels
        dynamic_voxel_resolution = torch.ceil(
            (aabb_max - aabb_min) / cfg.render.vis_voxel_size * 0.8
        ).long()
        all_occupied_dynamic_points = []
        empty_dynamic_voxels = torch.zeros(*dynamic_voxel_resolution, device=device)

    # collect some patches for PCA
    to_compute_pca_patches = []

    pbar = tqdm(
        dataset.full_pixel_set,
        desc="querying depth",
        dynamic_ncols=True,
        total=len(dataset.full_pixel_set),
    )
    for i, data_dict in enumerate(pbar):
        data_dict = dataset.full_pixel_set[i]
        for k, v in data_dict.items():
            data_dict[k] = v.to(device)
        if i < dataset.num_cams:
            # collect all patches from the first timestep
            with torch.no_grad():
                render_results = render_rays(
                    radiance_field=model,
                    proposal_estimator=proposal_estimator,
                    proposal_networks=proposal_networks,
                    data_dict=data_dict,
                    cfg=cfg,
                    proposal_requires_grad=False,
                )
            if "dino_pe_free" in render_results:
                dino_feats = render_results["dino_pe_free"]
            else:
                dino_feats = render_results["dino_feat"]
            dino_feats = dino_feats.reshape(-1, dino_feats.shape[-1])
            to_compute_pca_patches.append(dino_feats)
        # query the depth. we force a lidar mode here so that the renderer will skip
        # querying other features such as colors, features, etc.
        data_dict["lidar_origins"] = data_dict["origins"].to(device)
        data_dict["lidar_viewdirs"] = data_dict["viewdirs"].to(device)
        data_dict["lidar_normed_timestamps"] = data_dict["normed_timestamps"].to(device)
        with torch.no_grad():
            render_results = render_rays(
                radiance_field=model,
                proposal_estimator=proposal_estimator,
                proposal_networks=proposal_networks,
                data_dict=data_dict,
                cfg=cfg,
                proposal_requires_grad=False,
                prefix="lidar_",  # force lidar mode
                return_decomposition=True,
            )
        # ==== get the static voxels ======
        if is_dynamic:
            static_depth = render_results["static_depth"]
        else:
            static_depth = render_results["depth"]
        world_coords = (
            data_dict["lidar_origins"] + data_dict["lidar_viewdirs"] * static_depth
        )
        world_coords = world_coords[static_depth.squeeze() < 80]
        voxel_coords = world_coords_to_voxel_coords(
            world_coords, aabb_min, aabb_max, static_voxel_resolution
        )
        voxel_coords = voxel_coords.long()
        selector = (
            (voxel_coords[..., 0] >= 0)
            & (voxel_coords[..., 0] < static_voxel_resolution[0])
            & (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < static_voxel_resolution[1])
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < static_voxel_resolution[2])
        )
        # split the voxel_coords into separate dimensions
        voxel_coords_x = voxel_coords[..., 0][selector]
        voxel_coords_y = voxel_coords[..., 1][selector]
        voxel_coords_z = voxel_coords[..., 2][selector]
        # index into empty_voxels using the separated coordinates
        empty_static_voxels[voxel_coords_x, voxel_coords_y, voxel_coords_z] = 1

        # ==== get the dynamic voxels ======
        if is_dynamic:
            dynamic_depth = render_results["dynamic_depth"]
            world_coords = (
                data_dict["lidar_origins"] + data_dict["lidar_viewdirs"] * dynamic_depth
            )
            voxel_coords = world_coords_to_voxel_coords(
                world_coords, aabb_min, aabb_max, dynamic_voxel_resolution
            )
            voxel_coords = voxel_coords.long()
            selector = (
                (voxel_coords[..., 0] >= 0)
                & (voxel_coords[..., 0] < dynamic_voxel_resolution[0])
                & (voxel_coords[..., 1] >= 0)
                & (voxel_coords[..., 1] < dynamic_voxel_resolution[1])
                & (voxel_coords[..., 2] >= 0)
                & (voxel_coords[..., 2] < dynamic_voxel_resolution[2])
            )
            # split the voxel_coords into separate dimensions
            voxel_coords_x = voxel_coords[..., 0][selector]
            voxel_coords_y = voxel_coords[..., 1][selector]
            voxel_coords_z = voxel_coords[..., 2][selector]
            # index into empty_voxels using the separated coordinates
            empty_dynamic_voxels[voxel_coords_x, voxel_coords_y, voxel_coords_z] = 1
            if i % dataset.num_cams == 0 and i > 0:
                all_occupied_dynamic_points.append(
                    voxel_coords_to_world_coords(
                        aabb_min,
                        aabb_max,
                        dynamic_voxel_resolution,
                        torch.nonzero(empty_dynamic_voxels),
                    )
                )
                empty_dynamic_voxels = torch.zeros(
                    *dynamic_voxel_resolution, device=device
                )
    # compute the pca reduction
    dummy_pca_reduction, color_min, color_max = get_robust_pca(
        torch.cat(to_compute_pca_patches, dim=0).to(device), m=2.5
    )
    # now let's query the features
    all_occupied_static_points = voxel_coords_to_world_coords(
        aabb_min, aabb_max, static_voxel_resolution, torch.nonzero(empty_static_voxels)
    )
    chunk = 2**18
    pca_colors = []
    occupied_points = []
    pbar = tqdm(
        range(0, all_occupied_static_points.shape[0], chunk),
        desc="querying static features",
        dynamic_ncols=True,
    )
    for i in pbar:
        occupied_points_chunk = all_occupied_static_points[i : i + chunk]
        density_list = []
        # we need to accumulate the density from all proposal networks as well
        # to ensure reliable density estimation
        for p in proposal_networks:
            density_list.append(p(occupied_points_chunk)["density"].squeeze(-1))
        with torch.no_grad():
            results = model.forward(
                occupied_points_chunk,
                query_feature_head=False,
            )
        density_list.append(results["density"])
        density = torch.stack(density_list, dim=0)
        density = torch.mean(density, dim=0)
        # use a preset threshold to determine whether a voxel is occupied
        selector = density > 0.5
        occupied_points_chunk = occupied_points_chunk[selector]
        if len(occupied_points_chunk) == 0:
            # skip if no occupied points in this chunk
            continue
        with torch.no_grad():
            feats = model.forward(
                occupied_points_chunk,
                query_feature_head=True,
                query_pe_head=False,
            )["dino_feat"]
        colors = feats @ dummy_pca_reduction
        del feats
        colors = (colors - color_min) / (color_max - color_min)
        pca_colors.append(torch.clamp(colors, 0, 1))
        occupied_points.append(occupied_points_chunk)

    pca_colors = torch.cat(pca_colors, dim=0)
    occupied_points = torch.cat(occupied_points, dim=0)
    if is_dynamic:
        dynamic_pca_colors = []
        dynamic_occupied_points = []
        unq_timestamps = dataset.pixel_source.unique_normalized_timestamps.to(device)
        # query every 10 frames
        pbar = tqdm(
            range(0, len(all_occupied_dynamic_points), 10),
            desc="querying dynamic fields",
            dynamic_ncols=True,
        )
        for i in pbar:
            occupied_points_chunk = all_occupied_dynamic_points[i]
            normed_timestamps = unq_timestamps[i].repeat(
                occupied_points_chunk.shape[0], 1
            )
            with torch.no_grad():
                results = model.forward(
                    occupied_points_chunk,
                    data_dict={"normed_timestamps": normed_timestamps},
                    query_feature_head=False,
                )
            selector = results["dynamic_density"].squeeze() > 0.1
            occupied_points_chunk = occupied_points_chunk[selector]
            if len(occupied_points_chunk) == 0:
                continue
            # query some features
            normed_timestamps = unq_timestamps[i].repeat(
                occupied_points_chunk.shape[0], 1
            )
            with torch.no_grad():
                feats = model.forward(
                    occupied_points_chunk,
                    data_dict={"normed_timestamps": normed_timestamps},
                    query_feature_head=True,
                    query_pe_head=False,
                )["dynamic_dino_feat"]
            colors = feats @ dummy_pca_reduction
            del feats
            colors = (colors - color_min) / (color_max - color_min)
            dynamic_pca_colors.append(torch.clamp(colors, 0, 1))
            dynamic_occupied_points.append(occupied_points_chunk)
        dynamic_coords = [x.cpu().numpy() for x in dynamic_occupied_points]
        dynamic_colors = [x.cpu().numpy() for x in dynamic_pca_colors]
    else:
        dynamic_coords = None
        dynamic_colors = None

    figure = vis_occ_plotly(
        vis_aabb=vis_voxel_aabb.cpu().numpy().tolist(),
        coords=occupied_points.cpu().numpy(),
        colors=pca_colors.cpu().numpy(),
        dynamic_coords=dynamic_coords,
        dynamic_colors=dynamic_colors,
        x_ratio=1,
        y_ratio=(aabb_length[1] / aabb_length[0]).item(),
        z_ratio=(aabb_length[2] / aabb_length[0]).item(),
        size=3,
        black_bg=True,
        title=f"Lifted {cfg.data.pixel_source.feature_model_type} Features, PE_removed: {cfg.nerf.model.head.enable_learnable_pe}",
    )
    # for plotly
    data = figure.to_dict()["data"]
    layout = figure.to_dict()["layout"]
    output_path = os.path.join(cfg.log_dir, f"feature_field.json")
    with open(output_path, "w") as f:
        json.dump({"data": data, "layout": layout}, f, cls=NumpyEncoder)
    logger.info(f"Saved to {output_path}")
    output_path = os.path.join(cfg.log_dir, f"feature_field.html")
    if save_html:
        figure.write_html(output_path)
        logger.info(f"Query result saved to {output_path}")


def visualize_scene_flow(
    cfg: OmegaConf,
    model: RadianceField,
    dataset: SceneDataset = None,
    device: str = "cuda",
    save_html: bool = True,
):
    pbar = tqdm(
        range(0, len(dataset.full_lidar_set) - 1, 10),
        desc="querying flow",
        dynamic_ncols=True,
    )
    predicted_flow_colors, gt_flow_colors = [], []
    dynamic_coords = []
    for i in pbar:
        data_dict = dataset.full_lidar_set[i].copy()
        lidar_flow_class = data_dict["lidar_flow_class"]
        for k, v in data_dict.items():
            # remove invalid flow (the information is from GT)
            data_dict[k] = v[lidar_flow_class != -1]

        if data_dict[k].shape[0] == 0:
            logger.info(f"no valid points, skipping...")
            continue
        # filter out ground points
        # for k, v in data_dict.items():
        #     data_dict[k] = v[~data_dict["lidar_ground"]]
        valid_lidar_mask = dataset.get_valid_lidar_mask(i, data_dict)
        for k, v in data_dict.items():
            data_dict[k] = v[valid_lidar_mask]
        lidar_points = (
            data_dict["lidar_origins"]
            + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
        )
        normalized_timestamps = data_dict["lidar_normed_timestamps"]
        with torch.no_grad():
            pred_results = model.query_flow(
                positions=lidar_points,
                normed_timestamps=normalized_timestamps,
            )
        pred_flow = pred_results["forward_flow"]
        # flow is only valid when the point is not static
        pred_flow[pred_results["dynamic_density"] < 0.2] *= 0

        predicted_flow_colors.append(
            scene_flow_to_rgb(pred_flow, flow_max_radius=2.0, background="bright")
            .cpu()
            .numpy()
        )
        gt_flow_colors.append(
            scene_flow_to_rgb(
                data_dict["lidar_flow"], flow_max_radius=2.0, background="bright"
            )
            .cpu()
            .numpy()
        )
        dynamic_coords.append(lidar_points.cpu().numpy())

    vis_voxel_aabb = torch.tensor(model.aabb, device=device)
    # slightly expand the aabb to make sure all points are covered
    vis_voxel_aabb[1:3] -= 1
    vis_voxel_aabb[3:] += 1
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    aabb_length = aabb_max - aabb_min
    pred_figure = vis_occ_plotly(
        vis_aabb=vis_voxel_aabb.cpu().numpy().tolist(),
        dynamic_coords=dynamic_coords,
        dynamic_colors=predicted_flow_colors,
        x_ratio=1,
        y_ratio=(aabb_length[1] / aabb_length[0]).item(),
        z_ratio=(aabb_length[2] / aabb_length[0]).item(),
        size=2,
        black_bg=True,
        title=f"Predicted Flow",
    )
    gt_figure = vis_occ_plotly(
        vis_aabb=vis_voxel_aabb.cpu().numpy().tolist(),
        dynamic_coords=dynamic_coords,
        dynamic_colors=gt_flow_colors,
        x_ratio=1,
        y_ratio=(aabb_length[1] / aabb_length[0]).item(),
        z_ratio=(aabb_length[2] / aabb_length[0]).item(),
        size=2,
        black_bg=True,
        title=f"GT Flow",
    )
    if save_html:
        output_path = os.path.join(cfg.log_dir, f"predicted_flow.html")
        pred_figure.write_html(output_path)
        logger.info(f"Predicted flow result saved to {output_path}")
        output_path = os.path.join(cfg.log_dir, f"gt_flow.html")
        gt_figure.write_html(output_path)
        logger.info(f"GT flow saved to {output_path}")
