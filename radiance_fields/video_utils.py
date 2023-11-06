import logging
import os
from typing import Callable, Dict, List, Optional

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity as ssim
from torch import Tensor
from tqdm import tqdm, trange

from datasets.base import SplitWrapper
from datasets.metrics import compute_psnr
from radiance_fields.radiance_field import DensityField, RadianceField
from radiance_fields.render_utils import render_rays
from third_party.nerfacc_prop_net import PropNetEstimator
from utils.misc import get_robust_pca
from utils.visualization_tools import (
    resize_five_views,
    scene_flow_to_rgb,
    to8b,
    visualize_depth,
)

logger = logging.getLogger()

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)
flow_visualizer = (
    lambda frame: scene_flow_to_rgb(
        frame,
        background="bright",
        flow_max_radius=1.0,
    )
    .cpu()
    .numpy()
)
get_numpy: Callable[[Tensor], np.ndarray] = lambda x: x.squeeze().cpu().numpy()
non_zero_mean: Callable[[Tensor], float] = (
    lambda x: sum(x) / len(x) if len(x) > 0 else -1
)


def render_pixels(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_estimator: PropNetEstimator,
    dataset: SplitWrapper,
    proposal_networks: Optional[List[DensityField]] = None,
    compute_metrics: bool = False,
    vis_indices: Optional[List[int]] = None,
    return_decomposition: bool = True,
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
        return_decomposition (bool, optional): Whether to visualize the static-dynamic decomposition. Defaults to True.
    """
    model.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()
    if proposal_estimator is not None:
        proposal_estimator.eval()
    # set up render function
    render_func = lambda data_dict: render_rays(
        radiance_field=model,
        proposal_estimator=proposal_estimator,
        proposal_networks=proposal_networks,
        data_dict=data_dict,
        cfg=cfg,
        return_decomposition=return_decomposition,  # return static-dynamic decomposition
    )
    render_results = render(
        dataset,
        render_func,
        model=model,
        compute_metrics=compute_metrics,
        vis_indices=vis_indices,
    )
    if compute_metrics:
        num_samples = len(dataset) if vis_indices is None else len(vis_indices)
        logger.info(f"Eval over {num_samples} images:")
        logger.info(f"\tPSNR: {render_results['psnr']:.4f}")
        logger.info(f"\tSSIM: {render_results['ssim']:.4f}")
        logger.info(f"\tFeature PSNR: {render_results['feat_psnr']:.4f}")
        logger.info(f"\tMasked PSNR: {render_results['masked_psnr']:.4f}")
        logger.info(f"\tMasked SSIM: {render_results['masked_ssim']:.4f}")
        logger.info(f"\tMasked Feature PSNR: {render_results['masked_feat_psnr']:.4f}")

    return render_results


def render(
    dataset: SplitWrapper,
    render_func: Callable,
    model: Optional[RadianceField] = None,
    compute_metrics: bool = False,
    vis_indices: Optional[List[int]] = None,
):
    """
    Renders a dataset utilizing a specified render function.
    For efficiency and space-saving reasons, this function doesn't store the original features; instead, it keeps
    the colors reduced via PCA.
    TODO: clean up this function

    Parameters:
        dataset: Dataset to render.
        render_func: Callable function used for rendering the dataset.
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs = [], []
    static_rgbs, dynamic_rgbs = [], []
    shadow_reduced_static_rgbs, shadow_only_static_rgbs = [], []

    # depths
    depths, median_depths = [], []
    static_depths, static_opacities = [], []
    dynamic_depths, dynamic_opacities = [], []

    # sky
    opacities, sky_masks = [], []

    # features
    pred_dinos, gt_dinos = [], []
    pred_dinos_pe_free, pred_dino_pe = [], []
    static_dinos, dynamic_dinos = [], []  # should we also render this?

    # cross-rendering results
    dynamic_dino_on_static_rgbs, dynamic_rgb_on_static_dinos = [], []

    # flows
    forward_flows, backward_flows = [], []

    if compute_metrics:
        psnrs, ssim_scores, feat_psnrs = [], [], []
        masked_psnrs, masked_ssims = [], []
        masked_feat_psnrs = []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        computed = False
        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            data_dict = dataset[i]
            for k, v in data_dict.items():
                if isinstance(v, Tensor):
                    data_dict[k] = v.cuda(non_blocking=True)
            results = render_func(data_dict)
            # ------------- rgb ------------- #
            rgb = results["rgb"]
            rgbs.append(get_numpy(rgb))
            if "pixels" in data_dict:
                gt_rgbs.append(get_numpy(data_dict["pixels"]))
            if "static_rgb" in results:
                static_rgbs.append(get_numpy(results["static_rgb"]))
            if "dynamic_rgb" in results:
                # green screen blending for better visualization
                green_background = torch.tensor([0.0, 177, 64]) / 255.0
                green_background = green_background.to(results["dynamic_rgb"].device)
                dy_rgb = results["dynamic_rgb"] * results[
                    "dynamic_opacity"
                ] + green_background * (1 - results["dynamic_opacity"])
                dynamic_rgbs.append(get_numpy(dy_rgb))
            if "shadow_reduced_static_rgb" in results:
                shadow_reduced_static_rgbs.append(
                    get_numpy(results["shadow_reduced_static_rgb"])
                )
            if "shadow_only_static_rgb" in results:
                shadow_only_static_rgbs.append(
                    get_numpy(results["shadow_only_static_rgb"])
                )
            if "forward_flow" in results:
                forward_flows.append(flow_visualizer(results["forward_flow"]))
            if "backward_flow" in results:
                backward_flows.append(flow_visualizer(results["backward_flow"]))
            # ------------- depth ------------- #
            depth = results["depth"]
            depths.append(get_numpy(depth))
            # ------------- opacity ------------- #
            opacities.append(get_numpy(results["opacity"]))
            if "static_depth" in results:
                static_depths.append(get_numpy(results["static_depth"]))
                static_opacities.append(get_numpy(results["static_opacity"]))
            if "dynamic_depth" in results:
                dynamic_depths.append(get_numpy(results["dynamic_depth"]))
                dynamic_opacities.append(get_numpy(results["dynamic_opacity"]))
            elif "median_depth" in results:
                median_depths.append(get_numpy(results["median_depth"]))
            # -------- sky -------- #
            if "sky_masks" in data_dict:
                sky_masks.append(get_numpy(data_dict["sky_masks"]))

            if compute_metrics:
                psnrs.append(compute_psnr(rgb, data_dict["pixels"]))
                ssim_scores.append(
                    ssim(
                        get_numpy(rgb),
                        get_numpy(data_dict["pixels"]),
                        data_range=1.0,
                        channel_axis=-1,
                    )
                )
                if "dynamic_masks" in data_dict:
                    dynamic_mask = get_numpy(data_dict["dynamic_masks"]).astype(bool)
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], data_dict["pixels"][dynamic_mask]
                            )
                        )
                        masked_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(data_dict["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )

            # -------------- dino ------------- #
            if "dino_feat" in results:
                pred_dino_feat = results["dino_feat"]
                if "features" in data_dict:
                    gt_dino_feat = data_dict["features"]
                    if compute_metrics:
                        feat_psnrs.append(compute_psnr(pred_dino_feat, gt_dino_feat))
                        if "dynamic_masks" in data_dict:
                            dynamic_mask = data_dict["dynamic_masks"].squeeze().bool()
                            if dynamic_mask.sum() > 0:
                                masked_dino_error = compute_psnr(
                                    pred_dino_feat[dynamic_mask],
                                    gt_dino_feat[dynamic_mask],
                                )
                                masked_feat_psnrs.append(masked_dino_error)

                else:
                    gt_dino_feat = None
                # === use the registered pca matrix to get the colors
                pred_dino_feat = pred_dino_feat @ model.feats_reduction_mat.to(
                    pred_dino_feat
                )
                pred_dino_feat = (
                    pred_dino_feat - model.feat_color_min.to(pred_dino_feat)
                ) / (
                    model.feat_color_max.to(pred_dino_feat)
                    - model.feat_color_min.to(pred_dino_feat)
                )
                if gt_dino_feat is not None:
                    # visualize gt_dino_feat
                    gt_dino_feat = gt_dino_feat @ model.feats_reduction_mat.to(
                        pred_dino_feat
                    )
                    gt_dino_feat = (
                        gt_dino_feat - model.feat_color_min.to(pred_dino_feat)
                    ) / (
                        model.feat_color_max.to(pred_dino_feat)
                        - model.feat_color_min.to(pred_dino_feat)
                    )
                if "dino_pe_free" in results:
                    if not computed:
                        # need to compute new pca matrix for dino_pe_free
                        # because they're not longer aligned with the original dino_feat
                        computed = True
                        non_sky_dino_pe_free = results["dino_pe_free"] * (
                            ~data_dict["sky_masks"].bool().unsqueeze(-1)
                        ).to(results["dino_pe_free"])
                        (
                            dino_pe_free_reduction_mat,
                            dino_pe_free_color_min,
                            dino_pe_free_color_max,
                        ) = get_robust_pca(
                            non_sky_dino_pe_free.reshape(
                                -1,
                                results["dino_pe_free"].shape[-1],
                            ),
                            m=2.5,
                        )
                        # compute pca matrix for position embedding patterns
                        (
                            pe_reduction_mat,
                            pe_color_min,
                            pe_color_max,
                        ) = get_robust_pca(
                            results["dino_pe"].reshape(
                                -1, results["dino_pe"].shape[-1]
                            ),
                            m=2.5,
                        )
                    dino_pe_free = results["dino_pe_free"] @ dino_pe_free_reduction_mat
                    dino_pe_free = (
                        dino_pe_free - dino_pe_free_color_min.to(pred_dino_feat)
                    ) / (
                        dino_pe_free_color_max.to(pred_dino_feat)
                        - dino_pe_free_color_min.to(pred_dino_feat)
                    )
                    dino_pe_free = torch.clamp(dino_pe_free, 0, 1)
                    # optionally mask out sky for visualization
                    dino_pe_free *= results["opacity"]
                    pred_dinos_pe_free.append(get_numpy(dino_pe_free))

                    dino_pe = results["dino_pe"] @ pe_reduction_mat
                    dino_pe = (dino_pe - pe_color_min.to(pred_dino_feat)) / (
                        pe_color_max - pe_color_min
                    ).to(pred_dino_feat)
                    dino_pe = torch.clamp(dino_pe, 0, 1)
                    pred_dino_pe.append(get_numpy(dino_pe))

                    if "static_dino" in results:
                        static_dino_feat = (
                            results["static_dino"] @ dino_pe_free_reduction_mat
                        )
                        static_dino_feat = (
                            static_dino_feat - dino_pe_free_color_min.to(pred_dino_feat)
                        ) / (dino_pe_free_color_max - dino_pe_free_color_min).to(
                            pred_dino_feat
                        )
                        static_dino_feat = torch.clamp(static_dino_feat, 0, 1)
                        # dino_pe_free *= results["opacity"]
                        static_dinos.append(get_numpy(static_dino_feat))
                        # get dynamic_rgb on static_dino
                        dynamic_rgb_on_static_dino = get_numpy(
                            results["dynamic_rgb"]
                        ) * dynamic_opacities[-1][..., None] + static_dinos[-1] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_rgb_on_static_dino = np.clip(
                            dynamic_rgb_on_static_dino, 0, 1
                        )
                        dynamic_rgb_on_static_dinos.append(dynamic_rgb_on_static_dino)

                    if "dynamic_dino" in results:
                        dynamic_dino_feat = (
                            results["dynamic_dino"] @ dino_pe_free_reduction_mat
                        )
                        dynamic_dino_feat = (
                            dynamic_dino_feat
                            - dino_pe_free_color_min.to(pred_dino_feat)
                        ) / (dino_pe_free_color_max - dino_pe_free_color_min).to(
                            pred_dino_feat
                        )
                        dynamic_dino_feat = torch.clamp(dynamic_dino_feat, 0, 1)
                        dynamic_dino_feat *= results["dynamic_opacity"]
                        dynamic_dinos.append(get_numpy(dynamic_dino_feat))
                        # get dynamic_dino on static_rgb
                        dynamic_dino_on_static_rgb = dynamic_dinos[
                            -1
                        ] * dynamic_opacities[-1][..., None] + static_rgbs[-1] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_dino_on_static_rgb = np.clip(
                            dynamic_dino_on_static_rgb, 0, 1
                        )
                        dynamic_dino_on_static_rgbs.append(dynamic_dino_on_static_rgb)
                else:
                    if "static_dino" in results:
                        static_dino_feat = results[
                            "static_dino"
                        ] @ model.feats_reduction_mat.to(pred_dino_feat)
                        static_dino_feat = (
                            static_dino_feat - model.feat_color_min.to(pred_dino_feat)
                        ) / (
                            model.feat_color_max.to(pred_dino_feat)
                            - model.feat_color_min.to(pred_dino_feat)
                        )
                        static_dino_feat = torch.clamp(static_dino_feat, 0, 1)
                        static_dinos.append(get_numpy(static_dino_feat))
                        # get dynamic_rgb on static_dino
                        dynamic_rgb_on_static_dino = get_numpy(
                            results["dynamic_rgb"]
                        ) * dynamic_opacities[-1][..., None] + static_dinos[-1] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_rgb_on_static_dino = np.clip(
                            dynamic_rgb_on_static_dino, 0, 1
                        )
                        dynamic_rgb_on_static_dinos.append(dynamic_rgb_on_static_dino)
                    if "dynamic_dino" in results:
                        dynamic_dino_feat = (
                            results["dynamic_dino"] @ model.feats_reduction_mat
                        )
                        dynamic_dino_feat = (
                            dynamic_dino_feat - model.feat_color_min.to(pred_dino_feat)
                        ) / (
                            model.feat_color_max.to(pred_dino_feat)
                            - model.feat_color_min.to(pred_dino_feat)
                        )
                        dynamic_dino_feat = torch.clamp(dynamic_dino_feat, 0, 1)
                        dynamic_dinos.append(get_numpy(dynamic_dino_feat))
                        # get dynamic_dino on static_rgb
                        dynamic_dino_on_static_rgb = dynamic_dinos[
                            -1
                        ] * dynamic_opacities[-1][..., None] + static_rgbs[-1] * (
                            1 - dynamic_opacities[-1][..., None]
                        )
                        dynamic_dino_on_static_rgb = np.clip(
                            dynamic_dino_on_static_rgb, 0, 1
                        )
                        dynamic_dino_on_static_rgbs.append(dynamic_dino_on_static_rgb)

                pred_dino_feat = torch.clamp(pred_dino_feat, 0, 1)
                # pred_dino_feat *= results["opacity"]
                pred_dinos.append(get_numpy(pred_dino_feat))
                if gt_dino_feat is not None:
                    gt_dino_feat = torch.clamp(gt_dino_feat, 0, 1)
                    gt_dinos.append(get_numpy(gt_dino_feat))
    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["feat_psnr"] = non_zero_mean(feat_psnrs) if compute_metrics else -1
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    results_dict["masked_feat_psnr"] = (
        non_zero_mean(masked_feat_psnrs) if compute_metrics else -1
    )
    results_dict["rgbs"] = rgbs
    results_dict["static_rgbs"] = static_rgbs
    results_dict["dynamic_rgbs"] = dynamic_rgbs
    results_dict["depths"] = depths
    results_dict["opacities"] = opacities
    results_dict["static_depths"] = static_depths
    results_dict["static_opacities"] = static_opacities
    results_dict["dynamic_depths"] = dynamic_depths
    results_dict["dynamic_opacities"] = dynamic_opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(pred_dinos) > 0:
        results_dict["dino_feats"] = pred_dinos
    if len(gt_dinos) > 0:
        results_dict["gt_dino_feats"] = gt_dinos
    if len(pred_dinos_pe_free) > 0:
        results_dict["dino_feats_pe_free"] = pred_dinos_pe_free
    if len(pred_dino_pe) > 0:
        results_dict["dino_pe"] = pred_dino_pe
    if len(static_dinos) > 0:
        results_dict["static_dino_feats"] = static_dinos
    if len(dynamic_dinos) > 0:
        results_dict["dynamic_dino_feats"] = dynamic_dinos
    if len(dynamic_dino_on_static_rgbs) > 0:
        results_dict["dynamic_dino_on_static_rgbs"] = dynamic_dino_on_static_rgbs
    if len(dynamic_rgb_on_static_dinos) > 0:
        results_dict["dynamic_rgb_on_static_dinos"] = dynamic_rgb_on_static_dinos
    if len(shadow_reduced_static_rgbs) > 0:
        results_dict["shadow_reduced_static_rgbs"] = shadow_reduced_static_rgbs
    if len(shadow_only_static_rgbs) > 0:
        results_dict["shadow_only_static_rgbs"] = shadow_only_static_rgbs
    if len(forward_flows) > 0:
        results_dict["forward_flows"] = forward_flows
    if len(backward_flows) > 0:
        results_dict["backward_flows"] = backward_flows
    if len(median_depths) > 0:
        results_dict["median_depths"] = median_depths
    return results_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        for key in keys:
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            frames = resize_five_views(frames)
            frames = np.concatenate(frames, axis=1)
            merged_list.append(frames)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        logger.info(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if key not in render_results or len(render_results[key]) == 0:
            continue
        for i in range(num_timestamps):
            if key == "sky_masks":
                frames = render_results["opacities"][i * num_cams : (i + 1) * num_cams]
            else:
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif key == "sky_masks":
                frames = [
                    1 - np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                opacities = render_results[key.replace("depths", "opacities")][
                    i * num_cams : (i + 1) * num_cams
                ]
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            frames = resize_five_views(frames)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"_{i*3 + j:03d}.png"),
                        to8b(frame),
                    )
            frames = to8b(np.concatenate(frames, axis=1))
            writer.append_data(frames)
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        del writer
        if verbose:
            logger.info(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict
