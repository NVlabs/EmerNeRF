import itertools
import logging
from typing import List, Tuple

import torch
from omegaconf import OmegaConf

from datasets.base import SceneDataset
from radiance_fields import (
    DensityField,
    RadianceField,
    build_density_field,
    build_radiance_field_from_cfg,
)
from third_party.nerfacc_prop_net import PropNetEstimator

logger = logging.getLogger()


def build_model_from_cfg(
    cfg: OmegaConf,
    dataset: SceneDataset,
    device: torch.device = torch.device("cpu"),
) -> RadianceField:
    cfg.num_train_timesteps = dataset.num_train_timesteps
    if dataset.test_pixel_set is not None:
        if cfg.head.enable_img_embedding:
            cfg.head.enable_cam_embedding = True
            cfg.head.enable_img_embedding = False
            logger.info(
                "Overriding enable_img_embedding to False because we have a test set."
            )
    model = build_radiance_field_from_cfg(cfg)
    model.register_normalized_training_timesteps(
        dataset.unique_normalized_training_timestamps,
        time_diff=1 / dataset.num_img_timesteps,
    )
    if dataset.aabb is not None and cfg.resume_from is None:
        model.set_aabb(dataset.aabb)
    if dataset.pixel_source.features is not None and cfg.head.enable_feature_head:
        # we cache the PCA reduction matrix and min/max values for visualization
        model.register_feats_reduction_mat(
            dataset.pixel_source.feat_dimension_reduction_mat,
            dataset.pixel_source.feat_color_min,
            dataset.pixel_source.feat_color_max,
        )
    return model.to(device)


def build_optimizer_from_cfg(
    cfg: OmegaConf, model: RadianceField
) -> torch.optim.Optimizer:
    # a very simple optimizer for now
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        eps=1e-15,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.99),
    )
    return optimizer


def build_scheduler_from_cfg(
    cfg: OmegaConf, optimizer: torch.optim.Optimizer
) -> torch.optim.Optimizer:
    # ------ build scheduler -------- #
    scheduler_milestones = [
        cfg.num_iters // 2,
        cfg.num_iters * 3 // 4,
        cfg.num_iters * 9 // 10,
    ]
    if cfg.num_iters >= 10000:
        scheduler_milestones.insert(0, cfg.num_iters // 4)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            # warmup
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=cfg.num_iters // 10
            ),
            # Linear decay
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_milestones,
                gamma=0.33,
            ),
        ]
    )
    return scheduler


def build_estimator_and_propnet_from_cfg(
    nerf_cfg: OmegaConf,
    optim_cfg: OmegaConf,
    dataset: SceneDataset,
    device: torch.device = torch.device("cpu"),
) -> Tuple[PropNetEstimator, List[DensityField]]:
    proposal_networks = [
        build_density_field(
            n_input_dims=nerf_cfg.propnet.xyz_encoder.n_input_dims,
            n_levels=nerf_cfg.propnet.xyz_encoder.n_levels_per_prop[i],
            max_resolution=nerf_cfg.propnet.xyz_encoder.max_resolution_per_prop[i],
            log2_hashmap_size=nerf_cfg.propnet.xyz_encoder.lgo2_hashmap_size_per_prop[
                i
            ],
            n_features_per_level=nerf_cfg.propnet.xyz_encoder.n_features_per_level,
            unbounded=nerf_cfg.unbounded,
        ).to(device)
        for i in range(len(nerf_cfg.propnet.num_samples_per_prop))
    ]
    if dataset.aabb is not None and nerf_cfg.model.resume_from is None:
        for p in proposal_networks:
            p.set_aabb(dataset.aabb)
    prop_optimizer = torch.optim.Adam(
        itertools.chain(*[p.parameters() for p in proposal_networks]),
        lr=optim_cfg.lr,
        eps=1e-15,
        weight_decay=optim_cfg.weight_decay,
        betas=(0.9, 0.99),
    )

    scheduler_milestones = [
        optim_cfg.num_iters // 2,
        optim_cfg.num_iters * 3 // 4,
        optim_cfg.num_iters * 9 // 10,
    ]
    if optim_cfg.num_iters >= 10000:
        scheduler_milestones.insert(0, optim_cfg.num_iters // 4)
    prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                prop_optimizer,
                start_factor=0.01,
                total_iters=optim_cfg.num_iters // 10,
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                prop_optimizer,
                milestones=scheduler_milestones,
                gamma=0.33,
            ),
        ]
    )
    estimator = PropNetEstimator(
        prop_optimizer,
        prop_scheduler,
        enable_anti_aliasing_loss=nerf_cfg.propnet.enable_anti_aliasing_level_loss,
        anti_aliasing_pulse_width=nerf_cfg.propnet.anti_aliasing_pulse_width,
    ).to(device)
    return estimator, proposal_networks
