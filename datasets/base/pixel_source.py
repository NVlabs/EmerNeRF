import abc
import logging
import os
from typing import Dict, Tuple, Union, List

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from third_party.feature_extractor import delete_features, extract_and_save_features
from utils.misc import get_robust_pca

logger = logging.getLogger()


def idx_to_3d(idx, H, W):
    """
    Converts a 1D index to a 3D index (img_id, row_id, col_id)

    Args:
        idx (int): The 1D index to convert.
        H (int): The height of the 3D space.
        W (int): The width of the 3D space.

    Returns:
        tuple: A tuple containing the 3D index (i, j, k),
                where i is the image index, j is the row index,
                and k is the column index.
    """
    i = idx // (H * W)
    j = (idx % (H * W)) // W
    k = idx % W
    return i, j, k


def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: the horizontal coordinates of the pixels, shape: (num_rays,)
        y: the vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_cams, 4, 4)
        intrinsic: the camera intrinsic matrices, shape: (num_cams, 3, 3)
    Returns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
                (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [num_rays, 3]

    # rotate the camera rays w.r.t. the camera pose
    directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
    # TODO: not sure if we still need direction_norm
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    # normalize the ray directions
    viewdirs = directions / (direction_norm + 1e-8)
    return origins, viewdirs, direction_norm


class ScenePixelSource(abc.ABC):
    """
    The base class for all pixel sources of a scene.
    """

    # the original size of the images in the dataset
    # these values are from the waymo dataset as a placeholder
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]

    # define a transformation matrix to convert the opencv camera coordinate system to the dataset camera coordinate system
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    data_cfg: OmegaConf = None
    # the normalized timestamps of all images (normalized to [0, 1]), shape: (num_imgs,)
    _normalized_timestamps: Tensor = None
    # the timestamps of all images, shape: (num_imgs,)
    _timestamps: Tensor = None
    # the timesteps of all images, shape: (num_imgs,)
    #   - the difference between timestamps and timesteps is that
    #     timestamps are the actual timestamps (minus 1e9) of images
    #     while timesteps are the integer timestep indices of images.
    _timesteps: Tensor = None
    # camera ids of all images, shape: (num_imgs,)
    cam_ids: Tensor = None
    # camera-to-world matrices of all images, shape: (num_imgs, 4, 4)
    cam_to_worlds: Tensor = None
    # camera intrinsic matrices of all images, shape: (num_imgs, 3, 3)
    intrinsics: Tensor = None
    # all image tensors, shape: (num_imgs, load_size[0], load_size[1], 3)
    images: Tensor = None
    # the image ids of all images, shape: (num_imgs,)
    img_ids: Tensor = None
    # the binary masks of dynamic objects, shape: (num_imgs, load_size[0], load_size[1])
    dynamic_masks: Tensor = None
    # the binary masks of sky regions, shape: (num_imgs, load_size[0], load_size[1])
    sky_masks: Tensor = None
    # the feature tensors, shape: (num_imgs, num_patches_h, num_patches_w, C)
    features: Tensor = None
    # the pca matrix used to reduce the feature dimension to target_feature_dim,
    #   shape: (original_feature_dim, target_feature_dim)
    reduce_to_target_dim_mat: Tensor = None
    # the min and max values of the reduced features used for normalization,
    #   shape: (target_feature_dim,)
    feat_min: Tensor = None
    feat_max: Tensor = None

    # the pca matrix used to reduce the feature dimension for visualization,
    #   shape: (target_feature_dim, 3)
    feat_dimension_reduction_mat: Tensor = None
    # the min and max values of the original features used for visualization,
    #  shape: (3,)
    feat_color_min: Tensor = None
    feat_color_max: Tensor = None
    # the downscale factor of the features, shape: (2,)
    featmap_downscale_factor: Tuple[float, float] = None

    # importance sampling weights of all images,
    #   shape: (num_imgs, load_size[0] // buffer_scale, load_size[1] // buffer_scale)
    pixel_error_maps: Tensor = None
    pixel_error_buffered: bool = False

    def __init__(
        self, pixel_data_config: OmegaConf, device: torch.device = torch.device("cpu")
    ) -> None:
        # hold the config of the pixel data
        self.data_cfg = pixel_data_config
        self.device = device
        self._downscale_factor = 1 / pixel_data_config.downscale
        self._old_downscale_factor = 1 / pixel_data_config.downscale

    @abc.abstractmethod
    def create_all_filelist(self) -> None:
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        self.img_filepaths = []
        self.feat_filepaths = []
        self.sky_mask_filepaths = []
        self.dynamic_mask_filepaths = []
        raise NotImplementedError

    @abc.abstractmethod
    def load_calibrations(self) -> None:
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        raise NotImplementedError

    def load_data(self) -> None:
        """
        A general function to load all data.
        """
        self.load_calibrations()
        self.load_rgb()
        self.load_dynamic_mask()
        self.load_sky_mask()
        self.load_features()
        # build the pixel error buffer
        self.build_pixel_error_buffer()
        logger.info("[Pixel] All Pixel Data loaded.")

    def to(self, device: torch.device) -> "ScenePixelSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self.images is not None:
            self.images = self.images.to(device)
        if self.dynamic_masks is not None:
            self.dynamic_masks = self.dynamic_masks.to(device)
        if self.sky_masks is not None:
            self.sky_masks = self.sky_masks.to(device)
        if self.features is not None:
            # this step can be dangerous because the features are huge
            # TODO: add a flag to control whether to move the features to GPU
            self.features = self.features.to(device)
            if self.reduce_to_target_dim_mat is not None:
                self.reduce_to_target_dim_mat = self.reduce_to_target_dim_mat.to(
                    self.device
                )
            if self.feat_min is not None:
                self.feat_min = self.feat_min.to(self.device)
                self.feat_max = self.feat_max.to(self.device)
            if self.feat_dimension_reduction_mat is not None:
                self.feat_dimension_reduction_mat = (
                    self.feat_dimension_reduction_mat.to(self.device)
                )
            self.feat_color_min = self.feat_color_min.to(self.device)
            self.feat_color_max = self.feat_color_max.to(self.device)
        if self.cam_to_worlds is not None:
            self.cam_to_worlds = self.cam_to_worlds.to(device)
        if self.intrinsics is not None:
            self.intrinsics = self.intrinsics.to(device)
        if self.cam_ids is not None:
            self.cam_ids = self.cam_ids.to(device)
        if self._timestamps is not None:
            self._timestamps = self._timestamps.to(device)
        if self._timesteps is not None:
            self._timesteps = self._timesteps.to(device)
        if self._normalized_timestamps is not None:
            self._normalized_timestamps = self._normalized_timestamps.to(device)
        if self.pixel_error_maps is not None:
            self.pixel_error_maps = self.pixel_error_maps.to(device)
        return self

    def load_rgb(self) -> None:
        """
        Load the RGB images if they are available. We cache the images in memory for faster loading.
        Note this can be memory consuming.
        """
        if not self.data_cfg.load_rgb:
            return
        images = []
        for fname in tqdm(
            self.img_filepaths, desc="Loading images", dynamic_ncols=True
        ):
            rgb = Image.open(fname).convert("RGB")
            # resize them to the load_size
            rgb = rgb.resize(
                (self.data_cfg.load_size[1], self.data_cfg.load_size[0]), Image.BILINEAR
            )
            images.append(rgb)
        # normalize the images to [0, 1]
        self.images = torch.from_numpy(np.stack(images, axis=0)) / 255
        self.img_ids = torch.arange(len(self.images)).long()

    def load_dynamic_mask(self) -> None:
        """
        Load the dynamic masks if they are available.
        """
        if not self.data_cfg.load_dynamic_mask:
            return
        dynamic_masks = []
        for fname in tqdm(
            self.dynamic_mask_filepaths,
            desc="Loading dynamic masks",
            dynamic_ncols=True,
        ):
            dyn_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            dyn_mask = dyn_mask.resize(
                (self.data_cfg.load_size[1], self.data_cfg.load_size[0]), Image.BILINEAR
            )
            dynamic_masks.append(np.array(dyn_mask) > 0)
        self.dynamic_masks = torch.from_numpy(np.stack(dynamic_masks, axis=0)).float()

    def load_sky_mask(self) -> None:
        """
        Load the sky masks if they are available.
        """
        if not self.data_cfg.load_sky_mask:
            return
        sky_masks = []
        for fname in tqdm(
            self.sky_mask_filepaths, desc="Loading sky masks", dynamic_ncols=True
        ):
            sky_mask = Image.open(fname).convert("L")
            # resize them to the load_size
            sky_mask = sky_mask.resize(
                (self.data_cfg.load_size[1], self.data_cfg.load_size[0]), Image.NEAREST
            )
            sky_masks.append(np.array(sky_mask) > 0)
        self.sky_masks = torch.from_numpy(np.stack(sky_masks, axis=0)).float()

    def load_features(self) -> None:
        """
        Load the features if they are available.
        """
        if not self.data_cfg.load_features:
            return

        if not self.data_cfg.skip_feature_extraction:
            logger.info(f"Extracting {self.data_cfg.feature_model_type}...")
            return_dict = extract_and_save_features(
                input_img_path_list=self.img_filepaths,
                saved_feat_path_list=self.feat_filepaths,
                img_shape=self.data_cfg.feature_extraction_size,
                stride=self.data_cfg.feature_extraction_stride,
                model_type=self.data_cfg.feature_model_type,
            )

        features = []
        for fname in tqdm(
            self.feat_filepaths, desc="Loading features", dynamic_ncols=True
        ):
            # mmap_mode="r" is to avoid memory overflow when loading features
            # but it only slightly helps... do we have a better way to load features?
            feature = np.load(fname, mmap_mode="r").squeeze()
            features.append(feature)
        # shape: (num_imgs, num_patches_h, num_patches_w, C)
        self.features = torch.from_numpy(np.stack(features, axis=0)).float()
        # featmap_downscale_factor is used to convert the image coordinates to ViT feature coordinates.
        # resizing ViT features to (H, W) using bilinear interpolation is infeasible.
        # imagine a feature array of shape (num_timesteps x num_cams, 640, 960, 768). it's too large to fit in GPU memory.
        self.featmap_downscale_factor = (
            self.features.shape[1] / self.data_cfg.load_size[0],
            self.features.shape[2] / self.data_cfg.load_size[1],
        )
        logger.info(
            f"Loaded {self.features.shape} {self.data_cfg.feature_model_type} features."
        )
        logger.info(f"Feature scale: {self.featmap_downscale_factor}")
        logger.info(f"Computing features PCA...")
        # compute feature visualization matrix
        C = self.features.shape[-1]
        # no need to compute PCA on the entire set of features, we randomly sample 100k features
        temp_feats = self.features.reshape(-1, C)
        max_elements_to_compute_pca = min(100000, temp_feats.shape[0])
        selected_features = temp_feats[
            np.random.choice(
                temp_feats.shape[0], max_elements_to_compute_pca, replace=False
            )
        ]
        if self.data_cfg.target_feature_dim is not None:
            logger.info(
                f"Reducing features to {self.data_cfg.target_feature_dim} dimensions."
            )
            # compute PCA to reduce the feature dimension to target_feature_dim
            U, S, reduce_to_target_dim_mat = torch.pca_lowrank(
                selected_features, q=self.data_cfg.target_feature_dim, niter=20
            )
            # compute the fraction of variance explained by target_feature_dim
            variances = S**2
            fraction_var_explained = variances / variances.sum()
            logger.info(f"[PCA] fraction_var_explained: \n{fraction_var_explained}")
            logger.info(
                f"[PCA] fraction_var_explained sum: {fraction_var_explained.sum()}",
            )
            self.reduce_to_target_dim_mat = reduce_to_target_dim_mat

            # reduce the features to target_feature_dim
            selected_features = selected_features @ reduce_to_target_dim_mat
            self.features = self.features @ reduce_to_target_dim_mat
            C = self.features.shape[-1]

            # normalize the reduced features to [0, 1] along each dimension
            feat_min = self.features.reshape(-1, C).min(dim=0)[0]
            feat_max = self.features.reshape(-1, C).max(dim=0)[0]
            self.features = (self.features - feat_min) / (feat_max - feat_min)
            selected_features = (selected_features - feat_min) / (feat_max - feat_min)
            self.feat_min = feat_min.to(self.device)
            self.feat_max = feat_max.to(self.device)
            self.reduce_to_target_dim_mat = reduce_to_target_dim_mat.to(self.device)
        # we compute the first 3 principal components of the ViT features as the color
        reduction_mat, feat_color_min, feat_color_max = get_robust_pca(
            selected_features
        )
        # final features are of shape (num_imgs, num_patches_h, num_patches_w, target_feature_dim)
        self.features = self.features

        # save visualization parameters
        self.feat_dimension_reduction_mat = reduction_mat
        self.feat_color_min = feat_color_min
        self.feat_color_max = feat_color_max
        del temp_feats, selected_features

        logger.info(
            f"Feature PCA computed, shape: {self.feat_dimension_reduction_mat.shape}"
        )

    def delete_features(self) -> None:
        """
        Delete the features if they exist.
        This is to save disk space. 2D features of a single sequence can be 30GB+.
        """
        delete_features(self.feat_filepaths)

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            We compute the coarse aabb by using the front camera positions / trajectories. We then
            extend this aabb by 40 meters along horizontal directions and 20 meters up and 5 meters
            down along vertical directions.
        """
        assert (
            self.cam_to_worlds is not None
        ), "Camera poses not loaded, cannot compute aabb."
        logger.info("[Pixel] Computing auto AABB based on front camera trajectory....")
        if self.num_cams == 1:
            # if there is only one camera, it's front camera
            front_cameras_positions = self.cam_to_worlds[:, :3, 3]
        elif self.num_cams == 3:
            # if there are three cameras, they are ordered as front_left, front, front_right
            front_cameras_positions = self.cam_to_worlds[1::3, :3, 3]
        elif self.num_cams == 5:
            # if there are five cameras, they are ordered as side_left, front_left, front, front_right, side_right
            front_cameras_positions = self.cam_to_worlds[2::5, :3, 3]
        elif self.num_cams == 6:
            # if there are six cameras, they are ordered as front_left, front, front_right, back_left, back, back_right
            front_cameras_positions = self.cam_to_worlds[2::6, :3, 3]

        # compute the aabb
        aabb_min = front_cameras_positions.min(dim=0)[0]
        aabb_max = front_cameras_positions.max(dim=0)[0]

        # extend aabb by 40 meters along forward direction and 40 meters along the left/right direction
        # aabb direction: x, y, z: front, left, up
        aabb_max[0] += 40
        aabb_max[1] += 40
        # when the car is driving uphills
        aabb_max[2] = min(aabb_max[2] + 20, 20)

        # for waymo, there will be a lot of waste of space because we don't have images in the back,
        # it's more reasonable to extend the aabb only by a small amount, e.g., 5 meters
        # we use 40 meters here for a more general case
        aabb_min[0] -= 40
        aabb_min[1] -= 40
        # when a car is driving downhills
        aabb_min[2] = max(aabb_min[2] - 5, -5)
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Pixel] Auto AABB from camera: {aabb}")
        return aabb

    def get_features(
        self,
        img_id,
        y: Tensor,
        x: Tensor,
        downscale: Union[float, Tuple[float, float]] = 1.0,
    ) -> Tensor:
        """
        Get the features at the given pixel coordinates.
        Args:
            img_id: the image index.
            y: the vertical coordinates of the pixels, shape: (num_rays,)
            x: the horizontal coordinates of the pixels, shape: (num_rays,)
            downscale: the downscale factor of the features.
                If it's a float, we use the same downscale factor for both height and width.
                If it's a tuple, we use the first value as the downscale factor for height
                and the second value as the downscale factor for width.
        Returns:
            features: the features at the given pixel coordinates.
               shape: (num_rays, feat_dim)
        """
        if isinstance(downscale, float):
            downscale = (downscale, downscale)
        # we compute the nearest DINO feature for each pixel
        # map (x, y) in the (W, H) space to (x * dino_scale[0], y * dino_scale[1]) in the (W//patch_size, H//patch_size) space
        dino_y = (y * downscale[0]).long()
        dino_x = (x * downscale[1]).long()
        # dino_feats are in CPU memory (because they are huge), so we need to move them to GPU
        dino_feat = self.features[img_id, dino_y.cpu(), dino_x.cpu()]
        return dino_feat

    def build_pixel_error_buffer(self) -> None:
        """
        Build the pixel error buffer.
        """
        if self.buffer_ratio > 0:
            # shape: (num_imgs, H // buffer_downscale, W // buffer_downscale)
            self.pixel_error_maps = torch.ones(
                (
                    len(self.cam_to_worlds),
                    self.HEIGHT // self.buffer_downscale,
                    self.WIDTH // self.buffer_downscale,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            logger.info(
                f"Successfully built pixel error buffer (log2(num_pixels) = {np.log2(len(self.pixel_error_maps.reshape(-1))):.2f})."
            )
        else:
            logger.info("Not building pixel error buffer because buffer_ratio <= 0.")

    def update_pixel_error_maps(self, render_results: Dict[str, Tensor]) -> None:
        """
        Update the pixel error buffer with the given render results.
        """
        if self.pixel_error_maps is None:
            logger.info("Skipping pixel error buffer update because it's not built.")
            return
        gt_rgbs = render_results["gt_rgbs"]
        pred_rgbs = render_results["rgbs"]
        gt_rgbs = torch.from_numpy(np.stack(gt_rgbs, axis=0))
        pred_rgbs = torch.from_numpy(np.stack(pred_rgbs, axis=0))
        pixel_error_maps = torch.abs(gt_rgbs - pred_rgbs).mean(dim=-1)
        assert pixel_error_maps.shape == self.pixel_error_maps.shape
        if "dynamic_opacities" in render_results:
            if len(render_results["dynamic_opacities"]) > 0:
                dynamic_opacity = render_results["dynamic_opacities"]
                dynamic_opacity = torch.from_numpy(np.stack(dynamic_opacity, axis=0))
                # we prioritize the dynamic objects by multiplying the error by 5
                pixel_error_maps[dynamic_opacity > 0.1] *= 5
        # update the pixel error buffer
        self.pixel_error_maps: Tensor = pixel_error_maps.to(self.device)
        # normalize the pixel error buffer to [0, 1]
        self.pixel_error_maps = (
            self.pixel_error_maps - self.pixel_error_maps.min()
        ) / (self.pixel_error_maps.max() - self.pixel_error_maps.min())
        self.pixel_error_buffered = True
        logger.info("Successfully updated pixel error buffer")

    def visualize_pixel_sample_weights(self, indices: List[int]) -> np.ndarray:
        """
        Visualize the pixel sample weights.
        Args:
            indices: the image indices to visualize.
        Returns:
            frames: the pixel sample weights of the given image.
                shape: (len(indices) // cams, H, num_cams * W, 3)
        """
        frames = (
            self.pixel_error_maps.detach()
            .cpu()
            .numpy()
            .reshape(
                self.num_imgs,
                self.HEIGHT // self.buffer_downscale,
                self.WIDTH // self.buffer_downscale,
            )[indices]
        )
        frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
        return np.uint8(np.concatenate(frames, axis=1) * 255)

    def get_pixel_sample_weights_video(self) -> List[np.ndarray]:
        """
        Get the pixel sample weights video.
        Returns:
            frames: the pixel sample weights video.
                shape: (num_imgs // cams, H, num_cams * W, 3)
        """
        assert self.buffer_ratio > 0, "buffer_ratio must be > 0"
        maps = []
        loss_maps = (
            self.pixel_error_maps.detach()
            .cpu()
            .numpy()
            .reshape(
                self.num_imgs,
                self.HEIGHT // self.buffer_downscale,
                self.WIDTH // self.buffer_downscale,
            )
        )
        for i in range(self.num_imgs):
            maps.append(loss_maps[i])
        return maps

    def sample_important_rays(
        self, num_rays, img_candidate_indices: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample rays coordinates from the given images based on the pixel error buffer.
        Args:
            num_rays: the number of rays to sample.
            img_candidate_indices: the indices of the images to sample from.
                If None, sample from all the images.
                If not None, sample from the given images only.
        Returns:
            img_id: the image indices of the sampled rays.
                shape: (num_rays,)
            y: the vertical coordinates of the sampled rays.
                shape: (num_rays,)
            x: the horizontal coordinates of the sampled rays.
                shape: (num_rays,)
        """
        assert self.pixel_error_buffered, "Pixel error buffer not built."
        # if img_candidate_indices is None, use all image indices
        if img_candidate_indices is None:
            img_candidate_indices = torch.arange(len(self.images)).to(self.device)
        if not isinstance(img_candidate_indices, Tensor):
            img_candidate_indices = torch.tensor(img_candidate_indices).to(self.device)
        sampled_indices = torch.multinomial(
            self.pixel_error_maps[img_candidate_indices].reshape(-1),
            num_rays,
            replacement=False,
        )
        # convert the sampled 1d indices to (img_idx, y, x)
        img_idx, y, x = idx_to_3d(
            sampled_indices,
            self.HEIGHT // self.buffer_downscale,
            self.WIDTH // self.buffer_downscale,
        )
        img_idx = img_candidate_indices[img_idx]

        # Upscale to the original resolution
        y, x = (y * self.buffer_downscale).long(), (x * self.buffer_downscale).long()

        # Add a random offset to avoid sampling the same pixel
        y += torch.randint(
            0, self.buffer_downscale, (num_rays,), device=self.images.device
        )
        x += torch.randint(
            0, self.buffer_downscale, (num_rays,), device=self.images.device
        )
        # Clamp to ensure coordinates don't exceed the image bounds
        y = torch.clamp(y, 0, self.HEIGHT - 1)
        x = torch.clamp(x, 0, self.WIDTH - 1)
        return img_idx, y, x

    def sample_uniform_rays(
        self,
        num_rays: int,
        img_candidate_indices: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample rays coordinates uniformly from the given images.
        Args:
            num_rays: the number of rays to sample.
            img_candidate_indices: the indices of the images to sample from.
                If None, sample from all the images.
                If not None, sample from the given images only.
        Returns:
            img_id: the image indices of the sampled rays.
                shape: (num_rays,)
            y: the vertical coordinates of the sampled rays.
                shape: (num_rays,)
            x: the horizontal coordinates of the sampled rays.
                shape: (num_rays,)
        """
        # if img_candidate_indices is None, use all image indices
        if img_candidate_indices is None:
            img_candidate_indices = torch.arange(len(self.images)).to(self.device)
        if not isinstance(img_candidate_indices, Tensor):
            img_candidate_indices = torch.tensor(img_candidate_indices).to(self.device)
        # sample random index based on img_candidate_indices
        random_idx = torch.randint(
            0,
            len(img_candidate_indices),
            size=(num_rays,),
            device=self.device,
        )
        img_id = img_candidate_indices[random_idx]

        # sample pixels
        x = torch.randint(
            0,
            self.WIDTH,
            size=(num_rays,),
            device=self.device,
        )
        y = torch.randint(
            0,
            self.HEIGHT,
            size=(num_rays,),
            device=self.device,
        )
        x, y = x.long(), y.long()
        return img_id, y, x

    def get_train_rays(
        self,
        num_rays: int,
        candidate_indices: Tensor = None,
    ) -> Dict[str, Tensor]:
        """
        Get a batch of rays for training.
        Args:
            num_rays: the number of rays to sample.
            candidate_indices: the indices of the images to sample from.
                If None, sample from all the images.
                If not None, sample from the given images only.
        Returns:
            a dict of the sampled rays.
        """
        rgb, sky_mask, dynamic_mask, features = None, None, None, None
        pixel_coords, normalized_timestamps = None, None
        if self.buffer_ratio > 0 and self.pixel_error_buffered:
            num_roi_rays = int(num_rays * self.buffer_ratio)
            num_random_rays = num_rays - num_roi_rays
            random_img_idx, random_y, random_x = self.sample_uniform_rays(
                num_random_rays, candidate_indices
            )
            roi_img_idx, roi_y, roi_x = self.sample_important_rays(
                num_roi_rays, candidate_indices
            )
            img_idx = torch.cat([random_img_idx, roi_img_idx], dim=0)
            y = torch.cat([random_y, roi_y], dim=0)
            x = torch.cat([random_x, roi_x], dim=0)
        else:
            img_idx, y, x = self.sample_uniform_rays(
                num_rays=num_rays, img_candidate_indices=candidate_indices
            )
        pixel_coords = torch.stack([y / self.HEIGHT, x / self.WIDTH], dim=-1)
        if self.images is not None:
            rgb = self.images[img_idx, y, x]
        if self.sky_masks is not None:
            sky_mask = self.sky_masks[img_idx, y, x]
        if self.dynamic_masks is not None:
            dynamic_mask = self.dynamic_masks[img_idx, y, x].float()
        if self.features is not None:
            features = self.get_features(
                img_idx, y, x, downscale=self.featmap_downscale_factor
            )
        if self.normalized_timestamps is not None:
            normalized_timestamps = self.normalized_timestamps[img_idx]
        if self.cam_ids is not None:
            camera_id = self.cam_ids[img_idx]
        image_id = torch.ones_like(x) * img_idx
        c2w = self.cam_to_worlds[img_idx]
        intrinsics = self.intrinsics[img_idx]
        origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
        data = {
            "origins": origins,
            "viewdirs": viewdirs,
            "direction_norms": direction_norm,
            "pixel_coords": pixel_coords,
            "normed_timestamps": normalized_timestamps,
            "img_idx": image_id,
            "cam_idx": camera_id,
            "pixels": rgb,
            "sky_masks": sky_mask,
            "dynamic_masks": dynamic_mask,
            "features": features,
        }
        return {k: v for k, v in data.items() if v is not None}

    def get_render_rays(self, img_idx: int) -> Dict[str, Tensor]:
        """
        Get the rays for rendering the given image index.
        Args:
            img_idx: the image index.
        Returns:
            a dict containing the rays for rendering the given image index.
        """
        rgb, sky_mask, dynamic_mask, features = None, None, None, None
        pixel_coords, normalized_timestamps = None, None
        if self.images is not None:
            rgb = self.images[img_idx]
            if self.downscale_factor != 1.0:
                rgb = (
                    torch.nn.functional.interpolate(
                        rgb.unsqueeze(0).permute(0, 3, 1, 2),
                        scale_factor=self.downscale_factor,
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )
                img_height, img_width = rgb.shape[:2]
            else:
                img_height, img_width = self.HEIGHT, self.WIDTH

        x, y = torch.meshgrid(
            torch.arange(img_width),
            torch.arange(img_height),
            indexing="xy",
        )
        x, y = x.flatten(), y.flatten()
        x, y = x.to(self.device), y.to(self.device)
        # pixel coordinates
        pixel_coords = (
            torch.stack([y / img_height, x / img_width], dim=-1)
            .float()
            .reshape(img_height, img_width, 2)
        )

        if self.sky_masks is not None:
            sky_mask = self.sky_masks[img_idx]
            if self.downscale_factor != 1.0:
                sky_mask = (
                    torch.nn.functional.interpolate(
                        sky_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        if self.dynamic_masks is not None:
            dynamic_mask = self.dynamic_masks[img_idx].float()
            if self.downscale_factor != 1.0:
                dynamic_mask = (
                    torch.nn.functional.interpolate(
                        dynamic_mask.unsqueeze(0).unsqueeze(0),
                        scale_factor=self.downscale_factor,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
        if self.features is not None:
            features = self.get_features(
                img_idx,
                y,
                x,
                downscale=(
                    self.featmap_downscale_factor[0] / self.downscale_factor,
                    self.featmap_downscale_factor[1] / self.downscale_factor,
                ),
            ).reshape(img_height, img_width, -1)

        if self.normalized_timestamps is not None:
            normalized_timestamps = torch.full(
                (img_height, img_width),
                self.normalized_timestamps[img_idx],
                dtype=torch.float32,
            )
        if self.cam_ids is not None:
            camera_id = torch.full(
                (img_height, img_width),
                self.cam_ids[img_idx],
                dtype=torch.long,
            )
        image_id = torch.full(
            (img_height, img_width),
            img_idx,
            dtype=torch.long,
        )
        c2w = self.cam_to_worlds[img_idx]
        intrinsics = self.intrinsics[img_idx] * self.downscale_factor
        intrinsics[2, 2] = 1.0
        origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsics)
        origins = origins.reshape(img_height, img_width, 3)
        viewdirs = viewdirs.reshape(img_height, img_width, 3)
        direction_norm = direction_norm.reshape(img_height, img_width, 1)
        data = {
            "origins": origins,
            "viewdirs": viewdirs,
            "direction_norm": direction_norm,
            "pixel_coords": pixel_coords,
            "normed_timestamps": normalized_timestamps,
            "img_idx": image_id,
            "cam_idx": camera_id,
            "pixels": rgb,
            "sky_masks": sky_mask,
            "dynamic_masks": dynamic_mask,
            "features": features,
        }
        return {k: v for k, v in data.items() if v is not None}

    @property
    def num_cams(self) -> int:
        """
        Returns:
            the number of cameras in the dataset
        """
        return self.data_cfg.num_cams

    @property
    def num_imgs(self) -> int:
        """
        Returns:
            the number of images in the dataset
        """
        return len(self.cam_to_worlds)

    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of image timesteps in the dataset
        """
        return len(self.timesteps.unique())

    @property
    def timesteps(self) -> Tensor:
        """
        Returns:
            the integer timestep indices of all images,
            shape: (num_imgs,)
        Note:
            the difference between timestamps and timesteps is that
            timestamps are the actual timestamps (minus 1e9) of images
            while timesteps are the integer timestep indices of images.
        """
        return self._timesteps

    @property
    def timestamps(self) -> Tensor:
        """
        Returns:
            the actual timestamps (minus 1e9) of all images,
            shape: (num_imgs,)
        """
        return self._timestamps

    @property
    def normalized_timestamps(self) -> Tensor:
        """
        Returns:
            the normalized timestamps of all images
            (normalized to the range [0, 1]),
            shape: (num_imgs,)
        """
        return self._normalized_timestamps

    @property
    def unique_normalized_timestamps(self) -> Tensor:
        """
        Returns:
            the unique normalized timestamps of all images
            (normalized to the range [0, 1]).
            shape: (num_timesteps,)
        """
        return self._unique_normalized_timestamps

    def register_normalized_timestamps(self, normalized_timestamps: Tensor) -> None:
        """
        Register the normalized timestamps of all images.
        Args:
            normalized_timestamps: the normalized timestamps of all images
                (normalized to the range [0, 1]).
                shape: (num_imgs,)
        Note:
            we normalize the image timestamps together with the lidar timestamps,
            so that the both the image and lidar timestamps are in the range [0, 1].
        """
        assert normalized_timestamps.shape[0] == len(
            self.img_filepaths
        ), "The number of normalized timestamps must match the number of images."
        assert (
            normalized_timestamps.min() >= 0 and normalized_timestamps.max() <= 1
        ), "The normalized timestamps must be in the range [0, 1]."
        self._normalized_timestamps = normalized_timestamps.to(self.device)
        self._unique_normalized_timestamps = self._normalized_timestamps.unique()

    def find_closest_timestep(self, normed_timestamp: float) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        return torch.argmin(
            torch.abs(self.unique_normalized_timestamps - normed_timestamp)
        )

    @property
    def HEIGHT(self) -> int:
        return self.data_cfg.load_size[0]

    @property
    def WIDTH(self) -> int:
        return self.data_cfg.load_size[1]

    @property
    def downscale_factor(self) -> float:
        """
        Returns:
            downscale_factor: the downscale factor of the images
        """
        return self._downscale_factor

    def update_downscale_factor(self, downscale: float) -> None:
        """
        Args:
            downscale: the new downscale factor
        Updates the downscale factor
        """
        self._old_downscale_factor = self._downscale_factor
        self._downscale_factor = downscale

    def reset_downscale_factor(self) -> None:
        """
        Resets the downscale factor to the original value
        """
        self._downscale_factor = self._old_downscale_factor

    @property
    def buffer_downscale(self) -> float:
        """
        Returns:
            buffer_downscale: the downscale factor of the pixel error buffer
        """
        return self.data_cfg.sampler.buffer_downscale

    @property
    def buffer_ratio(self) -> float:
        """
        Returns:
            buffer_ratio: the ratio of the rays sampled from the pixel error buffer
        """
        return self.data_cfg.sampler.buffer_ratio
