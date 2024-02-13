import json
import logging
import os
from typing import Dict

import numpy as np
import torch
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from torch import Tensor
from tqdm import trange

from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from datasets.utils import voxel_coords_to_world_coords
from radiance_fields.video_utils import save_videos, depth_visualizer
from utils.misc import NumpyEncoder

logger = logging.getLogger()


class NuScenesPixelSource(ScenePixelSource):
    ORIGINAL_SIZE = [[900, 1600] for _ in range(6)]
    OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def __init__(
        self,
        pixel_data_config: OmegaConf,
        data_path: str,
        meta_file_path: str,
        nusc: NuScenes = None,
        scene_idx: int = 0,
        start_timestep: int = 0,
        end_timestep: int = -1,
        device: torch.device = torch.device("cpu"),
    ):
        pixel_data_config.load_dynamic_mask = False
        logger.info("[Pixel] Overriding load_dynamic_mask to False")
        super().__init__(pixel_data_config, device=device)
        self.data_path = data_path
        self.meta_file_path = meta_file_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.nusc = nusc
        self.scene_idx = scene_idx
        self.meta_dict = self.create_or_load_metas()
        self.create_all_filelist()
        self.load_data()

    def create_or_load_metas(self):
        # ---- define camera list ---- #
        if self.num_cams == 1:
            self.camera_list = ["CAM_FRONT"]
        elif self.num_cams == 3:
            self.camera_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        elif self.num_cams == 6:
            self.camera_list = [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT",
                "CAM_BACK",
                "CAM_BACK_RIGHT",
            ]
        else:
            raise NotImplementedError(
                f"num_cams: {self.num_cams} not supported for nuscenes dataset"
            )

        if os.path.exists(self.meta_file_path):
            with open(self.meta_file_path, "r") as f:
                meta_dict = json.load(f)
            logger.info(f"[Pixel] Loaded camera meta from {self.meta_file_path}")
            return meta_dict
        else:
            logger.info(f"[Pixel] Creating camera meta at {self.meta_file_path}")

        if self.nusc is None:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.data_path, verbose=True
            )
            self.scene = self.nusc.scene[self.scene_idx]
        total_camera_list = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        meta_dict = {
            camera: {
                "timestamp": [],
                "filepath": [],
                "ego_pose": [],
                "cam_id": [],
                "extrinsics": [],
                "intrinsics": [],
            }
            for i, camera in enumerate(total_camera_list)
        }

        # ---- get the first sample of each camera ---- #
        current_camera_data_tokens = {camera: None for camera in total_camera_list}
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        for camera in total_camera_list:
            current_camera_data_tokens[camera] = first_sample["data"][camera]

        while not all(token == "" for token in current_camera_data_tokens.values()):
            for i, camera in enumerate(total_camera_list):
                # skip if the current camera data token is empty
                if current_camera_data_tokens[camera] == "":
                    continue

                current_camera_data = self.nusc.get(
                    "sample_data", current_camera_data_tokens[camera]
                )

                # ---- timestamp and cam_id ---- #
                meta_dict[camera]["cam_id"].append(i)
                meta_dict[camera]["timestamp"].append(current_camera_data["timestamp"])
                meta_dict[camera]["filepath"].append(current_camera_data["filename"])

                # ---- intrinsics and extrinsics ---- #
                calibrated_sensor_record = self.nusc.get(
                    "calibrated_sensor", current_camera_data["calibrated_sensor_token"]
                )
                # intrinsics
                intrinsic = calibrated_sensor_record["camera_intrinsic"]
                meta_dict[camera]["intrinsics"].append(np.array(intrinsic))

                # extrinsics
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = Quaternion(
                    calibrated_sensor_record["rotation"]
                ).rotation_matrix
                extrinsic[:3, 3] = np.array(calibrated_sensor_record["translation"])
                meta_dict[camera]["extrinsics"].append(extrinsic)

                # ---- ego pose ---- #
                ego_pose_record = self.nusc.get(
                    "ego_pose", current_camera_data["ego_pose_token"]
                )
                ego_pose = np.eye(4)
                ego_pose[:3, :3] = Quaternion(
                    ego_pose_record["rotation"]
                ).rotation_matrix
                ego_pose[:3, 3] = np.array(ego_pose_record["translation"])
                meta_dict[camera]["ego_pose"].append(ego_pose)

                current_camera_data_tokens[camera] = current_camera_data["next"]

        with open(self.meta_file_path, "w") as f:
            json.dump(meta_dict, f, cls=NumpyEncoder)
        logger.info(f"[Pixel] Saved camera meta to {self.meta_file_path}")
        return meta_dict

    def create_all_filelist(self):
        # NuScenes dataset is not synchronized, so we need to find the minimum shared
        # scene length, and only use the frames within the shared scene length.
        # we also define the start and end timestep within the shared scene length

        # ---- find the minimum shared scene length ---- #
        num_timestamps = 100000000
        for camera in self.camera_list:
            if len(self.meta_dict[camera]["timestamp"]) < num_timestamps:
                num_timestamps = len(self.meta_dict[camera]["timestamp"])
        logger.info(f"[Pixel] Min shared scene length: {num_timestamps}")
        self.scene_total_num_timestamps = num_timestamps

        if self.end_timestep == -1:
            self.end_timestep = num_timestamps - 1
        else:
            self.end_timestep = min(self.end_timestep, num_timestamps - 1)

        # to make sure the last timestep is included
        self.end_timestep += 1
        self.start_timestep = min(self.start_timestep, self.end_timestep - 1)
        self.scene_fraction = (self.end_timestep - self.start_timestep) / num_timestamps

        logger.info(f"[Pixel] Start timestep: {self.start_timestep}")
        logger.info(f"[Pixel] End timestep: {self.end_timestep}")

        img_filepaths, feat_filepaths, sky_mask_filepaths = [], [], []
        # TODO: support dynamic masks

        for t in range(self.start_timestep, self.end_timestep):
            for cam_idx in self.camera_list:
                img_filepath = os.path.join(
                    self.data_path, self.meta_dict[cam_idx]["filepath"][t]
                )
                img_filepaths.append(img_filepath)
                sky_mask_filepaths.append(
                    img_filepath.replace("samples", "samples_sky_mask")
                    .replace("sweeps", "sweeps_sky_mask")
                    .replace(".jpg", ".png")
                )
                feat_filepaths.append(
                    img_filepath.replace(
                        "samples", f"samples_{self.data_cfg.feature_model_type}"
                    )
                    .replace("sweeps", f"sweeps_{self.data_cfg.feature_model_type}")
                    .replace(".jpg", ".npy")
                )
        self.img_filepaths = np.array(img_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.feat_filepaths = np.array(feat_filepaths)

    def load_calibrations(self):
        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, timesteps, cam_ids = [], [], []
        timestamps = []

        # we tranform the camera poses w.r.t. the first timestep to make the origin of
        # the first ego pose  as the origin of the world coordinate system.
        initial_ego_to_global = self.meta_dict["CAM_FRONT"]["ego_pose"][
            self.start_timestep
        ]
        global_to_initial_ego = np.linalg.inv(initial_ego_to_global)

        for t in range(self.start_timestep, self.end_timestep):
            for cam_name in self.camera_list:
                cam_to_ego = self.meta_dict[cam_name]["extrinsics"][t]
                ego_to_global_current = self.meta_dict[cam_name]["ego_pose"][t]
                # compute ego_to_world transformation
                ego_to_world = global_to_initial_ego @ ego_to_global_current
                # Because we use opencv coordinate system to generate camera rays,
                # we need to store the transformation from opencv coordinate system to dataset
                # coordinate system. However, the nuScenes dataset uses the same coordinate
                # system as opencv, so we just store the identity matrix.
                # opencv coordinate system: x right, y down, z front
                cam_to_ego = cam_to_ego @ self.OPENCV2DATASET
                cam2world = ego_to_world @ cam_to_ego
                cam_to_worlds.append(cam2world)
                intrinsics.append(self.meta_dict[cam_name]["intrinsics"][t])
                timesteps.append(t)
                cam_ids.append(self.meta_dict[cam_name]["cam_id"][t])
                timestamps.append(
                    self.meta_dict[cam_name]["timestamp"][t]
                    / 1e6
                    * np.ones_like(self.meta_dict[cam_name]["cam_id"][t])
                )

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        # scale the intrinsics according to the load size
        self.intrinsics[..., 0, 0] *= (
            self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[0][1]
        )
        self.intrinsics[..., 1, 1] *= (
            self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[0][0]
        )
        self.intrinsics[..., 0, 2] *= (
            self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[0][1]
        )
        self.intrinsics[..., 1, 2] *= (
            self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[0][0]
        )

        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        # self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        self.global_to_initial_ego = torch.from_numpy(global_to_initial_ego).float()
        self.cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # the underscore here is important.
        self._timestamps = torch.tensor(timestamps, dtype=torch.float64)
        self._timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()


class NuScenesLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        meta_file_path: str,
        nusc: NuScenes,
        scene_idx: int,
        start_timestep: int,
        fraction: float,  # a value in [0, 1] to indicate the fraction of the scene to use
        global_to_initial_ego: Tensor,
    ):
        super().__init__(lidar_data_config)
        self.data_path = data_path
        self.meta_file_path = meta_file_path
        self.nusc = nusc
        self.scene_idx = scene_idx
        self.start_timestep = start_timestep
        # because the lidar data is not synchronized with the image data, we need to
        # define the end timestep based on the fraction of the scene to use
        self.fraction = fraction
        self.global_to_initial_ego = global_to_initial_ego.numpy()
        self.meta_dict = self.create_or_load_metas()
        self.create_all_filelist()
        self.load_data()

    def create_or_load_metas(self):
        if os.path.exists(self.meta_file_path):
            with open(self.meta_file_path, "r") as f:
                meta_dict = json.load(f)
            logger.info(f"[Lidar] Loaded lidar meta from {self.meta_file_path}")
            return meta_dict
        else:
            logger.info(f"[Lidar] Creating lidar meta at {self.meta_file_path}")

        if self.nusc is None:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.data_path, verbose=True
            )
        self.scene = self.nusc.scene[self.scene_idx]

        meta_dict = {
            "timestamp": [],
            "filepath": [],
            "extrinsics": [],
            "ego_pose": [],
        }

        # ---- obtain initial pose ---- #
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
        current_data_token = first_sample["data"]["LIDAR_TOP"]

        while current_data_token != "":
            current_lidar_data = self.nusc.get("sample_data", current_data_token)
            # ---- timestamp and cam_id ---- #
            meta_dict["timestamp"].append(current_lidar_data["timestamp"])
            meta_dict["filepath"].append(current_lidar_data["filename"])

            # ---- extrinsics ---- #
            calibrated_sensor_record = self.nusc.get(
                "calibrated_sensor", current_lidar_data["calibrated_sensor_token"]
            )
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = Quaternion(
                calibrated_sensor_record["rotation"]
            ).rotation_matrix
            extrinsic[:3, 3] = np.array(calibrated_sensor_record["translation"])
            meta_dict["extrinsics"].append(extrinsic)

            # ---- ego pose ---- #
            ego_pose_record = self.nusc.get(
                "ego_pose", current_lidar_data["ego_pose_token"]
            )
            ego_pose = np.eye(4)
            ego_pose[:3, :3] = Quaternion(ego_pose_record["rotation"]).rotation_matrix
            ego_pose[:3, 3] = np.array(ego_pose_record["translation"])
            meta_dict["ego_pose"].append(ego_pose)
            current_data_token = current_lidar_data["next"]

        with open(self.meta_file_path, "w") as f:
            json.dump(meta_dict, f, cls=NumpyEncoder)
        logger.info(f"[Lidar] Saved lidar meta to {self.meta_file_path}")
        return meta_dict

    def create_all_filelist(self):
        # ---- define filepaths ---- #
        num_timestamps = len(self.meta_dict["timestamp"])
        self.end_timestep = int(num_timestamps * self.fraction)

        self.start_timestep = min(self.start_timestep, self.end_timestep - 1)

        logger.info(f"[Lidar] Start timestep: {self.start_timestep}")
        logger.info(f"[Lidar] End timestep: {self.end_timestep}")

        lidar_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, self.meta_dict["filepath"][t])
            )
        self.lidar_filepaths = np.array(lidar_filepaths)

    def load_calibrations(self):
        lidar_to_worlds, ego_to_worlds = [], []
        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first ego pose as the origin of the world coordinate system.
        for t in range(self.start_timestep, self.end_timestep):
            lidar_to_ego = np.array(self.meta_dict["extrinsics"][t])
            ego_to_global_current = np.array(self.meta_dict["ego_pose"][t])
            # compute ego_to_world transformation
            ego_to_world = self.global_to_initial_ego @ ego_to_global_current
            ego_to_worlds.append(ego_to_world)
            lidar_to_worlds.append(ego_to_world @ lidar_to_ego)
        self.lidar_to_worlds = torch.from_numpy(
            np.stack(lidar_to_worlds, axis=0)
        ).float()
        self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()

    def load_lidar(self):
        origins, directions, ranges, timesteps = [], [], [], []
        laser_ids = []
        timestamps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):
            lidar_pc = LidarPointCloud.from_file(self.lidar_filepaths[t])
            lidar_pc.remove_close(1.0)
            pc = lidar_pc.points[:3, :].T
            pc = np.hstack((pc, np.ones((pc.shape[0], 1))))
            pc = torch.from_numpy(pc).float()
            lidar_points = pc @ self.lidar_to_worlds[t].T
            lidar_points = lidar_points[:, :3]
            lidar_origins = (
                self.lidar_to_worlds[t][:3, 3]
                .unsqueeze(0)
                .repeat(lidar_points.shape[0], 1)
            )
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            accumulated_num_original_rays += len(lidar_pc.points[0])

            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (
                    lidar_points[:, 0] > self.data_cfg.truncated_min_range
                )
            lidar_origins = lidar_origins[valid_mask]
            lidar_directions = lidar_directions[valid_mask]
            lidar_ranges = lidar_ranges[valid_mask]
            lidar_timestep = torch.ones_like(lidar_ranges).squeeze(-1) * t
            lidar_ids = torch.zeros_like(lidar_origins[:, 0]).long()
            accumulated_num_rays += len(lidar_ranges)
            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            timesteps.append(lidar_timestep)
            laser_ids.append(lidar_ids)
            timestamps.append(
                self.meta_dict["timestamp"][t]
                / 1e6
                * torch.ones_like(lidar_ids, dtype=torch.float64)
            )

        logger.info(
            f"[Lidar] Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}%) of "
            f"{accumulated_num_original_rays} original rays)"
        )
        logger.info("[Lidar] Filter condition:")
        logger.info(f"  only_use_top_lidar: {self.data_cfg.only_use_top_lidar}")
        logger.info(f"  truncated_max_range: {self.data_cfg.truncated_max_range}")
        logger.info(f"  truncated_min_range: {self.data_cfg.truncated_min_range}")

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self._timesteps = torch.cat(timesteps, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        self._timestamps = torch.cat(timestamps, dim=0)

    def sample_uniform_rays(
        self,
        num_rays: int,
        candidate_indices: Tensor = None,
    ):
        # in nuscenes, we don't support novel view synthesis yet, so we don't need to
        # use candidate indices
        self.cached_origins = self.origins
        self.cached_directions = self.directions
        self.cached_ranges = self.ranges
        self.cached_normalized_timestamps = self.normalized_timestamps
        return torch.randint(
            0,
            len(self.cached_origins),
            size=(num_rays,),
            device=self.device,
        )


class NuScenesDataset(SceneDataset):
    dataset: str = "nuscenes"

    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)
        assert self.data_cfg.dataset == "nuscenes"
        self.data_path = self.data_cfg.data_root
        self.processed_data_path = os.path.join(
            self.data_path, "emernerf_metas", f"{self.scene_idx:03d}"
        )
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        self.img_meta_file_path = os.path.join(
            self.processed_data_path, "img_meta.json"
        )
        self.lidar_meta_file_path = os.path.join(
            self.processed_data_path, "lidar_meta.json"
        )

        # ---- create pixel source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()
        self.aabb = self.get_aabb()

        # ---- define train and test indices ---- #
        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()
        # ---- create split wrappers ---- #
        pixel_sets, lidar_sets = self.build_split_wrapper()
        self.train_pixel_set, self.test_pixel_set, self.full_pixel_set = pixel_sets
        self.train_lidar_set, self.test_lidar_set, self.full_lidar_set = lidar_sets

    def build_split_wrapper(self):
        """
        Makes each data source as a Pytorch Dataset
        """
        train_pixel_set, test_pixel_set, full_pixel_set = None, None, None
        train_lidar_set, test_lidar_set, full_lidar_set = None, None, None
        assert (
            len(self.test_indices) == 0
        ), "Test split is not supported yet for nuscenes"
        # ---- create split wrappers ---- #
        if self.pixel_source is not None:
            train_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                # train_indices are img indices, so the length is num_cams * num_train_timesteps
                split_indices=self.train_indices,
                split="train",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            full_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                # cover all the images
                split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
                split="full",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
        if self.lidar_source is not None:
            train_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                # the number of image timesteps is different from the number of lidar timesteps
                # TODO: find a better way to handle this
                # currently use all the lidar timesteps for training
                split_indices=np.arange(self.lidar_source.num_timesteps),
                split="train",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            full_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                # cover all the lidar scans
                split_indices=np.arange(self.lidar_source.num_timesteps),
                split="full",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )

        pixel_set = (train_pixel_set, test_pixel_set, full_pixel_set)
        lidar_set = (train_lidar_set, test_lidar_set, full_lidar_set)
        return pixel_set, lidar_set

    def build_data_source(self):
        pixel_source, lidar_source = None, None
        all_timestamps = []
        # ---- create pixel source ---- #
        load_pixel = (
            self.data_cfg.pixel_source.load_rgb
            or self.data_cfg.pixel_source.load_sky_mask
            or self.data_cfg.pixel_source.load_dynamic_mask
            or self.data_cfg.pixel_source.load_feature
        )
        if load_pixel:
            pixel_source = NuScenesPixelSource(
                pixel_data_config=self.data_cfg.pixel_source,
                data_path=self.data_path,
                scene_idx=self.scene_idx,
                meta_file_path=self.img_meta_file_path,
                start_timestep=self.data_cfg.start_timestep,
                end_timestep=self.data_cfg.end_timestep,
            )
            pixel_source.to(self.device)
            all_timestamps.append(pixel_source.timestamps)
            self.start_timestep = pixel_source.start_timestep
            self.end_timestep = pixel_source.end_timestep
            self.scene_fraction = pixel_source.scene_fraction
        # ---- create lidar source ---- #
        if self.data_cfg.lidar_source.load_lidar:
            lidar_source = NuScenesLiDARSource(
                lidar_data_config=self.data_cfg.lidar_source,
                data_path=self.data_path,
                meta_file_path=self.lidar_meta_file_path,
                nusc=pixel_source.nusc if pixel_source is not None else None,
                scene_idx=self.scene_idx,
                start_timestep=self.start_timestep,
                fraction=self.scene_fraction,
                global_to_initial_ego=pixel_source.global_to_initial_ego,
            )
            lidar_source.to(self.device)
            all_timestamps.append(lidar_source.timestamps)

        assert len(all_timestamps) > 0, "No data source is loaded"
        all_timestamps = torch.cat(all_timestamps, dim=0)
        # normalize the timestamps
        all_timestamps = (all_timestamps - all_timestamps.min()) / (
            all_timestamps.max() - all_timestamps.min()
        )
        all_timestamps = all_timestamps.float()
        if pixel_source is not None:
            pixel_source.register_normalized_timestamps(
                all_timestamps[: len(pixel_source.timestamps)]
            )
        if lidar_source is not None:
            lidar_source.register_normalized_timestamps(
                all_timestamps[-len(lidar_source.timestamps) :]
            )
        return pixel_source, lidar_source

    def split_train_test(self):
        assert (
            self.data_cfg.pixel_source.test_image_stride == 0
        ), "test_image_stride > 0 is not supported for nuscenes dataset. "
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                self.data_cfg.pixel_source.test_image_stride,
                self.num_img_timesteps,
                self.data_cfg.pixel_source.test_image_stride,
            )
        else:
            test_timesteps = []
        train_timesteps = np.array(
            [i for i in range(self.num_img_timesteps) if i not in test_timesteps]
        )
        logger.info(
            f"Train timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[train_timesteps]}"
        )
        logger.info(
            f"Test timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[test_timesteps]}"
        )

        # propagate the train and test timesteps to the train and test indices
        train_indices, test_indices = [], []
        for t in range(self.num_img_timesteps):
            if t in train_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    train_indices.append(t * self.pixel_source.num_cams + cam)
            elif t in test_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    test_indices.append(t * self.pixel_source.num_cams + cam)
        logger.info(f"Number of train indices: {len(train_indices)}")
        logger.info(f"Train indices: {train_indices}")
        logger.info(f"Number of test indices: {len(test_indices)}")
        logger.info(f"Test indices: {test_indices}")

        return train_timesteps, test_timesteps, train_indices, test_indices

    def save_videos(self, video_dict, **kwargs):
        return save_videos(
            render_results=video_dict,
            save_pth=kwargs["save_pth"],
            num_timestamps=kwargs["num_timestamps"],
            keys=kwargs["keys"],
            num_cams=kwargs["num_cams"],
            fps=kwargs["fps"],
            verbose=kwargs["verbose"],
        )

    def render_data_videos(
        self,
        save_pth: str,
        split: str = "full",
        fps: int = 24,
        verbose=True,
    ):
        """
        Render a video of the supervision.
        """
        pixel_dataset, lidar_dataset = None, None
        if split == "full":
            if self.pixel_source is not None:
                pixel_dataset = self.full_pixel_set
            if self.lidar_source is not None:
                lidar_dataset = self.full_lidar_set
        elif split == "train":
            if self.pixel_source is not None:
                pixel_dataset = self.train_pixel_set
            if self.lidar_source is not None:
                lidar_dataset = self.train_lidar_set
        elif split == "test":
            if self.pixel_source is not None:
                pixel_dataset = self.test_pixel_set
            if self.lidar_source is not None:
                lidar_dataset = self.test_lidar_set
        else:
            raise NotImplementedError(f"Split {split} not supported")

        # pixel source
        rgb_imgs, dynamic_objects = [], []
        sky_masks, feature_pca_colors = [], []
        lidar_depths = []

        for i in trange(
            len(pixel_dataset), desc="Rendering supervision videos", dynamic_ncols=True
        ):
            data_dict = pixel_dataset[i]
            if "pixels" in data_dict:
                rgb_imgs.append(data_dict["pixels"].cpu().numpy())
            if "dynamic_masks" in data_dict:
                dynamic_objects.append(
                    (data_dict["dynamic_masks"].unsqueeze(-1) * data_dict["pixels"])
                    .cpu()
                    .numpy()
                )
            if "sky_masks" in data_dict:
                sky_masks.append(data_dict["sky_masks"].cpu().numpy())
            if "features" in data_dict:
                features = data_dict["features"]
                features = features @ self.pixel_source.feat_dimension_reduction_mat
                features = (features - self.pixel_source.feat_color_min) / (
                    self.pixel_source.feat_color_max - self.pixel_source.feat_color_min
                ).clamp(0, 1)
                feature_pca_colors.append(features.cpu().numpy())
            if lidar_dataset is not None:
                # to deal with asynchronized data
                # find the closest lidar scan to the current image in time
                closest_lidar_idx = self.lidar_source.find_closest_timestep(
                    data_dict["normed_timestamps"].flatten()[0]
                )
                data_dict = lidar_dataset[closest_lidar_idx]
                lidar_points = (
                    data_dict["lidar_origins"]
                    + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
                )
                # project lidar points to the image plane
                # TODO: consider making this a function
                intrinsic_4x4 = torch.nn.functional.pad(
                    self.pixel_source.intrinsics[i], (0, 1, 0, 1)
                )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ self.pixel_source.cam_to_worlds[i].inverse()
                lidar_points = (
                    lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]
                ).T
                depth = lidar_points[:, 2]
                cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < self.pixel_source.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < self.pixel_source.HEIGHT)
                    & (depth > 0)
                )
                depth = depth[valid_mask]
                _cam_points = cam_points[valid_mask]
                depth_map = torch.zeros(
                    self.pixel_source.HEIGHT, self.pixel_source.WIDTH
                ).to(self.device)
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ] = depth.squeeze(-1)
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                # show the depth map on top of the rgb image
                image = rgb_imgs[-1] * (1 - mask) + depth_img * mask
                lidar_depths.append(image)

        video_dict = {
            "gt_rgbs": rgb_imgs,
            "stacked": lidar_depths,
            "gt_feature_pca_colors": feature_pca_colors,
            # "gt_sky_masks": sky_masks,
            "gt_dynamic_objects": dynamic_objects,
        }
        video_dict = {k: v for k, v in video_dict.items() if len(v) > 0}
        # use 3 cameras a row if there are 6 cameras
        return self.save_videos(
            video_dict,
            save_pth=save_pth,
            num_timestamps=self.num_img_timesteps,
            keys=video_dict.keys(),
            num_cams=self.pixel_source.num_cams,
            fps=fps,
            verbose=verbose,
        )

    @property
    def unique_normalized_training_timestamps(self):
        # overwrite this.
        normalized_t = (
            torch.arange(self.pixel_source.num_timesteps, dtype=torch.float32)
            / self.pixel_source.num_timesteps
        )
        return normalized_t[self.train_timesteps]
