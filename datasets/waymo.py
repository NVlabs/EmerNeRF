import logging
import os
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import trange

from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from datasets.utils import voxel_coords_to_world_coords
from radiance_fields.video_utils import depth_visualizer, save_videos, scene_flow_to_rgb

logger = logging.getLogger()


class WaymoPixelSource(ScenePixelSource):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )

    def __init__(
        self,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create file lists for all data files.
        e.g., img files, feature files, etc.
        """
        # ---- define camera list ---- #
        # 0: front, 1: front_left, 2: front_right, 3: side_left, 4: side_right
        if self.num_cams == 1:
            self.camera_list = [0]
        elif self.num_cams == 3:
            self.camera_list = [1, 0, 2]
        elif self.num_cams == 5:
            self.camera_list = [3, 1, 0, 2, 4]
        else:
            raise NotImplementedError(
                f"num_cams: {self.num_cams} not supported for waymo dataset"
            )

        # ---- define filepaths ---- #
        img_filepaths, feat_filepaths = [], []
        dynamic_mask_filepaths, sky_mask_filepaths = [], []

        # Note: we assume all the files in waymo dataset are synchronized
        for t in range(self.start_timestep, self.end_timestep):
            for cam_idx in self.camera_list:
                img_filepaths.append(
                    os.path.join(self.data_path, "images", f"{t:03d}_{cam_idx}.jpg")
                )
                dynamic_mask_filepaths.append(
                    os.path.join(
                        self.data_path, "dynamic_masks", f"{t:03d}_{cam_idx}.png"
                    )
                )
                sky_mask_filepaths.append(
                    os.path.join(self.data_path, "sky_masks", f"{t:03d}_{cam_idx}.png")
                )
                feat_filepaths.append(
                    os.path.join(
                        self.data_path,
                        self.data_cfg.feature_model_type,
                        f"{t:03d}_{cam_idx}.npy",
                    )
                )
        self.img_filepaths = np.array(img_filepaths)
        self.dynamic_mask_filepaths = np.array(dynamic_mask_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.feat_filepaths = np.array(feat_filepaths)

    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # to store per-camera intrinsics and extrinsics
        _intrinsics = []
        cam_to_egos = []
        for i in range(self.num_cams):
            # load camera intrinsics
            # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
            # ====!! we did not use distortion parameters for simplicity !!====
            # to be improved!!
            intrinsic = np.loadtxt(
                os.path.join(self.data_path, "intrinsics", f"{i}.txt")
            )
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            # scale intrinsics w.r.t. load size
            fx, fy = (
                fx * self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[i][1],
                fy * self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[i][0],
            )
            cx, cy = (
                cx * self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[i][1],
                cy * self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[i][0],
            )
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            _intrinsics.append(intrinsic)

            # load camera extrinsics
            cam_to_ego = np.loadtxt(
                os.path.join(self.data_path, "extrinsics", f"{i}.txt")
            )
            # because we use opencv coordinate system to generate camera rays,
            # we need a transformation matrix to covnert rays from opencv coordinate
            # system to waymo coordinate system.
            # opencv coordinate system: x right, y down, z front
            # waymo coordinate system: x front, y left, z up
            cam_to_egos.append(cam_to_ego @ self.OPENCV2DATASET)

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, cam_ids = [], []
        # ===! for waymo, we simplify timestamps as the time indices
        timestamps, timesteps = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            for cam_id in self.camera_list:
                cam_ids.append(cam_id)
                # transformation:
                #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
                cam2world = ego_to_world @ cam_to_egos[cam_id]
                cam_to_worlds.append(cam2world)
                intrinsics.append(_intrinsics[cam_id])
                # ===! we use time indices as the timestamp for waymo dataset for simplicity
                # ===! we can use the actual timestamps if needed
                # to be improved
                timestamps.append(t - self.start_timestep)
                timesteps.append(t - self.start_timestep)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        self.cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # the underscore here is important.
        self._timestamps = torch.from_numpy(np.stack(timestamps, axis=0)).float()
        self._timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()


class WaymoLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, "lidar", f"{t:03d}.bin")
            )
        self.lidar_filepaths = np.array(lidar_filepaths)

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        # Note that in the Waymo Open Dataset, the lidar coordinate system is the same
        # as the vehicle coordinate system
        lidar_to_worlds = []

        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(
            np.stack(lidar_to_worlds, axis=0)
        ).float()

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        # flow/ground info are used for evaluation only
        flows, flow_classes, grounds = [], [], []
        # in waymo, we simplify timestamps as the time indices
        timestamps, timesteps = [], []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):
            # each lidar_info contains an Nx14 array
            # from left to right:
            # origins: 3d, points: 3d, flows: 3d, flow_class: 1d,
            # ground_labels: 1d, intensities: 1d, elongations: 1d, laser_ids: 1d
            lidar_info = np.memmap(
                self.lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length

            # select lidar points based on the laser id
            if self.data_cfg.only_use_top_lidar:
                # laser_ids: 0: TOP, 1: FRONT, 2: SIDE_LEFT, 3: SIDE_RIGHT, 4: REAR
                lidar_info = lidar_info[lidar_info[:, 13] == 0]

            lidar_origins = torch.from_numpy(lidar_info[:, :3]).float()
            lidar_points = torch.from_numpy(lidar_info[:, 3:6]).float()
            lidar_ids = torch.from_numpy(lidar_info[:, 13]).float()
            lidar_flows = torch.from_numpy(lidar_info[:, 6:9]).float()
            lidar_flow_classes = torch.from_numpy(lidar_info[:, 9]).long()
            ground_labels = torch.from_numpy(lidar_info[:, 10]).long()
            # we don't collect intensities and elongations for now

            # select lidar points based on a truncated ego-forward-directional range
            # this is to make sure most of the lidar points are within the range of the camera
            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (
                    lidar_points[:, 0] > self.data_cfg.truncated_min_range
                )
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            lidar_flows = lidar_flows[valid_mask]
            lidar_flow_classes = lidar_flow_classes[valid_mask]
            ground_labels = ground_labels[valid_mask]
            # transform lidar points from lidar coordinate system to world coordinate system
            lidar_origins = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            # scene flows are in the lidar coordinate system, so we need to rotate them
            lidar_flows = (self.lidar_to_worlds[t][:3, :3] @ lidar_flows.T).T
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            # we use time indices as the timestamp for waymo dataset
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)
            flows.append(lidar_flows)
            flow_classes.append(lidar_flow_classes)
            grounds.append(ground_labels)
            # we use time indices as the timestamp for waymo dataset
            timestamps.append(lidar_timestamp)
            timesteps.append(lidar_timestamp)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )
        logger.info("Filter condition:")
        logger.info(f"  only_use_top_lidar: {self.data_cfg.only_use_top_lidar}")
        logger.info(f"  truncated_max_range: {self.data_cfg.truncated_max_range}")
        logger.info(f"  truncated_min_range: {self.data_cfg.truncated_min_range}")

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.laser_ids = torch.cat(laser_ids, dim=0)
        # becasue the flows here are velocities (m/s), and the fps of the lidar is 10,
        # we need to divide the velocities by 10 to get the displacements/flows
        # between two consecutive lidar scans
        self.flows = torch.cat(flows, dim=0) / 10.0
        self.flow_classes = torch.cat(flow_classes, dim=0)
        self.grounds = torch.cat(grounds, dim=0).bool()

        # the underscore here is important.
        self._timestamps = torch.cat(timestamps, dim=0)
        self._timesteps = torch.cat(timesteps, dim=0)

    def to(self, device: torch.device):
        super().to(device)
        self.flows = self.flows.to(device)
        self.flow_classes = self.flow_classes.to(device)
        self.grounds = self.grounds.to(self.device)

    def get_render_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Override the base class function to add more information to the render rays.
        """
        return {
            "lidar_origins": self.origins[self.timesteps == time_idx],
            "lidar_viewdirs": self.directions[self.timesteps == time_idx],
            "lidar_ranges": self.ranges[self.timesteps == time_idx],
            # normalized timestamps between 0 and 1
            "lidar_normed_timestamps": self.normalized_timestamps[
                self.timesteps == time_idx
            ],
            "lidar_flow": self.flows[self.timesteps == time_idx],
            "lidar_flow_class": self.flow_classes[self.timesteps == time_idx],
            "lidar_ground": self.grounds[self.timesteps == time_idx],
        }


class WaymoDataset(SceneDataset):
    dataset: str = "waymo"

    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)
        self.data_path = os.path.join(self.data_cfg.data_root, f"{self.scene_idx:03d}")
        assert self.data_cfg.dataset == "waymo"
        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"

        # ---- find the number of synchronized frames ---- #
        if self.data_cfg.end_timestep == -1:
            num_files = len(os.listdir(os.path.join(self.data_path, "ego_pose")))
            end_timestep = num_files - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        # to make sure the last timestep is included
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep

        # ---- create data source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()
        self.aabb = self.get_aabb()

        # ---- define train and test indices ---- #
        # note that the timestamps of the pixel source and the lidar source are the same in waymo dataset
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
            if len(self.test_indices) > 0:
                test_pixel_set = SplitWrapper(
                    datasource=self.pixel_source,
                    # test_indices are img indices, so the length is num_cams * num_test_timesteps
                    split_indices=self.test_indices,
                    split="test",
                    ray_batch_size=self.data_cfg.ray_batch_size,
                )
        if self.lidar_source is not None:
            train_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                # train_timesteps are lidar indices, so the length is num_train_timesteps
                split_indices=self.train_timesteps,
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
            if len(self.test_indices) > 0:
                test_lidar_set = SplitWrapper(
                    datasource=self.lidar_source,
                    # test_timesteps are lidar indices, so the length is num_test_timesteps
                    split_indices=self.test_timesteps,
                    split="test",
                    ray_batch_size=self.data_cfg.ray_batch_size,
                )
        pixel_set = (train_pixel_set, test_pixel_set, full_pixel_set)
        lidar_set = (train_lidar_set, test_lidar_set, full_lidar_set)
        return pixel_set, lidar_set

    def build_data_source(self):
        """
        Create the data source for the dataset.
        """
        pixel_source, lidar_source = None, None
        # to collect all timestamps from pixel source and lidar source
        all_timestamps = []
        # ---- create pixel source ---- #
        load_pixel = (
            self.data_cfg.pixel_source.load_rgb
            or self.data_cfg.pixel_source.load_sky_mask
            or self.data_cfg.pixel_source.load_dynamic_mask
            or self.data_cfg.pixel_source.load_feature
        )
        if load_pixel:
            pixel_source = WaymoPixelSource(
                self.data_cfg.pixel_source,
                self.data_path,
                self.start_timestep,
                self.end_timestep,
                device=self.device,
            )
            pixel_source.to(self.device)
            # collect img timestamps
            all_timestamps.append(pixel_source.timestamps)
        # ---- create lidar source ---- #
        if self.data_cfg.lidar_source.load_lidar:
            lidar_source = WaymoLiDARSource(
                self.data_cfg.lidar_source,
                self.data_path,
                self.start_timestep,
                self.end_timestep,
                device=self.device,
            )
            lidar_source.to(self.device)
            # collect lidar timestamps
            all_timestamps.append(lidar_source.timestamps)

        assert len(all_timestamps) > 0, "No data source is loaded"
        all_timestamps = torch.cat(all_timestamps, dim=0)
        # normalize the timestamps jointly for pixel source and lidar source
        # so that the normalized timestamps are between 0 and 1
        all_timestamps = (all_timestamps - all_timestamps.min()) / (
            all_timestamps.max() - all_timestamps.min()
        )
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
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                # it makes no sense to have test timesteps before the start timestep
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

        # Again, training and testing indices are indices into the full dataset
        # train_indices are img indices, so the length is num_cams * num_train_timesteps
        # but train_timesteps are timesteps, so the length is num_train_timesteps (len(unique_train_timestamps))
        return train_timesteps, test_timesteps, train_indices, test_indices

    def get_occ(self, index: int):
        """
        Get the Occ3D data of the scene at the given index.
        """
        # from: https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-waymo
        # The dataset contains 15 classes. The definition of classes from 0 to 14 is
        # 0: TYPE_GENERALOBJECT, 1: TYPE_VEHICLE, 2: TYPE_PEDESTRIAN, 3: TYPE_SIGN,
        # 4: TYPE_CYCLIST, 5: TYPE_TRAFFIC_LIGHT, 6: TYPE_POLE, 7: TYPE_CONSTRUCTION_CONE,
        # 8: TYPE_BICYCLE, 9: TYPE_MOTORCYCLE, 10: TYPE_BUILDING, 11: TYPE_VEGETATION,
        # 12: TYPE_TREE_TRUNK, 13: TYPE_ROAD, 14: TYPE_WALKABLE.
        self.label_mapping = {
            0: "general_obj",
            1: "vehicle",
            2: "pedestrian",
            3: "sign",
            4: "cyclist",
            5: "traffic_light",
            6: "pole",
            7: "construction_cone",
            8: "bicyle",
            9: "motorcycle",
            10: "building",
            11: "vegetation",
            12: "tree_trunck",
            13: "road",
            14: "walkable",
        }
        if self.data_cfg.occ_source.voxel_size == 0.4:
            occ_path = f"{self.data_path}/occ3d/{index:03d}_04.npz"
            occupancy_resolution = [100, 200, 16]
            occupancy_aabb_min = [0, -40, -1]
            occupancy_aabb_max = [40, 40, 5.4]
        elif self.data_cfg.occ_source.voxel_size == 0.1:
            occ_path = f"{self.data_path}/occ3d/{index:03d}.npz"
            occupancy_resolution = [800, 1600, 64]
            occupancy_aabb_min = [0, -80, -5]
            occupancy_aabb_max = [80, 80, 7.8]
        else:
            raise NotImplementedError(
                f"voxel size {self.data_cfg.occ_source.voxel_size} not supported"
            )

        if not os.path.exists(occ_path):
            raise FileNotFoundError(f"{occ_path} does not exist")

        # loading the occupancy grid
        gt_occ = np.load(occ_path)

        # np.unique(gt_occ['voxel_label']): array([ 0,  1,  2,  3,  6,  8,  9, 10, 11, 12, 13, 14, 23], dtype=uint8)
        semantic_labels = gt_occ["voxel_label"]

        # final voxel_state will indicate what voxels are visible from the camera
        mask_camera = gt_occ["final_voxel_state"]

        # we don't have back-cameras, so we remove the back part of the grid
        semantic_labels = semantic_labels[len(semantic_labels) // 2 :, :, :]
        mask_camera = mask_camera[len(mask_camera) // 2 :, :, :]

        # semantic_labels == 23 means the free space, i.e. empty
        semantic_labels[semantic_labels == 23] = 15
        # mask_camera == 0 means invisible from the camera
        semantic_labels[mask_camera == 0] = 15

        semantic_labels = (
            torch.from_numpy(semantic_labels.copy()).long().to(self.device)
        )
        # compute the coordinates and labels of the occupied voxels
        occ_coords = torch.nonzero(semantic_labels != 15).float()
        occ_labels = semantic_labels[semantic_labels != 15]

        # transform the coordinates from voxel space to world space
        ego_occ_coords = voxel_coords_to_world_coords(
            occupancy_aabb_min,
            occupancy_aabb_max,
            occupancy_resolution,
            points=occ_coords,
        ).to(self.device)
        world_occ_coords = (
            self.lidar_source.lidar_to_worlds[index][:3, :3] @ ego_occ_coords.T
            + self.lidar_source.lidar_to_worlds[index][:3, 3:4]
        ).T
        normed_timestamps = (
            torch.ones_like(world_occ_coords[..., 0])
            * index
            / (self.lidar_source.num_timesteps + 1e-6 - 1)
        )
        return world_occ_coords, occ_labels, normed_timestamps

    def get_valid_lidar_mask(self, lidar_timestep: int, data_dict: dict):
        # filter out the lidar points that are not visible from the camera
        lidar_points = (
            data_dict["lidar_origins"]
            + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
        )
        valid_mask = torch.zeros_like(lidar_points[:, 0]).bool()
        # project lidar points to the image plane
        for i in range(self.pixel_source.num_cams):
            img_idx = lidar_timestep * self.pixel_source.num_cams + i
            intrinsic_4x4 = torch.nn.functional.pad(
                self.pixel_source.intrinsics[img_idx], (0, 1, 0, 1)
            )
            intrinsic_4x4[3, 3] = 1.0
            lidar2img = (
                intrinsic_4x4 @ self.pixel_source.cam_to_worlds[img_idx].inverse()
            )
            projected_points = (
                lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]
            ).T
            depth = projected_points[:, 2]
            cam_points = projected_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)
            current_valid_mask = (
                (cam_points[:, 0] >= 0)
                & (cam_points[:, 0] < self.pixel_source.WIDTH)
                & (cam_points[:, 1] >= 0)
                & (cam_points[:, 1] < self.pixel_source.HEIGHT)
                & (depth > 0)
            )
            valid_mask = valid_mask | current_valid_mask
        return valid_mask

    def save_videos(self, video_dict: dict, **kwargs):
        """
        Save the a video of the data.
        """
        return save_videos(
            render_results=video_dict,
            save_pth=kwargs["save_pth"],
            num_timestamps=kwargs["num_timestamps"],
            keys=kwargs["keys"],
            num_cams=kwargs["num_cams"],
            fps=kwargs["fps"],
            verbose=kwargs["verbose"],
            save_seperate_video=kwargs["save_seperate_video"],
        )

    def render_data_videos(
        self,
        save_pth: str,
        split: str = "full",
        fps: int = 24,
        verbose=True,
    ):
        """
        Render a video of data.
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
        lidar_depths, flow_colors = [], []

        for i in trange(
            len(pixel_dataset), desc="Rendering data videos", dynamic_ncols=True
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
                # use registered parameters to normalize the features for visualization
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

                # project lidar flows to the image plane
                flow_img = torch.zeros(
                    self.pixel_source.HEIGHT, self.pixel_source.WIDTH, 3
                ).to(self.device)
                # to examine whether the ground labels are correct
                valid_mask = valid_mask & (~data_dict["lidar_ground"])
                _cam_points = cam_points[valid_mask]
                # final color:
                #  white if no flow, black if ground, and flow color otherwise
                flow_color = scene_flow_to_rgb(
                    data_dict["lidar_flow"][valid_mask],
                    background="bright",
                    flow_max_radius=1.0,
                )
                flow_img[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ] = flow_color
                flow_img = flow_img.cpu().numpy()
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                # show the flow on top of the rgb image
                image = rgb_imgs[-1] * (1 - mask) + flow_img * mask
                flow_colors.append(image)

        video_dict = {
            "gt_rgbs": rgb_imgs,
            "stacked": lidar_depths,
            "flow_colors": flow_colors,
            "gt_feature_pca_colors": feature_pca_colors,
            # "gt_dynamic_objects": dynamic_objects,
            # "gt_sky_masks": sky_masks,
        }
        video_dict = {k: v for k, v in video_dict.items() if len(v) > 0}
        return self.save_videos(
            video_dict,
            save_pth=save_pth,
            num_timestamps=self.num_img_timesteps,
            keys=video_dict.keys(),
            num_cams=self.pixel_source.num_cams,
            fps=fps,
            verbose=verbose,
            save_seperate_video=False,
        )
