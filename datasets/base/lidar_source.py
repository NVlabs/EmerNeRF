import abc
import logging
from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

logger = logging.getLogger()


class SceneLidarSource(abc.ABC):
    """
    The base class for the lidar source of a scene.
    """

    data_cfg: OmegaConf = None
    # the normalized timestamps of all points (normalized to [0, 1]), shape: (num_points,)
    _normalized_timestamps: Tensor = None
    # the timestamps of all points, shape: (num_points,)
    _timestamps: Tensor = None
    # the timesteps of all points, shape: (num_points,)
    #   - the difference between timestamps and timesteps is that
    #     timestamps are the actual timestamps (minus 1e9) of lidar scans,
    #     while timesteps are the integer timestep indices of lidar scans.
    _timesteps: Tensor = None
    # origin of each lidar point, shape: (num_points, 3)
    origins: Tensor = None
    # unit direction of each lidar point, shape: (num_points, 3)
    directions: Tensor = None
    # range of each lidar point, shape: (num_points,)
    ranges: Tensor = None
    # the transformation matrices from lidar to world coordinate system,
    lidar_to_worlds: Tensor = None
    # the indices of the lidar scans that are cached
    cached_indices: Tensor = None
    cached_origins: Tensor = None
    cached_directions: Tensor = None
    cached_ranges: Tensor = None
    cached_normalized_timestamps: Tensor = None

    def __init__(
        self,
        lidar_data_config: OmegaConf,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # hold the config of the lidar data
        self.data_cfg = lidar_data_config
        self.device = device

    @abc.abstractmethod
    def create_all_filelist(self) -> None:
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        raise NotImplementedError

    def load_data(self):
        self.load_calibrations()
        self.load_lidar()
        logger.info("[Lidar] All Lidar Data loaded.")

    def to(self, device: torch.device) -> "SceneLidarSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self.origins is not None:
            self.origins = self.origins.to(device)
        if self.directions is not None:
            self.directions = self.directions.to(device)
        if self.ranges is not None:
            self.ranges = self.ranges.to(device)
        if self._timestamps is not None:
            self._timestamps = self._timestamps.to(device)
        if self._timesteps is not None:
            self._timesteps = self._timesteps.to(device)
        if self._normalized_timestamps is not None:
            self._normalized_timestamps = self._normalized_timestamps.to(device)
        if self.lidar_to_worlds is not None:
            self.lidar_to_worlds = self.lidar_to_worlds.to(device)
        return self

    @abc.abstractmethod
    def load_calibrations(self) -> None:
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_lidar(self) -> None:
        """
        Load the lidar data of the dataset from the filelist.
        """
        raise NotImplementedError

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            we assume the lidar points are already in the world coordinate system
            we first downsample the lidar points, then compute the aabb by taking the
            given percentiles of the lidar coordinates in each dimension.
        """
        assert (
            self.origins is not None
            and self.directions is not None
            and self.ranges is not None
        ), "Lidar points not loaded, cannot compute aabb."
        logger.info("[Lidar] Computing auto AABB based on downsampled lidar points....")

        lidar_pts = self.origins + self.directions * self.ranges

        # downsample the lidar points by uniformly sampling a subset of them
        lidar_pts = lidar_pts[
            torch.randperm(len(lidar_pts))[
                : int(len(lidar_pts) / self.data_cfg.lidar_downsample_factor)
            ]
        ]
        # compute the aabb by taking the given percentiles of the lidar coordinates in each dimension
        aabb_min = torch.quantile(lidar_pts, self.data_cfg.lidar_percentile, dim=0)
        aabb_max = torch.quantile(lidar_pts, 1 - self.data_cfg.lidar_percentile, dim=0)
        del lidar_pts
        torch.cuda.empty_cache()

        # usually the lidar's height is very small, so we slightly increase the height of the aabb
        if aabb_max[-1] < 20:
            aabb_max[-1] = 20.0
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Lidar] Auto AABB from LiDAR: {aabb}")
        return aabb

    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of lidar timestamps in the dataset,
            usually the number of captured lidar scans.
        """
        return len(self.timesteps.unique())

    @property
    def timesteps(self) -> Tensor:
        """
        Returns:
            the integer timestep indices of each lidar timestamp,
            shape: (num_lidar_points,)
        Note:
            the difference between timestamps and timesteps is that
            timestamps are the actual timestamps (minus 1e9) of the lidar scans,
            while timesteps are the integer timestep indices of the lidar scans.
        """
        return self._timesteps

    @property
    def timestamps(self) -> Tensor:
        """
        Returns:
            the actual timestamps (minus 1e9) of the lidar scans.
            shape: (num_lidar_points,)
        """
        return self._timestamps

    @property
    def normalized_timestamps(self) -> Tensor:
        """
        Returns:
            the normalized timestamps of the lidar scans
            (normalized to the range [0, 1]).
            shape: (num_lidar_points,)
        """
        return self._normalized_timestamps

    @property
    def unique_normalized_timestamps(self) -> Tensor:
        """
        Returns:
            the unique normalized timestamps of the lidar scans
            (normalized to the range [0, 1]).
            shape: (num_timesteps,)
        """
        return self._unique_normalized_timestamps

    def register_normalized_timestamps(self, normalized_timestamps: Tensor) -> None:
        """
        Register the normalized timestamps of the lidar scans.
        Args:
            normalized_timestamps: the normalized timestamps of the lidar scans
                (normalized to the range [0, 1]).
                shape: (num_lidar_points,)
        Note:
            we normalize the lidar timestamps together with the image timestamps,
            so that the both the lidar and image timestamps are in the range [0, 1].
        """
        assert normalized_timestamps.size(0) == self.origins.size(
            0
        ), "The number of lidar points and the number of normalized timestamps must match."
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

    def sample_uniform_rays(
        self,
        num_rays: int,
        candidate_indices: Tensor = None,
    ) -> Tensor:
        """
        Sample a batch of rays uniformly from the dataset.
        Args:
            num_rays: the number of rays to sample.
            candidate_indices: the indices of the lidar scans to sample from.
                If None, sample from all the lidar scans.
                If not None, sample from the given lidar scans.
        Returns:
            lidar_idx: the indices of the sampled lidar points.
                shape: (num_rays,)
        """
        if candidate_indices is None:
            return torch.randint(
                0, len(self.origins), size=(num_rays,), device=self.device
            )
        else:
            if not isinstance(candidate_indices, Tensor):
                candidate_indices = torch.tensor(candidate_indices, device=self.device)
                if self.cached_indices is None:
                    self.cached_indices = candidate_indices
                    mask = self.timesteps.new_zeros(
                        self.timesteps.size(0), dtype=torch.bool
                    )  # Create a mask of False
                    for index in self.cached_indices:
                        mask |= (
                            self.timesteps == index
                        )  # Set mask values to True where timesteps match an index
                    self.cached_origins = self.origins[mask]
                    self.cached_directions = self.directions[mask]
                    self.cached_ranges = self.ranges[mask]
                    self.cached_normalized_timestamps = self.normalized_timestamps[mask]
            if not torch.equal(candidate_indices, self.cached_indices):
                print("Recomputing cached indices")
                self.cached_indices = candidate_indices
                mask = self.timesteps.new_zeros(
                    self.timesteps.size(0), dtype=torch.bool
                )  # Create a mask of False
                for index in self.cached_indices:
                    mask |= (
                        self.timesteps == index
                    )  # Set mask values to True where timesteps match an index
                self.cached_origins = self.origins[mask]
                self.cached_directions = self.directions[mask]
                self.cached_ranges = self.ranges[mask]
                self.cached_normalized_timestamps = self.normalized_timestamps[mask]
            random_idx = torch.randint(
                0,
                len(self.cached_origins),
                size=(num_rays,),
                device=self.device,
            )
            return random_idx

    def get_train_rays(
        self,
        num_rays: int,
        candidate_indices: Tensor = None,
    ) -> Dict[str, Tensor]:
        """
        Get a batch of rays for training.
        Args:
            num_rays: the number of rays to sample.
            candidate_indices: the indices of the lidar scans to sample from.
                If None, sample from all the lidar scans.
                If not None, sample from the given lidar scans.
        Returns:
            a dict of the sampled rays.
        """
        lidar_idx = self.sample_uniform_rays(
            num_rays=num_rays, candidate_indices=candidate_indices
        )
        origins = self.cached_origins[lidar_idx]
        directions = self.cached_directions[lidar_idx]
        ranges = self.cached_ranges[lidar_idx]
        normalized_timestamps = self.cached_normalized_timestamps[lidar_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_timestamps": normalized_timestamps,
        }

    def get_render_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_timestamps = self.normalized_timestamps[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_timestamps": normalized_timestamps,
        }
