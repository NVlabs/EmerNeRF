from typing import List, Union

import torch
import torch.nn.functional as F

from .lidar_source import SceneLidarSource
from .pixel_source import ScenePixelSource


class SplitWrapper(torch.utils.data.Dataset):

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: Union[ScenePixelSource, SceneLidarSource],
        split_indices: List[int] = None,
        split: str = "train",
        ray_batch_size: int = 4096,
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.split = split
        self.ray_batch_size = ray_batch_size

    def __getitem__(self, idx) -> dict:
        if self.split == "train":
            # randomly sample rays from the training set
            return self.datasource.get_train_rays(
                num_rays=self.ray_batch_size,
                candidate_indices=self.split_indices,
            )
        else:
            # return all rays for the given index
            return self.datasource.get_render_rays(self.split_indices[idx])

    def __len__(self) -> int:
        if self.split == "train":
            return self.num_iters
        else:
            return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
