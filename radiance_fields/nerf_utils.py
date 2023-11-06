import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

logger = logging.getLogger()


def contract(
    x: Tensor,
    aabb: Tensor,
    ord: Union[str, int] = None,
) -> Tensor:
    """
    Contract the input tensor to the unit cube using piecewise projective function.
    """
    # similar to the one in MeRF paper
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)  # 0~1
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    x = torch.where(mag < 1, x, (2 - 1 / mag) * (x / mag))
    x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    return x


def find_topk_nearby_timesteps(
    original: Tensor, query: Tensor, topk: int = 2, return_indices: bool = False
) -> Tensor:
    """
    Find the closest two closest in `original` tensor for each value in `query` tensor.

    Parameters:
    - original (torch.Tensor): Original tensor of timesteps.
    - query (torch.Tensor): Query tensor of timesteps for which closest indices are to be found.

    Returns:
    - torch.Tensor: Indices in the original tensor that are the two closest to each timestep in the query tensor.
    """

    # Expand dimensions of tensors to compute pairwise distances
    original_expanded = original.unsqueeze(0)
    query_expanded = query.unsqueeze(1)

    # Compute pairwise absolute differences
    abs_diffs = torch.abs(original_expanded - query_expanded)

    # Find indices of the two minimum absolute differences along the dimension of the original tensor
    _, closest_indices = torch.topk(abs_diffs, k=topk, dim=1, largest=False)
    if return_indices:
        return original[closest_indices], closest_indices
    return original[closest_indices]


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply
