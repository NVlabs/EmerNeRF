from typing import List, Tuple, Union

import numpy as np
import torch
from pyquaternion import Quaternion
from torch import Tensor


def voxel_coords_to_world_coords(
    aabb_min: Union[Tensor, List[float]],
    aabb_max: Union[Tensor, List[float]],
    voxel_resolution: Union[Tensor, List[int]],
    points: Union[Tensor, List[float]] = None,
) -> Tensor:
    """
    Converts voxel coordinates to world coordinates.

    Args:
        aabb_min (Union[Tensor, List[float]]): Minimum coordinates of the axis-aligned bounding box (AABB) of the voxel grid.
        aabb_max (Union[Tensor, List[float]]): Maximum coordinates of the AABB of the voxel grid.
        voxel_resolution (Union[Tensor, List[int]]): Number of voxels in each dimension of the voxel grid.
        points (Union[Tensor, List[float]], optional):
            Tensor of voxel coordinates to convert to world coordinates.
            If None, returns a grid of world coordinates. Defaults to None.
    Returns:
        Tensor: Tensor of world coordinates.
    """
    aabb_min = torch.tensor(aabb_min) if isinstance(aabb_min, List) else aabb_min
    aabb_max = torch.tensor(aabb_max) if isinstance(aabb_max, List) else aabb_max
    voxel_resolution = (
        torch.tensor(voxel_resolution)
        if isinstance(voxel_resolution, List)
        else voxel_resolution
    )

    if points is None:
        x, y, z = torch.meshgrid(
            torch.linspace(aabb_min[0], aabb_max[0], voxel_resolution[0]),
            torch.linspace(aabb_min[1], aabb_max[1], voxel_resolution[1]),
            torch.linspace(aabb_min[2], aabb_max[2], voxel_resolution[2]),
        )
        return torch.stack([x, y, z], dim=-1)
    else:
        points = torch.tensor(points) if isinstance(points, List) else points

        # Compute voxel size
        voxel_size = (aabb_max - aabb_min) / voxel_resolution

        # Convert voxel coordinates to world coordinates
        world_coords = aabb_min.to(points.device) + points * voxel_size.to(
            points.device
        )
        return world_coords


def world_coords_to_voxel_coords(
    point: Union[Tensor, List[float]],
    aabb_min: Union[Tensor, List[float]],
    aabb_max: Union[Tensor, List[float]],
    voxel_resolution: Union[Tensor, List[int]],
) -> Tensor:
    """
    Convert a point in world coordinates to voxel coordinates.

    Args:
        point (Union[Tensor, List[float]]): The point to convert.
        aabb_min (Union[Tensor, List[float]]): The minimum corner of the axis-aligned bounding box (AABB) of the voxel grid.
        aabb_max (Union[Tensor, List[float]]): The maximum corner of the AABB of the voxel grid.
        voxel_resolution (Union[Tensor, List[int]]): The number of voxels in each dimension of the voxel grid.

    Returns:
        Tensor: The voxel coordinates of the point.
    """
    # Convert lists to tensors if necessary
    point = torch.tensor(point) if isinstance(point, List) else point
    aabb_min = torch.tensor(aabb_min) if isinstance(aabb_min, List) else aabb_min
    aabb_max = torch.tensor(aabb_max) if isinstance(aabb_max, List) else aabb_max
    voxel_resolution = (
        torch.tensor(voxel_resolution)
        if isinstance(voxel_resolution, List)
        else voxel_resolution
    )

    # Compute the size of each voxel
    voxel_size = (aabb_max - aabb_min) / voxel_resolution

    # Compute the voxel index for the given point
    voxel_coords = ((point - aabb_min) / voxel_size).long()

    return voxel_coords


def interpolate_matrices(T1, T2, alpha):
    """
    Interpolate between two transformation matrices using given alpha.
    :param T1: First transformation matrix.
    :param T2: Second transformation matrix.
    :param alpha: Interpolation weight (0 <= alpha <= 1).
    :return: Interpolated transformation matrix.
    """

    # Extract rotation and translation components
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]

    # Interpolate translation linearly
    t_interp = alpha * t1 + (1 - alpha) * t2

    # Convert rotation matrices to quaternions
    q1 = Quaternion(matrix=R1.cpu().numpy(), rtol=1e-5, atol=1e-5)
    q2 = Quaternion(matrix=R2.cpu().numpy(), rtol=1e-5, atol=1e-5)

    # Spherical linear interpolation (slerp) for quaternions
    q_interp = Quaternion.slerp(q1, q2, alpha)

    # Convert interpolated quaternion back to matrix
    R_interp = q_interp.rotation_matrix

    # Construct interpolated transformation matrix
    T_interp = torch.eye(4, device=T1.device)
    T_interp[:3, :3] = torch.tensor(R_interp)
    T_interp[:3, 3] = t_interp
    return T_interp


def get_ground(pts):
    """
    This function performs ground removal on a point cloud.
    Modified from https://github.com/tusen-ai/LiDAR_SOT/blob/main/waymo_data/data_preprocessing/ground_removal.py

    Args:
        pts (numpy.ndarray or torch.Tensor): The input point cloud.

    Returns:
        numpy.ndarray or torch.Tensor: A boolean array indicating whether each point is ground or not.
    """
    if isinstance(pts, np.ndarray):
        pts = torch.tensor(pts)
        is_numpy = True
    else:
        is_numpy = False
    original_device = pts.device
    pts = pts.to(torch.device("cuda"))
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    lpr = torch.mean(pts_sort[:num_lpr_, 2])
    pts_g = pts_sort[pts_sort[:, 2] < lpr + th_seeds_, :]
    normal_ = torch.zeros(3)
    for i in range(n_iter):
        mean = torch.mean(pts_g, axis=0)[:3]
        xx = torch.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = torch.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = torch.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = torch.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = torch.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = torch.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = torch.tensor(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=torch.float32,
            device=pts.device,
        )
        U, S, V = torch.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3] @ normal_.unsqueeze(1)
        pts_g = pts[result.squeeze(-1) < th_dist_d_]
    ground_label = result < th_dist_d_
    if is_numpy:
        return ground_label.cpu().numpy()
    else:
        return ground_label.to(original_device)


def get_ground_np(pts):
    """
    This function performs ground removal on a point cloud.
    Modified from https://github.com/tusen-ai/LiDAR_SOT/blob/main/waymo_data/data_preprocessing/ground_removal.py

    Args:
        pts (numpy.ndarray): The input point cloud.

    Returns:
        numpy.ndarray: A boolean array indicating whether each point is ground or not.
    """
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    lpr = np.mean(pts_sort[:num_lpr_, 2])
    pts_g = pts_sort[pts_sort[:, 2] < lpr + th_seeds_, :]
    normal_ = np.zeros(3)
    for i in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=np.float32,
        )
        U, S, V = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3] @ normal_[..., np.newaxis]
        pts_g = pts[result.squeeze(-1) < th_dist_d_]
    ground_label = result < th_dist_d_
    return ground_label
