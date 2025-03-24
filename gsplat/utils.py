import math
import struct

import torch
import torch.nn.functional as F
import open3d as o3d
from torch import Tensor
import numpy as np


def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    print(f"Saving ply to {dir}")

    # Convert all tensors to numpy arrays
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    # Extract data arrays
    means = numpy_data["means"]
    scales = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]

    # Handle colors or spherical harmonics based on whether colors is provided
    if colors is not None:
        colors = colors.detach().cpu().numpy()
    else:
        sh0 = numpy_data["sh0"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()
        shN = numpy_data["shN"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()

    # Create a mask to identify rows with NaN or Inf in any of the numpy_data arrays
    invalid_mask = (
            np.isnan(means).any(axis=1)
            | np.isinf(means).any(axis=1)
            | np.isnan(scales).any(axis=1)
            | np.isinf(scales).any(axis=1)
            | np.isnan(quats).any(axis=1)
            | np.isinf(quats).any(axis=1)
            | np.isnan(opacities).any(axis=0)
            | np.isinf(opacities).any(axis=0)
            | np.isnan(sh0).any(axis=1)
            | np.isinf(sh0).any(axis=1)
            | np.isnan(shN).any(axis=1)
            | np.isinf(shN).any(axis=1)
    )

    # Filter out rows with NaNs or Infs from all data arrays
    means = means[~invalid_mask]
    scales = scales[~invalid_mask]
    quats = quats[~invalid_mask]
    opacities = opacities[~invalid_mask]

    # Initialize ply_data with positions and normals
    ply_data = {
        "positions": o3d.core.Tensor(means, dtype=o3d.core.Dtype.Float32),
        "normals": o3d.core.Tensor(np.zeros_like(means), dtype=o3d.core.Dtype.Float32),
    }

    # Add features
    if colors is not None:
        colors = colors[~invalid_mask]
        # Use provided colors, converted to SH coefficients
        for j in range(colors.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                (colors[:, j: j + 1] - 0.5) / 0.2820947917738781,
                dtype=o3d.core.Dtype.Float32,
                )
    else:
        sh0 = sh0[~invalid_mask]
        shN = shN[~invalid_mask]
        # Use spherical harmonics (sh0 for DC, shN for rest)
        for j in range(sh0.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                sh0[:, j: j + 1], dtype=o3d.core.Dtype.Float32
            )
        for j in range(shN.shape[1]):
            ply_data[f"f_rest_{j}"] = o3d.core.Tensor(
                shN[:, j: j + 1], dtype=o3d.core.Dtype.Float32
            )

    # Add opacity
    ply_data["opacity"] = o3d.core.Tensor(
        opacities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32
    )

    # Add scales
    for i in range(scales.shape[1]):
        ply_data[f"scale_{i}"] = o3d.core.Tensor(
            scales[:, i: i + 1], dtype=o3d.core.Dtype.Float32
        )

    # Add rotations
    for i in range(quats.shape[1]):
        ply_data[f"rot_{i}"] = o3d.core.Tensor(
            quats[:, i: i + 1], dtype=o3d.core.Dtype.Float32
        )

    # Create and save the point cloud
    pcd = o3d.t.geometry.PointCloud(ply_data)
    success = o3d.t.io.write_point_cloud(dir, pcd)
    assert success, "Ply file saving failed."


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


def depth_to_points(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to 3D points

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        points: 3D points in the world coordinate system [..., H, W, 3]
    """
    assert depths.shape[-1] == 1, f"Invalid depth shape: {depths.shape}"
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), f"Invalid viewmats shape: {camtoworlds.shape}"
    assert Ks.shape[-2:] == (3, 3), f"Invalid Ks shape: {Ks.shape}"
    assert (
        depths.shape[:-3] == camtoworlds.shape[:-2] == Ks.shape[:-2]
    ), f"Shape mismatch! depths: {depths.shape}, viewmats: {camtoworlds.shape}, Ks: {Ks.shape}"

    device = depths.device
    height, width = depths.shape[-3:-1]

    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )  # [H, W]

    fx = Ks[..., 0, 0]  # [...]
    fy = Ks[..., 1, 1]  # [...]
    cx = Ks[..., 0, 2]  # [...]
    cy = Ks[..., 1, 2]  # [...]

    # camera directions in camera coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx[..., None, None] + 0.5) / fx[..., None, None],
                (y - cy[..., None, None] + 0.5) / fy[..., None, None],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [..., H, W, 3]

    # ray directions in world coordinates
    directions = torch.einsum(
        "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
    )  # [..., H, W, 3]
    origins = camtoworlds[..., :3, -1]  # [..., 3]

    if not z_depth:
        directions = F.normalize(directions, dim=-1)

    points = origins[..., None, None, :] + depths * directions
    return points


def depth_to_normal(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to surface normals

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        normals: Surface normals in the world coordinate system [..., H, W, 3]
    """
    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)  # [..., H, W, 3]
    dx = torch.cat(
        [points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=-3
    )  # [..., H-2, W-2, 3]
    dy = torch.cat(
        [points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=-2
    )  # [..., H-2, W-2, 3]
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # [..., H-2, W-2, 3]
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
    return normals


def get_projection_matrix(znear, zfar, fovX, fovY, device="cuda"):
    """Create OpenGL-style projection matrix"""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# def depth_to_normal(
#     depths: Tensor, camtoworlds: Tensor, Ks: Tensor, near_plane: float, far_plane: float
# ) -> Tensor:
#     """
#     Convert depth to surface normal

#     Args:
#         depths: Z-depth of the Gaussians.
#         camtoworlds: camera to world transformation matrix.
#         Ks: camera intrinsics.
#         near_plane: Near plane distance.
#         far_plane: Far plane distance.

#     Returns:
#         Surface normals.
#     """
#     height, width = depths.shape[1:3]
#     viewmats = torch.linalg.inv(camtoworlds)  # [C, 4, 4]

#     normals = []
#     for cid, depth in enumerate(depths):
#         FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
#         FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
#         world_view_transform = viewmats[cid].transpose(0, 1)
#         projection_matrix = _get_projection_matrix(
#             znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=depths.device
#         ).transpose(0, 1)
#         full_proj_transform = (
#             world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
#         ).squeeze(0)
#         normal = _depth_to_normal(
#             depth,
#             world_view_transform,
#             full_proj_transform,
#             Ks[cid, 0, 0],
#             Ks[cid, 1, 1],
#         )
#         normals.append(normal)
#     normals = torch.stack(normals, dim=0)
#     return normals


# # ref: https://github.com/hbb1/2d-gaussian-splatting/blob/61c7b417393d5e0c58b742ad5e2e5f9e9f240cc6/utils/point_utils.py#L26
# def _depths_to_points(
#     depthmap, world_view_transform, full_proj_transform, fx, fy
# ) -> Tensor:
#     c2w = (world_view_transform.T).inverse()
#     H, W = depthmap.shape[:2]

#     intrins = (
#         torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
#         .float()
#         .cuda()
#     )

#     grid_x, grid_y = torch.meshgrid(
#         torch.arange(W, device="cuda").float(),
#         torch.arange(H, device="cuda").float(),
#         indexing="xy",
#     )
#     points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
#         -1, 3
#     )
#     rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
#     rays_o = c2w[:3, 3]
#     points = depthmap.reshape(-1, 1) * rays_d + rays_o
#     return points


# def _depth_to_normal(
#     depth, world_view_transform, full_proj_transform, fx, fy
# ) -> Tensor:
#     points = _depths_to_points(
#         depth,
#         world_view_transform,
#         full_proj_transform,
#         fx,
#         fy,
#     ).reshape(*depth.shape[:2], 3)
#     output = torch.zeros_like(points)
#     dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
#     dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
#     normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
#     output[1:-1, 1:-1, :] = normal_map
#     return output


# def _get_projection_matrix(znear, zfar, fovX, fovY, device="cuda") -> Tensor:
#     tanHalfFovY = math.tan((fovY / 2))
#     tanHalfFovX = math.tan((fovX / 2))

#     top = tanHalfFovY * znear
#     bottom = -top
#     right = tanHalfFovX * znear
#     left = -right

#     P = torch.zeros(4, 4, device=device)

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)
#     return P
