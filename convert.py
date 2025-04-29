import torch
import argparse
import open3d as o3d
from pathlib import Path
import numpy as np


def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    print(f"Saving ply to {dir}")

    # Convert all tensors to numpy arrays
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    # Extract data arrays
    means = numpy_data["means"]
    scales_log = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]

    has_w = "w" in numpy_data
    if has_w:
        w_log = numpy_data["w"]  # (N,)
        w = np.exp(w_log).reshape(-1, 1)  # W in linear space
        # Convert homogeneous means -> 3D means in world space
        means_3d = (means / w)
        scales_3d = np.log(np.exp(scales_log) / w)
    else:
        # If there's no 'w', we assume 'means' is already standard 3D
        means_3d = means
        scales_3d = np.exp(scales_log)

    # Filter all data arrays
    means = means_3d
    scales = scales_3d
    quats = quats
    opacities = opacities

    # Handle colors or spherical harmonics based on whether colors is provided
    if colors is not None:
        colors = colors.detach().cpu().numpy()
    else:
        sh0 = numpy_data["sh0"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()
        shN = numpy_data["shN"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()

    # Initialize ply_data with positions and normals
    ply_data = {
        "positions": o3d.core.Tensor(means, dtype=o3d.core.Dtype.Float32),
        "normals": o3d.core.Tensor(np.zeros_like(means), dtype=o3d.core.Dtype.Float32),
    }

    # Add features
    if colors is not None:
        # Use provided colors, converted to SH coefficients
        for j in range(colors.shape[1]):
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                (colors[:, j: j + 1] - 0.5) / 0.2820947917738781,
                dtype=o3d.core.Dtype.Float32,
            )
    else:
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
    assert success, "Could not save ply file."


def convert_ckpt_to_ply(ckpt_file: str, output_ply: str):
    """
    Load a single checkpoint file and convert it to a PLY file, filtering out invalid Gaussians.

    Args:
        ckpt_file: Path to a single checkpoint (.pt) file
        output_ply: The exact PLY file path to write
    """
    ckpt_file = Path(ckpt_file)
    if not ckpt_file.exists():
        raise ValueError(f"Checkpoint file does not exist: {ckpt_file}")

    print(f"Loading single checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location='cpu')
    splats = ckpt['splats']

    # Wrap into a ParameterDict
    splats_param_dict = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v) for k, v in splats.items()}
    )

    # Compute valid mask to filter out NaN or Inf values
    num_gaussians = splats_param_dict['means'].shape[0]
    valid_mask = torch.ones(num_gaussians, dtype=torch.bool, device=splats_param_dict['means'].device)
    for attr in splats_param_dict.values():
        valid_mask = valid_mask & torch.isfinite(attr).view(attr.shape[0], -1).all(dim=1)

    # Filter the splats to keep only valid Gaussians
    filtered_splats = torch.nn.ParameterDict({
        key: torch.nn.Parameter(attr[valid_mask]) for key, attr in splats_param_dict.items()
    })

    num_filtered = valid_mask.sum().item()
    print(f"Filtered out {num_gaussians - num_filtered} invalid Gaussians out of {num_gaussians}")

    # Save single file as PLY
    save_ply(filtered_splats, str(output_ply))
    print(f"Successfully saved PLY to {output_ply}")


def main():
    parser = argparse.ArgumentParser(description='Convert or merge checkpoint files into PLY')

    # We support either single-file mode or folder-merge mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str,
                       help='Path to a single checkpoint file (.pt)')
    # Output arguments
    parser.add_argument('--output', type=str, default="output.ply",
                        help='Output PLY file name (used only if --ckpt-file is specified)')
    args = parser.parse_args()

    convert_ckpt_to_ply(args.input, args.output)

if __name__ == '__main__':
    main()
