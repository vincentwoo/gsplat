import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing as mp
from typing_extensions import assert_never
import torch.nn.functional as F
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def compute_frequency_energy(img: torch.Tensor) -> float:
    """
    Computes the 'high-frequency energy' of a single image using DFT,
    as in Eq.(2) of your reference paper.
    Image is expected to be [C,H,W].
    """
    if img.dim() != 3:
        raise ValueError("Image must be [C,H,W].")

    # Sum across color channels for a single-luma approach
    if img.shape[0] > 1:
        x = img.float().mean(dim=0, keepdim=True)  # [1, H, W]
    else:
        x = img.float()  # [1, H, W] if single channel

    fft_img = torch.fft.fft2(x, norm='ortho')
    mag_sqr = fft_img.real ** 2 + fft_img.imag ** 2
    return mag_sqr.sum().item()


def process_single_image(
        img_path: str,
        device: str,
        candidate_factors: List[float]
) -> Tuple[float, Dict[float, float]]:
    """
    Load a single image, compute full-resolution frequency energy,
    and compute the frequency energy for each candidate downsample factor.

    Returns:
        (full_res_energy, {factor: factor_energy, ...})
    """
    # If something fails (bad image, etc.), return None so we can skip
    try:
        with Image.open(img_path) as pil_img:
            # Convert to RGB if needed
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Convert to tensor: (C, H, W)
            img_tensor = transforms.ToTensor()(pil_img).to(device)

            # 1) Full-resolution energy
            full_energy = compute_frequency_energy(img_tensor)

            # 2) Downsampled energies
            C, H, W = img_tensor.shape
            down_energies = {}
            for r in candidate_factors:
                h2, w2 = int(H * r), int(W * r)
                if h2 < 2 or w2 < 2:
                    # If the downsampling would be too small, skip
                    continue
                ds = F.interpolate(
                    img_tensor.unsqueeze(0), size=(h2, w2), mode="area"
                ).squeeze(0)
                down_energies[r] = compute_frequency_energy(ds)

        return full_energy, down_energies

    except Exception as ex:
        # You might want to log or print errors
        # print(f"Error processing {img_path}: {ex}")
        return None, {}


def compute_dataset_freq_metrics(
        image_paths: List[str],
        device: str = "cuda",
        batch_size: int = 16  # unused but retained for signature
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Computes average frequency metrics for full-res and downsampled images.

    Args:
        image_paths: List of paths to images.
        device: The device to use (e.g. 'cuda' or 'cpu').
        batch_size: (Unused) retained for compatibility.

    Returns:
        XF_full: Average energy of full-resolution images
        results: List of (downsample_factor, average_energy) pairs, sorted by factor
    """

    # Fail early if no paths
    if not image_paths:
        raise RuntimeError("No image paths provided.")

    # Candidate downscale factors
    candidate_factors = [1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0]

    # Accumulators for full res
    full_energy_sum = 0.0
    valid_count = 0

    # Factor-wise sums/counters
    factor_sums = {r: 0.0 for r in candidate_factors}
    factor_counts = {r: 0 for r in candidate_factors}

    # We'll process images in parallel
    # Decide if you want threads or processes:
    #   - ThreadPoolExecutor: good if most of your time is I/O (e.g. loading images).
    #   - ProcessPoolExecutor: can bypass GIL, might help for CPU-bound tasks.
    # For GPU usage, often single-thread or small concurrency can still be enough.

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        # Submit a job for each image
        futures = []
        for path in image_paths:
            futures.append(executor.submit(
                process_single_image, path, device, candidate_factors
            ))

        # Collect results as they come in
        for future in tqdm(as_completed(futures), total=len(futures), desc="Frequency Analysis"):
            result = future.result()
            if result is None:
                continue

            full_e, down_energies = result
            if full_e is None:
                # If something failed for this image, skip
                continue

            full_energy_sum += full_e
            valid_count += 1

            # Update factor sums
            for r, down_e in down_energies.items():
                factor_sums[r] += down_e
                factor_counts[r] += 1

    if valid_count == 0:
        raise RuntimeError("Could not compute frequency energy for any image.")

    # Average full-resolution
    XF_full = full_energy_sum / valid_count

    # Average per factor
    results = []
    for r in candidate_factors:
        if factor_counts[r] > 0:
            avg_e = factor_sums[r] / factor_counts[r]
            results.append((r, avg_e))

    # Sort by factor
    results.sort(key=lambda x: x[0])
    return XF_full, results


def allocate_iterations_by_frequency(S, XF_full, down_list):
    """
    Allocates iteration counts per stage based on frequency energy ratios (Eq. 6/7).
    """
    used = 0
    schedule = []
    # Ensure XF_full is not zero to avoid division by zero
    if XF_full <= 1e-9:
        print("Warning: Full frequency energy is near zero. Falling back to equal allocation.")
        # Fallback: allocate equally minus 1 step for full res
        num_stages = len(down_list) + 1
        steps_per_stage = S // num_stages
        for factor, _ in down_list:
            schedule.append((factor, steps_per_stage))
            used += steps_per_stage
        leftover = S - used
        schedule.append((1.0, leftover))
        return schedule

    for (factor, XFr) in down_list:
        frac = max(0.0, XFr / XF_full)  # Clamp fraction just in case
        steps = int(S * frac)
        if steps > 0:
            schedule.append((factor, steps))
            used += steps

    leftover = S - used
    if leftover > 0:
        schedule.append((1.0, leftover))
    elif not any(f == 1.0 for f, s in schedule):  # Ensure full res stage exists
        # Steal one step from the last stage if possible
        if schedule:
            last_factor, last_steps = schedule[-1]
            if last_steps > 1:
                schedule[-1] = (last_factor, last_steps - 1)
                schedule.append((1.0, 1))
            else:  # Cannot steal, just add a 1-step full res phase
                schedule.append((1.0, 1))
        else:  # No downsample stages, all full res
            schedule.append((1.0, S))

    # Normalize steps if they don't sum up exactly to S due to int() rounding
    current_total_steps = sum(s for f, s in schedule)
    if current_total_steps != S and current_total_steps > 0:
        # print(f"Adjusting schedule steps from {current_total_steps} to {S}")
        diff = S - current_total_steps
        # Add/remove difference to/from the longest stage (usually full-res)
        longest_stage_idx = -1
        max_steps = -1
        for idx, (f, s) in enumerate(schedule):
            if s > max_steps:
                max_steps = s
                longest_stage_idx = idx

        if longest_stage_idx != -1:
            adj_factor, adj_steps = schedule[longest_stage_idx]
            new_steps = max(1, adj_steps + diff)  # Ensure stage has at least 1 step
            schedule[longest_stage_idx] = (adj_factor, new_steps)
            # Recalculate total and handle potential overshoot/undershoot again if needed (rare)
            final_total = sum(s for f, s in schedule)
            if final_total != S:
                # As a final fallback, just dump remainder into the last stage
                final_diff = S - final_total
                last_f, last_s = schedule[-1]
                schedule[-1] = (last_f, max(1, last_s + final_diff))
                # print(f"Final schedule adjustment: total steps {sum(s for f, s in schedule)}")

    return schedule

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        total_iterations: int = 30_000,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        #camera_locations = camtoworlds[:, :3, 3]
        #scene_center = np.median(camera_locations, axis=0)
        #dists = np.linalg.norm(self.points - scene_center, axis=1)
        #scene_scale = np.median(dists)
        #mad = np.median(np.abs(dists - scene_scale))
        #k = 3
        #threshold = scene_scale + k * mad
        #mask = dists < threshold
        #self.points = self.points[mask]
        #self.points_rgb = self.points_rgb[mask]

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(self.points - scene_center, axis=1)
        self.scene_scale = np.median(dists)

        XF_full, down_list = compute_dataset_freq_metrics(self.image_paths)
        self.schedule = allocate_iterations_by_frequency(total_iterations, XF_full, down_list)
        print(f"Generated Resolution Schedule (factor, steps): {self.schedule}")
        if sum(s for f, s in self.schedule) != total_iterations:
            print(f"Warning: Schedule steps sum to {sum(s for f, s in self.schedule)}, expected {total_iterations}. Check allocation logic.")

class Dataset:
    """A simple dataset class that downscales images according to a schedule."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        # A multiprocessing.Value to track the global iteration or "step"
        self._step = mp.Value('i', 0)

        # We now incorporate the resolution schedule from the parser
        # schedule is a list of (down_factor, steps)
        self.schedule = parser.schedule

        # Convert schedule into a "cumulative" list for quick factor lookup
        self._cumulative_schedule = []
        running = 0
        for (factor, steps) in self.schedule:
            running += steps
            self._cumulative_schedule.append((factor, running))

        # We will pick training vs test images by modding with test_every
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def get_down_factor_for_step(self, step: int) -> float:
        """
        Return the resolution down_factor for the given global 'step'
        by walking through the cumulative schedule.
        """
        for (factor, accum_step) in self._cumulative_schedule:
            if step <= accum_step:
                return factor
        # If for some reason we exceed the last stage, default to 1.0
        return 1.0

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworld = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        # If there's camera distortion, undistort
        if len(params) > 0 and camera_id in self.parser.mapx_dict:
            mapx = self.parser.mapx_dict[camera_id]
            mapy = self.parser.mapy_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        # Now apply the "frequency-based scheduling" factor
        step_now = self._step.value
        down_factor = self.get_down_factor_for_step(step_now)

        if down_factor < 1.0:
            # e.g. if factor=0.25 => new size = 25% of original
            image = self.downscale_image_by_factor(image, down_factor)
            # Adjust the intrinsics
            K[0:2, :] *= down_factor
            if mask is not None:
                mask = self.downscale_image_by_factor(mask.astype(np.float32), down_factor) > 0.5

        # Optionally do patches
        if self.patch_size is not None:
            h, w = image.shape[:2]
            x0 = np.random.randint(0, max(w - self.patch_size, 1))
            y0 = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            K[0, 2] -= x0
            K[1, 2] -= y0
            if mask is not None:
                mask = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

        # Prepare outputs
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index in the dataset
            "down_factor": down_factor,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # If we want to load (sparse) depth from 3D points:
        if self.load_depths:
            # Project 3D points into the camera
            worldtocam = np.linalg.inv(camtoworld)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, [])
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocam[:3, :3] @ points_world.T + worldtocam[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            pts_xy = points_proj[:, :2] / points_proj[:, 2:3]
            pts_depth = points_cam[:, 2]

            # Filter out invalid
            h, w = image.shape[:2]
            valid = (
                (pts_xy[:, 0] >= 0)
                & (pts_xy[:, 0] < w)
                & (pts_xy[:, 1] >= 0)
                & (pts_xy[:, 1] < h)
                & (pts_depth > 0)
            )
            pts_xy = pts_xy[valid]
            pts_depth = pts_depth[valid]
            data["points"] = torch.from_numpy(pts_xy).float()
            data["depths"] = torch.from_numpy(pts_depth).float()

        return data

    def downscale_image_by_factor(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Downscale the input image by a float factor < 1.0 using AREA interpolation.
        """
        h, w = image.shape[:2]
        new_h = int(round(h * factor))
        new_w = int(round(w * factor))
        if new_h < 2 or new_w < 2:
            # For safety, clamp
            new_h = max(new_h, 2)
            new_w = max(new_w, 2)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def update_step(self, step: int):
        """Update the global step from outside, e.g. in your training loop."""
        with self._step.get_lock():
            self._step.value = step


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
