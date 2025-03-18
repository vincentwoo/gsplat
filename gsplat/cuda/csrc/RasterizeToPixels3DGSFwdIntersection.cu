/*****************************************************************************
 * Rasterize3DGS.cu
 * ---------------------------------------------------------------------------
 * Implements a forward intersection kernel for 3D Gaussians (or ellipsoids)
 * that have already been projected into 2D. This version uses perspective
 * camera models given by extrinsic (4×4) and intrinsic (3×3) matrices.
 *
 * to run the forward‐intersection rasterization and accumulate color & alpha.
 *****************************************************************************/

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace gsplat {

namespace cg = cooperative_groups;

/**
 * Minimal vector structs for convenience (feel free to replace these with
 * float2 / float3 from CUDA headers or your own library).
 */
struct vec2 {
    float x, y;
};
struct vec3 {
    float x, y, z;
};

/**
 * Main forward‐intersection kernel. Renders splats (2D Gaussian footprints)
 * while also computing the 3D intersection of a ray from the camera through
 * each pixel with each ellipsoid (via scales + quaternion).
 *
 * For each pixel:
 * 1. We accumulate color and alpha using the 2D Gaussian conic.
 * 2. We compute the best ellipsoid intersection along that pixel's view ray.
 *
 * Inputs:
 *  - means2d:      [C,N,2] or [nnz,2]      — 2D positions of the Gaussian
 *  - conics:       [C,N,3] or [nnz,3]      — 2D conic “inverse cov” => [A,B,C]
 *  - colors:       [C,N,CDIM] or [nnz,CDIM]— color(s) per Gaussian
 *  - opacities:    [C,N] or [nnz]          — opacity or alpha scaling
 *
 *  - image_width, image_height: image size
 *  - tile_size, tile_offsets, flatten_ids: tiling data
 *
 *  - means3D:   [nnz,3] or [C,N,3]         — 3D centers of ellipsoids
 *  - scales:    [nnz,3]                    — ellipsoid scales
 *  - rotations: [nnz,4]                    — ellipsoid rotation as quaternion
 *
 *  - viewmats:  [C,4,4] (row-major)        — extrinsic transforms
 *  - Ks:        [C,3,3] (row-major)        — intrinsics
 *
 * Outputs:
 *  - render_alphas: [C, image_h, image_w]  — accumulated alpha
 *  - out_pts:       [C, image_h, image_w,3]— 3D intersection points
 */
template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_intersection_kernel(
    // Number of cameras (C) and number of gaussians (N) if not packed:
    const uint32_t C,
    const uint32_t N,
    // Number of tile->gaussian intersections:
    const uint32_t n_isects,
    // Whether data is "packed" (nnz) or not:
    const bool packed,

    // 2D Gaussian data
    const vec2 *__restrict__ means2d,       // [C,N,2] or [nnz,2]
    const vec3 *__restrict__ conics,        // [C,N,3] or [nnz,3]
    const scalar_t *__restrict__ colors,    // [C,N,CDIM] or [nnz,CDIM]
    const scalar_t *__restrict__ opacities, // [C,N] or [nnz]

    // Image / Tiling info
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,

    // For each tile in the image, we store offsets to flatten_ids[]:
    const int32_t *__restrict__ tile_offsets, // [C, tile_h, tile_w]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    // 3D intersection data
    const scalar_t *__restrict__ means3D,   // [nnz,3] or [C,N,3]
    const scalar_t *__restrict__ scales,    // [nnz,3]
    const scalar_t *__restrict__ rotations, // [nnz,4]

    // Camera extrinsics / intrinsics
    const scalar_t *__restrict__ viewmats,  // [C,16], row-major
    const scalar_t *__restrict__ Ks,        // [C,9],  row-major

    // Outputs
    scalar_t *__restrict__ render_alphas, // [C, image_height, image_width]
    scalar_t *__restrict__ out_pts        // [C, image_height, image_width, 3]
)
{
    // Cooperative groups for block sync.
    auto block = cg::this_thread_block();

    // Identify which camera and which tile block we are in.
    int32_t camera_id = block.group_index().x;
    int32_t tile_id   = block.group_index().y * tile_width + block.group_index().z;

    // Compute pixel row (i) and col (j) from tile coordinates + thread index.
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // Shift pointers to this camera's region.
    tile_offsets   += camera_id * tile_height * tile_width;
    render_alphas  += camera_id * image_height * image_width;
    out_pts        += camera_id * (image_height * image_width * 3);

    // Pixel coords (center = +0.5 offset).
    float px = static_cast<float>(j) + 0.5f;
    float py = static_cast<float>(i) + 0.5f;
    int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done   = !inside;

    // Range of Gaussians for this tile.
    int32_t range_start = tile_offsets[tile_id];
    // If this is the last tile, range ends at n_isects; else read next tile offset.
    int32_t range_end = (
        (camera_id == (int32_t)(C - 1)) &&
        (tile_id   == (int32_t)(tile_width*tile_height - 1))
    ) ? n_isects : tile_offsets[tile_id + 1];

    // Number of threads in this block.
    const uint32_t block_size = block.size();
    // Number of batches we need to load from flatten_ids for this tile.
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Dynamically allocated shared memory layout:
    extern __shared__ int s[];
    // 1) Indices of Gaussians in flatten_ids.
    int32_t *id_batch = (int32_t *) s;  // [block_size]

    // 2) Next chunk: store (x, y, opacity) for each Gaussian
    vec3 *xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[block_size]);

    // 3) Next chunk: conic parameters (A, B, C)
    vec3 *conic_batch = reinterpret_cast<vec3*>(&xy_opacity_batch[block_size]);

    // Initialize partial color & alpha for this pixel.
    float T = 1.f; // “transmittance” so far
    float pix_out[CDIM];
    #pragma unroll
    for (uint32_t cc = 0; cc < CDIM; cc++) {
        pix_out[cc] = 0.f;
    }

    // Build the camera (world‐space) ray for this pixel using extrinsics + intrinsics.
    float fx = (float)Ks[camera_id*9 + 0];
    float fy = (float)Ks[camera_id*9 + 4];
    float cx = (float)Ks[camera_id*9 + 2];
    float cy = (float)Ks[camera_id*9 + 5];

    // row-major [4×4] extrinsic
    const scalar_t *vm = viewmats + camera_id*16;
    float R00 = vm[0],  R01 = vm[1],  R02 = vm[2],  t0 = vm[3];
    float R10 = vm[4],  R11 = vm[5],  R12 = vm[6],  t1 = vm[7];
    float R20 = vm[8],  R21 = vm[9],  R22 = vm[10], t2 = vm[11];

    // Camera center in world space = -R^T * t
    float rx = -(R00*t0 + R10*t1 + R20*t2);
    float ry = -(R01*t0 + R11*t1 + R21*t2);
    float rz = -(R02*t0 + R12*t1 + R22*t2);

    float3 ray_origin = { rx, ry, rz };

    // Direction in camera coords => (x_cam, y_cam, 1.f).
    float x_cam = (px - cx) / (fx + 1e-12f);
    float y_cam = (py - cy) / (fy + 1e-12f);

    // Rotate by R^T => direction in world.
    float dx = (R00*x_cam + R10*y_cam + R20*1.f);
    float dy = (R01*x_cam + R11*y_cam + R21*1.f);
    float dz = (R02*x_cam + R12*y_cam + R22*1.f);

    // Normalize.
    float len_dir = sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    float3 ray_dir = { dx/len_dir, dy/len_dir, dz/len_dir };

    // Track the "best" intersection.
    float weight_max = 0.f;
    float3 best_point = {0.f, 0.f, 0.f};

    // Process all gaussians in tile, in batches:
    uint32_t tr = block.thread_rank();
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            // If all threads are done, skip further work.
            break;
        }

        // Load the next batch into shared memory.
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < (uint32_t)range_end) {
            int32_t g = flatten_ids[idx];
            // Save the ID:
            id_batch[tr] = g;

            // Pack 2D coords + opacity:
            vec2 xy     = means2d[g];
            float opac  = (float)opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};

            // Conic (A,B,C):
            conic_batch[tr] = conics[g];
        }
        block.sync();

        // Now every thread processes the loaded batch.
        uint32_t batch_size = min(block_size, (uint32_t)(range_end - batch_start));
        for (uint32_t t = 0; (t < batch_size) && !done; t++) {
            // Evaluate 2D footprint for this pixel.
            const vec3 &c   = conic_batch[t];        // (A,B,C)
            const vec3 &xyo = xy_opacity_batch[t];   // (x,y,opacity)
            float opac      = xyo.z;

            float dx_ = xyo.x - px;
            float dy_ = xyo.y - py;
            float sigma = 0.5f*(c.x * dx_ * dx_ + c.z * dy_ * dy_) + c.y * dx_ * dy_;

            // Evaluate alpha (simple Gaussian).
            float alpha = fminf(0.999f, opac * __expf(-sigma));
            // Threshold to skip negligible contributions.
            if (sigma < 0.f || alpha < 1.f/255.f) {
                continue;
            }

            // Composite in "over" style:
            float vis = alpha * T;
            float next_T = T * (1.f - alpha);
            T = next_T;
            // Accumulate color if inside.
            if (inside) {
                int32_t g = id_batch[t];
                const float *c_ptr = colors + g*CDIM;
                #pragma unroll
                for (uint32_t kk = 0; kk < CDIM; kk++) {
                    pix_out[kk] += c_ptr[kk] * vis;
                }
            }
            // Stop if nearly opaque:
            if (T <= 1e-4f) {
                done = true;
            }

            // ----- 3D Intersection: check ellipsoid along ray -----
            {
                int32_t g = id_batch[t];
                float mx = (float)means3D[g*3 + 0];
                float my = (float)means3D[g*3 + 1];
                float mz = (float)means3D[g*3 + 2];

                float sx = (float)scales[g*3 + 0];
                float sy = (float)scales[g*3 + 1];
                float sz = (float)scales[g*3 + 2];

                float qr = (float)rotations[g*4 + 0];
                float qx = (float)rotations[g*4 + 1];
                float qy = (float)rotations[g*4 + 2];
                float qz = (float)rotations[g*4 + 3];

                // Convert quaternion -> rotation matrix (row-major, 3×3).
                float x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
                float xx2 = qx*x2, yy2 = qy*y2, zz2 = qz*z2;
                float xy2 = qx*y2, xz2 = qx*z2, yz2 = qy*z2;
                float rz2 = qr*z2, ry2 = qr*y2, rx2 = qr*x2;

                float Rm00 = 1.f - (yy2 + zz2);
                float Rm01 = xy2 - rz2;
                float Rm02 = xz2 + ry2;
                float Rm10 = xy2 + rz2;
                float Rm11 = 1.f - (xx2 + zz2);
                float Rm12 = yz2 - rx2;
                float Rm20 = xz2 - ry2;
                float Rm21 = yz2 + rx2;
                float Rm22 = 1.f - (xx2 + yy2);

                // Shift the ray origin into ellipsoid's local space.
                float ox = ray_origin.x - mx;
                float oy = ray_origin.y - my;
                float oz = ray_origin.z - mz;

                float oX = Rm00*ox + Rm01*oy + Rm02*oz;
                float oY = Rm10*ox + Rm11*oy + Rm12*oz;
                float oZ = Rm20*ox + Rm21*oy + Rm22*oz;

                float dX = Rm00*ray_dir.x + Rm01*ray_dir.y + Rm02*ray_dir.z;
                float dY = Rm10*ray_dir.x + Rm11*ray_dir.y + Rm12*ray_dir.z;
                float dZ = Rm20*ray_dir.x + Rm21*ray_dir.y + Rm22*ray_dir.z;

                // Scale the ellipsoid radii by 3, for example:
                float invsx = 1.f / (sx * 3.f);
                float invsy = 1.f / (sy * 3.f);
                float invsz = 1.f / (sz * 3.f);

                float ooX = oX * invsx, ooY = oY * invsy, ooZ = oZ * invsz;
                float ddX = dX * invsx, ddY = dY * invsy, ddZ = dZ * invsz;

                // Solve quadratic for intersection.
                float A = ddX*ddX + ddY*ddY + ddZ*ddZ;
                float B = 2.f*(ddX*ooX + ddY*ooY + ddZ*ooZ);
                float C = (ooX*ooX + ooY*ooY + ooZ*ooZ) - 1.f;

                float disc = B*B - 4.f*A*C;
                if (disc > 0.f) {
                    float t_ = -B / (2.f*A);
                    // Only consider t_ > 0 => in front of camera.
                    if (t_ > 0.f) {
                        float candidate_weight = vis; // Weighted by how much it contributes
                        if (candidate_weight > weight_max) {
                            weight_max = candidate_weight;
                            // Intersection in world space
                            best_point.x = ray_origin.x + t_*ray_dir.x;
                            best_point.y = ray_origin.y + t_*ray_dir.y;
                            best_point.z = ray_origin.z + t_*ray_dir.z;
                        }
                    }
                }
            }
        } // end of batch loop
        block.sync();
    } // end for b in num_batches

    // Store final results for this pixel if inside bounds.
    if (inside) {
        // Final alpha
        render_alphas[pix_id] = static_cast<scalar_t>(1.f - T);
        // Best intersection
        out_pts[pix_id*3 + 0] = static_cast<scalar_t>(best_point.x);
        out_pts[pix_id*3 + 1] = static_cast<scalar_t>(best_point.y);
        out_pts[pix_id*3 + 2] = static_cast<scalar_t>(best_point.z);

        // Write out the color as needed (you might have an image buffer if required).
        // This sample code only returns alpha & 3D points.
        // If you need color output, either create a separate buffer or
        // adapt the code accordingly.
    }
}


/**
 * Host launcher: sets up the CUDA grid & dynamic shared mem, then calls
 * the rasterize_to_pixels_3dgs_fwd_intersection_kernel.
 */
template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_intersection_kernel(
    // 2D splat inputs
    const at::Tensor means2d,      // [C,N,2] or [nnz,2]
    const at::Tensor conics,       // [C,N,3] or [nnz,3]
    const at::Tensor colors,       // [C,N,CDIM] or [nnz,CDIM]
    const at::Tensor opacities,    // [C,N] or [nnz]

    // Image / Tiling
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [C,tile_h,tile_w]
    const at::Tensor flatten_ids,  // [n_isects]

    // 3D intersection data
    const at::Tensor means3D,      // [nnz,3] or [C,N,3]
    const at::Tensor scales,       // [nnz,3]
    const at::Tensor rotations,    // [nnz,4]

    // Camera extrinsics + intrinsics
    const at::Tensor viewmats,     // [C,4,4]
    const at::Tensor Ks,           // [C,3,3]

    // Outputs
    at::Tensor render_alphas, // [C, image_h, image_w]
    at::Tensor out_pts        // [C, image_h, image_w, 3]
)
{
    // Determine if data is "packed" or not by checking dims of means2d.
    bool packed = (means2d.dim() == 2);

    // # of cameras:
    uint32_t C = tile_offsets.size(0);
    // Tiling geometry:
    uint32_t tile_h = tile_offsets.size(1);
    uint32_t tile_w = tile_offsets.size(2);

    // # of intersections:
    uint32_t n_isects = flatten_ids.size(0);

    // If not packed, read N from means2d.size(1); else 0.
    uint32_t N = packed ? 0 : means2d.size(1);

    // Grid: (camera, tile_h, tile_w)
    dim3 grid(C, tile_h, tile_w);
    // Each block is tile_size×tile_size
    dim3 threads(tile_size, tile_size, 1);

    // We'll need dynamic shared mem: block_size * ( sizeof(int32_t) + 2×sizeof(vec3) ).
    // block_size = tile_size * tile_size
    int64_t block_size = static_cast<int64_t>(tile_size) * static_cast<int64_t>(tile_size);
    int64_t shmem_size = block_size * (
        sizeof(int32_t) +    // id_batch
        sizeof(vec3)     +   // xy_opacity_batch
        sizeof(vec3)         // conic_batch
    );

    // Set optional max dynamic shared memory limit (if your GPU supports).
    cudaFuncSetAttribute(
        rasterize_to_pixels_3dgs_fwd_intersection_kernel<CDIM, float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_size
    );

    // Launch
    rasterize_to_pixels_3dgs_fwd_intersection_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            // 2D data
            reinterpret_cast<const vec2*>(means2d.data_ptr<float>()),
            reinterpret_cast<const vec3*>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            // Image dims
            image_width,
            image_height,
            tile_size,
            tile_w,
            tile_h,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            // 3D data
            means3D.data_ptr<float>(),
            scales.data_ptr<float>(),
            rotations.data_ptr<float>(),
            // Cameras
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            // Outputs
            render_alphas.data_ptr<float>(),
            out_pts.data_ptr<float>()
        );
}

// ---------------------------------------------------------------------
// Explicit instantiations for some CDIM values.
// Adjust or expand as you need.
// ---------------------------------------------------------------------
#define __INS__(CDIM)                                                          \
  template void launch_rasterize_to_pixels_3dgs_fwd_intersection_kernel<CDIM>( \
      const at::Tensor means2d,  \
      const at::Tensor conics,  \
      const at::Tensor colors,  \
      const at::Tensor opacities,  \
      uint32_t image_width, \
      uint32_t image_height, \
      uint32_t tile_size, \
      const at::Tensor tile_offsets, \
      const at::Tensor flatten_ids, \
      const at::Tensor means3D,  \
      const at::Tensor scales,      \
      const at::Tensor rotations,   \
      const at::Tensor viewmats,  \
      const at::Tensor Ks, \
      at::Tensor render_alphas, \
      at::Tensor out_pts\
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat
