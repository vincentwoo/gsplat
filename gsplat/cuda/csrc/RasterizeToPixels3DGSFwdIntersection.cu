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
 * Minimal vector structs for convenience
 */
struct vec2 {
    float x, y;
};
struct vec3 {
    float x, y, z;
};

/**
 * Kernel: forward intersection for 3D Gaussians.
 * Now it also writes debug info (16 floats per pixel) into `debug_out`.
 *
 * We store in debug_out for each pixel (pix_id):
 *   debug_out[pix_id*16 + 0]  = best_gaussian_id
 *   debug_out[pix_id*16 + 1]  = best_disc
 *   debug_out[pix_id*16 + 2]  = best_t
 *   debug_out[pix_id*16 + 3..5] = best_point (x,y,z)
 *   debug_out[pix_id*16 + 6]  = best_weight
 *   debug_out[pix_id*16 + 7..9]  = ray_origin (x,y,z)
 *   debug_out[pix_id*16 + 10..12]= ray_dir (x,y,z)
 *   debug_out[pix_id*16 + 13..15]= placeholders or extra usage
 */
/**
 * A small helper: convert from NDC (-1..1) to pixel coords.
 */
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    // ((v + 1) * S - 1) * 0.5
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

/**
 * Kernel: forward intersection for 3D Gaussians.
 * Now it also writes debug info (16 floats per pixel) into `debug_out`.
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
    scalar_t *__restrict__ out_pts,       // [C, image_height, image_width, 3]

    // Debug buffer (16 floats per pixel) allocated via cudaMalloc
    scalar_t *__restrict__ debug_out
)
{
    // Cooperative groups for block sync.
    auto block = cg::this_thread_block();

    // Identify which camera and which tile block we are in.
    int32_t camera_id = block.group_index().x;
    int32_t tile_id   = block.group_index().y * tile_width + block.group_index().z;

    // Compute pixel row (i) and col (j) from tile coords + thread index.
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // Shift pointers to this camera's region.
    tile_offsets   += camera_id * tile_height * tile_width;
    render_alphas  += camera_id * image_height * image_width;
    out_pts        += camera_id * (image_height * image_width * 3);
    debug_out      += camera_id * (image_height * image_width * 16);

    // Convert (i, j) to an NDC coordinate, then back to pixel coords
    // so that we replicate the 'ndc2Pix' usage exactly.
    float ndc_x = (2.f*(float)j + 1.f) / (float)image_width  - 1.f;
    float ndc_y = (2.f*(float)i + 1.f) / (float)image_height - 1.f;

    float px = ndc2Pix(ndc_x, image_width);
    float py = ndc2Pix(ndc_y, image_height);

    int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done   = !inside;

    // Range of Gaussians for this tile.
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (
        (camera_id == (int32_t)(C - 1)) &&
        (tile_id   == (int32_t)(tile_width*tile_height - 1))
    ) ? n_isects : tile_offsets[tile_id + 1];

    // Threads in the block
    const uint32_t block_sz = block.size();
    uint32_t num_batches = (range_end - range_start + block_sz - 1) / block_sz;

    // Dynamic shared memory: indices + (x,y,opac) + (A,B,C)
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t*) s;  // [block_sz]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3*>(&id_batch[block_sz]);
    vec3 *conic_batch      = reinterpret_cast<vec3*>(&xy_opacity_batch[block_sz]);

    // Initialize partial alpha
    float T = 1.f;

    // Initialize partial color
    float pix_out[CDIM];
    #pragma unroll
    for (uint32_t cc = 0; cc < CDIM; cc++) {
        pix_out[cc] = 0.f;
    }

    // Camera intrinsics
    float fx = (float)Ks[camera_id*9 + 0];
    float fy = (float)Ks[camera_id*9 + 4];
    float cx = (float)Ks[camera_id*9 + 2];
    float cy = (float)Ks[camera_id*9 + 5];

    // Extrinsic: row-major
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

    // Rotate by R^T => direction in world
    float dx = (R00*x_cam + R10*y_cam + R20*1.f);
    float dy = (R01*x_cam + R11*y_cam + R21*1.f);
    float dz = (R02*x_cam + R12*y_cam + R22*1.f);

    // Normalize
    float len_dir = sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    float3 ray_dir = { dx/len_dir, dy/len_dir, dz/len_dir };

    // Track best intersection info
    float weight_max = 0.f;
    float3 best_point = {0.f, 0.f, 0.f};
    int best_id = -1;
    float best_disc = 0.f;
    float best_t = 0.f;

    // Iterate over tile gaussians in batches
    uint32_t tr = block.thread_rank();
    for (uint32_t b = 0; b < num_batches; b++) {
        // If entire block is done, break
        if (__syncthreads_count(done) >= block_sz) {
            break;
        }

        // Load this batch into shared
        uint32_t batch_start = range_start + block_sz*b;
        uint32_t idx = batch_start + tr;
        if (idx < (uint32_t)range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;

            vec2 xy   = means2d[g];
            float opac= (float)opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};

            conic_batch[tr] = conics[g]; // (A,B,C)
        }
        block.sync();

        uint32_t batch_size = min(block_sz, (uint32_t)(range_end - batch_start));
        for (uint32_t t = 0; (t < batch_size) && !done; t++) {
            const vec3 &co   = conic_batch[t];      // (A,B,C)
            const vec3 &xyo  = xy_opacity_batch[t]; // (x,y,opac)
            float opac       = xyo.z;

            // 2D elliptical footprint
            float dx_ = xyo.x - px;
            float dy_ = xyo.y - py;
            // sigma = 0.5*(A*x^2 + C*y^2) + B*x*y
            float sigma = 0.5f*(co.x*dx_*dx_ + co.z*dy_*dy_) + co.y*(dx_*dy_);

            // alpha = opac * exp(-sigma)
            float alpha = fminf(0.999f, opac*__expf(-sigma));
            if (sigma < 0.f || alpha < 1.f/255.f) {
                continue;
            }
            float vis = alpha * T;
            T *= (1.f - alpha);

            // Accumulate color (if inside image)
            if (inside) {
                int32_t g = id_batch[t];
                const float* c_ptr = colors + g*CDIM;
                #pragma unroll
                for (uint32_t kk = 0; kk < CDIM; kk++) {
                    pix_out[kk] += c_ptr[kk] * vis;
                }
            }

            // If almost fully opaque, set done
            if (T <= 1e-4f) {
                done = true;
            }

            // 3D intersection test
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

            // Build rotation matrix from quaternion
            float x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
            float xx2= qx*x2, yy2= qy*y2, zz2= qz*z2;
            float xy2= qx*y2, xz2= qx*z2, yz2= qy*z2;
            float rz2= qr*z2, ry2= qr*y2, rx2= qr*x2;

            float Rm00=1.f - (yy2 + zz2);
            float Rm01=xy2 - rz2;
            float Rm02=xz2 + ry2;
            float Rm10=xy2 + rz2;
            float Rm11=1.f - (xx2 + zz2);
            float Rm12=yz2 - rx2;
            float Rm20=xz2 - ry2;
            float Rm21=yz2 + rx2;
            float Rm22=1.f - (xx2 + yy2);

            float ox = ray_origin.x - mx;
            float oy = ray_origin.y - my;
            float oz = ray_origin.z - mz;

            float oX = Rm00*ox + Rm01*oy + Rm02*oz;
            float oY = Rm10*ox + Rm11*oy + Rm12*oz;
            float oZ = Rm20*ox + Rm21*oy + Rm22*oz;

            float dX = Rm00*ray_dir.x + Rm01*ray_dir.y + Rm02*ray_dir.z;
            float dY = Rm10*ray_dir.x + Rm11*ray_dir.y + Rm12*ray_dir.z;
            float dZ = Rm20*ray_dir.x + Rm21*ray_dir.y + Rm22*ray_dir.z;

            float invsx = 1.f/(sx*3.f);
            float invsy = 1.f/(sy*3.f);
            float invsz = 1.f/(sz*3.f);

            float ooX = oX*invsx, ooY = oY*invsy, ooZ = oZ*invsz;
            float ddX = dX*invsx, ddY = dY*invsy, ddZ = dZ*invsz;

            float A = ddX*ddX + ddY*ddY + ddZ*ddZ;
            float B = 2.f*(ddX*ooX + ddY*ooY + ddZ*ooZ);
            float C = (ooX*ooX + ooY*ooY + ooZ*ooZ) - 1.f;

            float disc = B*B - 4.f*A*C;
            if (disc > 0.f) {
                float sqrt_disc = sqrtf(disc);
                float t1 = (-B - sqrt_disc) / (2.f * A);
                float t2 = (-B + sqrt_disc) / (2.f * A);
                // pick the smaller positive root
                float t_ = -1.f;
                if (t1>0.f && t2>0.f) t_ = fminf(t1, t2);
                else if (t1>0.f)     t_ = t1;
                else if (t2>0.f)     t_ = t2;

                if (t_ > 0.f) {
                    float candidate_weight = vis;
                    if (candidate_weight > weight_max) {
                        weight_max = candidate_weight;
                        best_point.x = ray_origin.x + t_*ray_dir.x;
                        best_point.y = ray_origin.y + t_*ray_dir.y;
                        best_point.z = ray_origin.z + t_*ray_dir.z;
                        best_id      = g;
                        best_disc    = disc;
                        best_t       = t_;
                    }
                }
            }
        }
        block.sync();
    }

    // Write final outputs
    if (inside) {
        render_alphas[pix_id] = (scalar_t)(1.f - T);
        out_pts[pix_id*3 + 0] = (scalar_t)best_point.x;
        out_pts[pix_id*3 + 1] = (scalar_t)best_point.y;
        out_pts[pix_id*3 + 2] = (scalar_t)best_point.z;

        // Write debug: 16 floats per pixel
        float* dbg = debug_out + (pix_id*16);
        dbg[0]  = (float)best_id;
        dbg[1]  = best_disc;
        dbg[2]  = best_t;
        dbg[3]  = best_point.x;
        dbg[4]  = best_point.y;
        dbg[5]  = best_point.z;
        dbg[6]  = weight_max;
        dbg[7]  = ray_origin.x;
        dbg[8]  = ray_origin.y;
        dbg[9]  = ray_origin.z;
        dbg[10] = ray_dir.x;
        dbg[11] = ray_dir.y;
        dbg[12] = ray_dir.z;
        dbg[13] = -1.f;
        dbg[14] = -1.f;
        dbg[15] = -1.f;
    }
}

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
    // Are means2d etc. "packed" or not
    bool packed = (means2d.dim() == 2);

    // # of cameras + tiling
    uint32_t C = tile_offsets.size(0);
    uint32_t tile_h = tile_offsets.size(1);
    uint32_t tile_w = tile_offsets.size(2);

    // # intersections
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t N = packed ? 0 : means2d.size(1);

    // 1) Allocate debug buffer on device: shape [C, image_h, image_w, 16 floats]
    size_t total_pixels = (size_t)C*(size_t)image_height*(size_t)image_width;
    size_t debug_size   = total_pixels * 16 * sizeof(float);
    float* d_debug_out  = nullptr;
    cudaMalloc(&d_debug_out, debug_size);
    cudaMemset(d_debug_out, 0, debug_size); // init to zero

    // 2) Setup grid
    dim3 grid(C, tile_h, tile_w);
    dim3 threads(tile_size, tile_size, 1);

    int64_t block_sz = (int64_t)tile_size*(int64_t)tile_size;
    int64_t shmem_sz = block_sz*(sizeof(int32_t)+sizeof(vec3)+sizeof(vec3));

    cudaFuncSetAttribute(
      rasterize_to_pixels_3dgs_fwd_intersection_kernel<CDIM, float>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)shmem_sz
    );

    // 3) Launch kernel
    rasterize_to_pixels_3dgs_fwd_intersection_kernel<CDIM, float>
      <<<grid, threads, shmem_sz, at::cuda::getCurrentCUDAStream()>>>(
         C, N, n_isects, packed,
         // 2D data
         reinterpret_cast<const vec2*>(means2d.data_ptr<float>()),
         reinterpret_cast<const vec3*>(conics.data_ptr<float>()),
         colors.data_ptr<float>(),
         opacities.data_ptr<float>(),
         // image dims
         image_width,
         image_height,
         tile_size,
         tile_w,
         tile_h,
         tile_offsets.data_ptr<int32_t>(),
         flatten_ids.data_ptr<int32_t>(),
         // 3D
         means3D.data_ptr<float>(),
         scales.data_ptr<float>(),
         rotations.data_ptr<float>(),
         // Cameras
         viewmats.data_ptr<float>(),
         Ks.data_ptr<float>(),
         // Outputs
         render_alphas.data_ptr<float>(),
         out_pts.data_ptr<float>(),

         // debug
         d_debug_out
      );

    // 4) Sync + check
    cudaDeviceSynchronize();

    // 5) Copy debug data to host + print partial
    float* h_debug_out = (float*)malloc(debug_size);
    cudaMemcpy(h_debug_out, d_debug_out, debug_size, cudaMemcpyDeviceToHost);
    bool debug = false;
    if (debug) {

        printf("\n=== Debug Info (partial) ===\n");
        printf("C=%d, H=%d, W=%d, debug=16 floats per pixel\n",
               (int)C, (int)image_height, (int)image_width);

        // Print up to 2 cameras, 4×4 region
        int max_cam = (C<2)?(int)C:2;
        int max_i   = (image_height<4)?(int)image_height:4;
        int max_j   = (image_width<4)?(int)image_width:4;

        for (int c = 0; c < max_cam; c++) {
          for (int i = 0; i < max_i; i++) {
            for (int j = 0; j < max_j; j++) {
              size_t pix_id = (size_t)c*image_height*image_width + i*image_width + j;
              const float* dbg = &h_debug_out[pix_id*16];
              int   best_id   = (int)dbg[0];
              float best_disc = dbg[1];
              float best_t    = dbg[2];
              float bx        = dbg[3];
              float by        = dbg[4];
              float bz        = dbg[5];
              float w         = dbg[6];
              float ox        = dbg[7];
              float oy        = dbg[8];
              float oz        = dbg[9];
              float dx        = dbg[10];
              float dy        = dbg[11];
              float dz        = dbg[12];

              printf("[cam=%d, i=%d, j=%d]: bestID=%d, disc=%.3f, t=%.3f, w=%.3f\n"
                     "   best_point=(%.3f,%.3f,%.3f)\n"
                     "   ray_origin=(%.3f,%.3f,%.3f), ray_dir=(%.3f,%.3f,%.3f)\n",
                     c, i, j,
                     best_id, best_disc, best_t, w,
                     bx, by, bz,
                     ox, oy, oz,
                     dx, dy, dz);
            }
          }
        }
        printf("=== End Debug Info ===\n\n");
    }

    // 6) Free memory; We let it here for now: TODO: remove this memory stuff
    free(h_debug_out);
    cudaFree(d_debug_out);
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
