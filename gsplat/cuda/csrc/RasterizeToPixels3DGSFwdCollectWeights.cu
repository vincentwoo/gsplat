#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_collect_weights_kernel(
    // We have removed 'N' and 'packed' since they are not needed; added accum_* for collecting data.
    const uint32_t C,
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,         // [C, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    float *__restrict__ accum_weights,        // [C*N or nnz], but flatten_ids indexes into it
    int32_t *__restrict__ accum_weights_count,// same size
    float *__restrict__ accum_max_count       // same size
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // gather which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1 && tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

    // current visibility left to render
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // new variables to track maximum alpha*T for this pixel
    bool updated_max = false;
    float weight_max = 0.f;
    int idx_max = -1;

    // collect and process batches of gaussians
    uint32_t tr = block.thread_rank();
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetches 1 gaussian from front to back
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C*N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;

            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
            }

            // accumulate data for each contributing gaussian
            int32_t g = id_batch[t];
            const float vis = alpha * T;
            atomicAdd(&(accum_weights[g]), vis);
            atomicAdd(&(accum_weights_count[g]), 1);

            // track max
            if (vis > weight_max) {
                weight_max = vis;
                idx_max = g;
                updated_max = true;
            }
            cur_idx = batch_start + t;

            T = next_T;
            if (done) {
                break;
            }
        }
    }

    // if we found a maximum, increment accum_max_count
    if (updated_max && idx_max >= 0) {
        atomicAdd(&(accum_max_count[idx_max]), 1.f);
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_collect_weights_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,    // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,    // [C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [C, N]  or [nnz]

    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,

    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]

    // New tensors to collect per-gaussian data:
    at::Tensor accum_weights,        // same size as flatten_ids references
    at::Tensor accum_weights_count,  // same size
    at::Tensor accum_max_count       // same size
) {
    uint32_t C = tile_offsets.size(0);       // number of cameras
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

    // set shared memory limit
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_fwd_collect_weights_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_fwd_collect_weights_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            n_isects,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            accum_weights.data_ptr<float>(),
            accum_weights_count.data_ptr<int32_t>(),
            accum_max_count.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp file.
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_fwd_collect_weights_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        at::Tensor accum_weights,                                              \
        at::Tensor accum_weights_count,                                        \
        at::Tensor accum_max_count                                             \
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
