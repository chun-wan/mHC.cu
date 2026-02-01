#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>
#include "../include/mhc_types_hip.h"
#include "../include/utils_hip.h"

namespace cg = cooperative_groups;

namespace mhc {

constexpr int STREAM_MIX_TC_THRESHOLD = 32;

// ============================================================================
// Stream Operations - Forward Kernels
// ============================================================================

template<int BLOCK_SIZE>
__global__ void stream_add_kernel(float* __restrict__ out, const float* __restrict__ a,
                                  const float* __restrict__ b, int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_fused_sigmoid_kernel(floatX* __restrict__ out,
                                                           float* __restrict__ H_pre_activated,
                                                           const float* __restrict__ inp,
                                                           const float* __restrict__ H_pre_raw,
                                                           int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        float activated = fast_sigmoid(H_pre_raw[threadIdx.x]);
        s_H_pre[threadIdx.x] = activated;
        H_pre_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C)
        return;

    int b = idx / C, c = idx % C;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += s_H_pre[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = static_cast<floatX>(sum);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_fused_sigmoid_vec4_kernel(floatX* __restrict__ out,
                                                                float* __restrict__ H_pre_activated,
                                                                const float* __restrict__ inp,
                                                                const float* __restrict__ H_pre_raw,
                                                                int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n) {
        float activated = fast_sigmoid(H_pre_raw[threadIdx.x]);
        s_H_pre[threadIdx.x] = activated;
        H_pre_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx4 = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int C4 = C / 4;
    if (idx4 >= B * C4)
        return;

    int b = idx4 / C4;
    int c4 = idx4 % C4;
    int c = c4 * 4;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float h = s_H_pre[i];
            const float4* inp4 = reinterpret_cast<const float4*>(&inp[b * n * C + i * C + c]);
            float4 v = *inp4;
            sum.x += h * v.x;
            sum.y += h * v.y;
            sum.z += h * v.z;
            sum.w += h * v.w;
        }
    }
    
    // Store as bf16
    hip_bfloat16* out_bf16 = reinterpret_cast<hip_bfloat16*>(&out[b * C + c]);
    out_bf16[0] = static_cast<hip_bfloat16>(sum.x);
    out_bf16[1] = static_cast<hip_bfloat16>(sum.y);
    out_bf16[2] = static_cast<hip_bfloat16>(sum.z);
    out_bf16[3] = static_cast<hip_bfloat16>(sum.w);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_fused_kernel(
    float* __restrict__ out, float* __restrict__ H_post_activated, const float* __restrict__ x_inp,
    const floatX* __restrict__ y_norm, const float* __restrict__ H_post_raw,
    const float* __restrict__ M, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H_post[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n) {
        float activated = 2.0f * fast_sigmoid(H_post_raw[threadIdx.x]);
        s_H_post[threadIdx.x] = activated;
        H_post_activated[threadIdx.x] = activated;
    }
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    float mix_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n)
            mix_sum += s_M[i * n + j] * x_inp[b * n * C + j * C + c];
    }
    out[idx] = mix_sum + s_H_post[i] * static_cast<float>(y_norm[b * C + c]);
}

// ============================================================================
// Dynamic H Path Kernels
// ============================================================================

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_bf16_dynamic_kernel(floatX* __restrict__ out,
                                                     const float* __restrict__ inp,
                                                     const float* __restrict__ H_pre, int B, int n, int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * C)
        return;

    int b = idx / C, c = idx % C;
    const float* h = H_pre + b * n;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            sum += h[i] * inp[b * n * C + i * C + c];
    }
    out[idx] = static_cast<floatX>(sum);
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_add_dynamic_kernel(
    float* __restrict__ out, const float* __restrict__ x_inp, const floatX* __restrict__ y_norm,
    const float* __restrict__ H_post, const float* __restrict__ M, int B, int n, int C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;

    const float* h = H_post + b * n;
    const float* m = M + b * n * n;

    float mix_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < MAX_N; j++) {
        if (j < n)
            mix_sum += m[i * n + j] * x_inp[b * n * C + j * C + c];
    }
    out[idx] = mix_sum + h[i] * static_cast<float>(y_norm[b * C + c]);
}

// ============================================================================
// Launch Functions
// ============================================================================

inline void stream_aggregate_bf16_fused_sigmoid(floatX* out, float* H_pre_activated,
                                                const float* inp, const float* H_pre_raw, int B,
                                                int n, int C, hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    bool use_vec4 = (C % 4 == 0) && (C >= 64);

    if (use_vec4) {
        int blocks = (B * (C / 4) + BLOCK - 1) / BLOCK;
        
        #define DISPATCH_AGGREGATE_VEC4(MAX_N_VAL) \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_aggregate_bf16_fused_sigmoid_vec4_kernel<BLOCK, MAX_N_VAL>), \
                               dim3(blocks), dim3(BLOCK), 0, stream, out, H_pre_activated, inp, H_pre_raw, B, n, C)
        
        if (n <= 4) { DISPATCH_AGGREGATE_VEC4(4); }
        else if (n <= 8) { DISPATCH_AGGREGATE_VEC4(8); }
        else if (n <= 16) { DISPATCH_AGGREGATE_VEC4(16); }
        else if (n <= 32) { DISPATCH_AGGREGATE_VEC4(32); }
        #undef DISPATCH_AGGREGATE_VEC4
    } else {
        int blocks = (B * C + BLOCK - 1) / BLOCK;
        
        #define DISPATCH_AGGREGATE_FUSED(MAX_N_VAL) \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_aggregate_bf16_fused_sigmoid_kernel<BLOCK, MAX_N_VAL>), \
                               dim3(blocks), dim3(BLOCK), 0, stream, out, H_pre_activated, inp, H_pre_raw, B, n, C)
        
        if (n <= 4) { DISPATCH_AGGREGATE_FUSED(4); }
        else if (n <= 8) { DISPATCH_AGGREGATE_FUSED(8); }
        else if (n <= 16) { DISPATCH_AGGREGATE_FUSED(16); }
        else if (n <= 32) { DISPATCH_AGGREGATE_FUSED(32); }
        else { fprintf(stderr, "stream_aggregate_bf16_fused_sigmoid: n > 32 not implemented\n"); }
        #undef DISPATCH_AGGREGATE_FUSED
    }
}

inline void stream_distribute_mix_add_fused(float* out, float* H_post_activated, const float* x_inp,
                                            const floatX* y_norm, const float* H_post_raw,
                                            const float* M, int B, int n, int C,
                                            hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;

    #define DISPATCH_MIX_ADD_FUSED(MAX_N_VAL) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_distribute_mix_add_fused_kernel<BLOCK, MAX_N_VAL>), \
                           dim3(blocks), dim3(BLOCK), 0, stream, out, H_post_activated, x_inp, y_norm, H_post_raw, M, B, n, C)

    if (n <= 4) { DISPATCH_MIX_ADD_FUSED(4); }
    else if (n <= 8) { DISPATCH_MIX_ADD_FUSED(8); }
    else if (n <= 16) { DISPATCH_MIX_ADD_FUSED(16); }
    else if (n <= 32) { DISPATCH_MIX_ADD_FUSED(32); }
    else { fprintf(stderr, "stream_distribute_mix_add_fused: n > 32 not implemented\n"); }
    #undef DISPATCH_MIX_ADD_FUSED
}

inline void stream_aggregate_bf16_dynamic(floatX* out, const float* inp, const float* H_pre, int B,
                                          int n, int C, hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * C + BLOCK - 1) / BLOCK;

    #define DISPATCH_AGGREGATE_DYN(MAX_N_VAL) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_aggregate_bf16_dynamic_kernel<BLOCK, MAX_N_VAL>), \
                           dim3(blocks), dim3(BLOCK), 0, stream, out, inp, H_pre, B, n, C)

    if (n <= 4) { DISPATCH_AGGREGATE_DYN(4); }
    else if (n <= 8) { DISPATCH_AGGREGATE_DYN(8); }
    else if (n <= 16) { DISPATCH_AGGREGATE_DYN(16); }
    else if (n <= 32) { DISPATCH_AGGREGATE_DYN(32); }
    else { fprintf(stderr, "stream_aggregate_bf16_dynamic: n > 32 not implemented\n"); }
    #undef DISPATCH_AGGREGATE_DYN
}

inline void stream_distribute_mix_add_fused_dynamic(float* out, const float* x_inp,
                                                    const floatX* y_norm, const float* H_post,
                                                    const float* M, int B, int n, int C,
                                                    hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks = (B * n * C + BLOCK - 1) / BLOCK;

    #define DISPATCH_MIX_ADD_DYN(MAX_N_VAL) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_distribute_mix_add_dynamic_kernel<BLOCK, MAX_N_VAL>), \
                           dim3(blocks), dim3(BLOCK), 0, stream, out, x_inp, y_norm, H_post, M, B, n, C)

    if (n <= 4) { DISPATCH_MIX_ADD_DYN(4); }
    else if (n <= 8) { DISPATCH_MIX_ADD_DYN(8); }
    else if (n <= 16) { DISPATCH_MIX_ADD_DYN(16); }
    else if (n <= 32) { DISPATCH_MIX_ADD_DYN(32); }
    else { fprintf(stderr, "stream_distribute_mix_add_fused_dynamic: n > 32 not implemented\n"); }
    #undef DISPATCH_MIX_ADD_DYN
}

// ============================================================================
// Backward Kernels
// ============================================================================

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_backward_dx_kernel(float* __restrict__ d_inp,
                                                    const float* __restrict__ grad,
                                                    const float* __restrict__ H_pre, int B, int n, int C) {
    __shared__ float s_H_pre[MAX_N];
    if (threadIdx.x < n)
        s_H_pre[threadIdx.x] = H_pre[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int i = remainder / C;
    int c = remainder % C;
    d_inp[idx] = grad[b * C + c] * s_H_pre[i];
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_aggregate_backward_dH_partial_kernel(float* __restrict__ partials,
                                                            const float* __restrict__ grad,
                                                            const float* __restrict__ inp, int B,
                                                            int n, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
    __shared__ float s_warp_sums[MAX_N][BLOCK_SIZE / 64];

    float local_sum[MAX_N] = {0.0f};
    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < B * C; idx += gridDim.x * BLOCK_SIZE) {
        int b = idx / C, c = idx % C;
        float g = grad[idx];
        #pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n)
                local_sum[i] += g * inp[b * n * C + i * C + c];
        }
    }

    int warp_id = threadIdx.x / 64;
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float warp_sum = cg::reduce(warp, local_sum[i], cg::plus<float>());
            if (warp.thread_rank() == 0)
                s_warp_sums[i][warp_id] = warp_sum;
        }
    }
    block.sync();

    if (threadIdx.x < n) {
        float block_sum = 0.0f;
        for (int w = 0; w < BLOCK_SIZE / 64; w++)
            block_sum += s_warp_sums[threadIdx.x][w];
        partials[blockIdx.x * n + threadIdx.x] = block_sum;
    }
}

template<int MAX_N>
__global__ void reduce_partials_kernel(float* __restrict__ out, const float* __restrict__ partials,
                                       int n, int num_partials) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
    int i = blockIdx.x;
    if (i >= n)
        return;

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partials; p += blockDim.x)
        sum += partials[p * n + i];
    sum = cg::reduce(warp, sum, cg::plus<float>());

    __shared__ float s_warp_sums[8];
    if (warp.thread_rank() == 0)
        s_warp_sums[threadIdx.x / 64] = sum;
    block.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 63) / 64; w++)
            total += s_warp_sums[w];
        out[i] = total;
    }
}

inline void stream_aggregate_backward(float* d_inp, float* d_H_pre, const float* grad,
                                      const float* inp, const float* H_pre, int B, int n, int C,
                                      float* workspace, int workspace_num_blocks,
                                      hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int blocks_dx = (B * n * C + BLOCK - 1) / BLOCK;

    #define DISPATCH_AGG_BWD(MAX_N_VAL) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_aggregate_backward_dx_kernel<BLOCK, MAX_N_VAL>), \
                           dim3(blocks_dx), dim3(BLOCK), 0, stream, d_inp, grad, H_pre, B, n, C); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_aggregate_backward_dH_partial_kernel<BLOCK, MAX_N_VAL>), \
                           dim3(workspace_num_blocks), dim3(BLOCK), 0, stream, workspace, grad, inp, B, n, C); \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_partials_kernel<MAX_N_VAL>), \
                           dim3(n), dim3(128), 0, stream, d_H_pre, workspace, n, workspace_num_blocks)

    if (n <= 4) { DISPATCH_AGG_BWD(4); }
    else if (n <= 8) { DISPATCH_AGG_BWD(8); }
    else if (n <= 16) { DISPATCH_AGG_BWD(16); }
    else if (n <= 32) { DISPATCH_AGG_BWD(32); }
    else { fprintf(stderr, "stream_aggregate_backward: n > 32 not implemented\n"); }
    #undef DISPATCH_AGG_BWD
}

// ============================================================================
// Distribute Mix Backward Kernels
// ============================================================================

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_dx_dy_kernel(
    float* __restrict__ d_x, float* __restrict__ d_y_norm, const float* __restrict__ grad,
    const float* __restrict__ M, const float* __restrict__ H_post, int B, int n, int C) {
    __shared__ float s_M[MAX_N * MAX_N];
    __shared__ float s_H[MAX_N];

    if (threadIdx.x < n * n)
        s_M[threadIdx.x] = M[threadIdx.x];
    if (threadIdx.x < n)
        s_H[threadIdx.x] = H_post[threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= B * n * C)
        return;

    int b = idx / (n * C);
    int remainder = idx % (n * C);
    int j = remainder / C;
    int c = remainder % C;

    float dx_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n)
            dx_sum += s_M[i * n + j] * grad[b * n * C + i * C + c];
    }
    d_x[idx] = dx_sum;

    if (j == 0) {
        float dy_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n)
                dy_sum += s_H[i] * grad[b * n * C + i * C + c];
        }
        d_y_norm[b * C + c] = dy_sum;
    }
}

template<int BLOCK_SIZE, int MAX_N>
__global__ void stream_distribute_mix_backward_partials_kernel(
    float* __restrict__ partials_M, float* __restrict__ partials_H, const float* __restrict__ grad,
    const float* __restrict__ x, const float* __restrict__ y_norm, int B, int n, int C) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
    constexpr int NUM_WARPS = BLOCK_SIZE / 64;
    __shared__ float s_warp_M[MAX_N][MAX_N][NUM_WARPS];
    __shared__ float s_warp_H[MAX_N][NUM_WARPS];

    float local_M[MAX_N][MAX_N];
    float local_H[MAX_N];
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        local_H[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < MAX_N; j++)
            local_M[i][j] = 0.0f;
    }

    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < B * C; idx += gridDim.x * BLOCK_SIZE) {
        int b = idx / C, c = idx % C;
        float y_val = y_norm[b * C + c];
        #pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n) {
                float g = grad[b * n * C + i * C + c];
                local_H[i] += g * y_val;
                #pragma unroll
                for (int j = 0; j < MAX_N; j++) {
                    if (j < n)
                        local_M[i][j] += g * x[b * n * C + j * C + c];
                }
            }
        }
    }

    int warp_id = threadIdx.x / 64;
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            #pragma unroll
            for (int j = 0; j < MAX_N; j++) {
                if (j < n) {
                    float ws = cg::reduce(warp, local_M[i][j], cg::plus<float>());
                    if (warp.thread_rank() == 0)
                        s_warp_M[i][j][warp_id] = ws;
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float ws = cg::reduce(warp, local_H[i], cg::plus<float>());
            if (warp.thread_rank() == 0)
                s_warp_H[i][warp_id] = ws;
        }
    }
    block.sync();

    if (threadIdx.x < n * n) {
        int i = threadIdx.x / n, j = threadIdx.x % n;
        float bs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            bs += s_warp_M[i][j][w];
        partials_M[blockIdx.x * n * n + threadIdx.x] = bs;
    }
    if (threadIdx.x < n) {
        float bs = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            bs += s_warp_H[threadIdx.x][w];
        partials_H[blockIdx.x * n + threadIdx.x] = bs;
    }
}

template<int MAX_N>
__global__ void reduce_partials_matrix_kernel(float* __restrict__ out,
                                              const float* __restrict__ partials, int n,
                                              int num_partials) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
    int k = blockIdx.x;
    if (k >= n * n)
        return;

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partials; p += blockDim.x)
        sum += partials[p * n * n + k];
    sum = cg::reduce(warp, sum, cg::plus<float>());

    __shared__ float s_warp_sums[8];
    if (warp.thread_rank() == 0)
        s_warp_sums[threadIdx.x / 64] = sum;
    block.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 63) / 64; w++)
            total += s_warp_sums[w];
        out[k] = total;
    }
}

inline void stream_distribute_mix_backward_fused(float* d_x, float* d_y_norm, float* d_M,
                                                 float* d_H_post, const float* grad, const float* x,
                                                 const float* y_norm, const float* M,
                                                 const float* H_post, int B, int n, int C,
                                                 float* workspace_M, float* workspace_H,
                                                 int workspace_num_blocks,
                                                 hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;

    #define DISPATCH_DIST_BWD(MAX_N_VAL) \
        do { \
            int blocks = (B * n * C + BLOCK - 1) / BLOCK; \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_distribute_mix_backward_dx_dy_kernel<BLOCK, MAX_N_VAL>), \
                               dim3(blocks), dim3(BLOCK), 0, stream, d_x, d_y_norm, grad, M, H_post, B, n, C); \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_distribute_mix_backward_partials_kernel<BLOCK, MAX_N_VAL>), \
                               dim3(workspace_num_blocks), dim3(BLOCK), 0, stream, workspace_M, workspace_H, grad, x, y_norm, B, n, C); \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_partials_matrix_kernel<MAX_N_VAL>), \
                               dim3(n * n), dim3(128), 0, stream, d_M, workspace_M, n, workspace_num_blocks); \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_partials_kernel<MAX_N_VAL>), \
                               dim3(n), dim3(128), 0, stream, d_H_post, workspace_H, n, workspace_num_blocks); \
        } while (0)

    if (n <= 4) { DISPATCH_DIST_BWD(4); }
    else if (n <= 8) { DISPATCH_DIST_BWD(8); }
    else if (n <= 16) { DISPATCH_DIST_BWD(16); }
    else if (n <= 32) { DISPATCH_DIST_BWD(32); }
    else { fprintf(stderr, "stream_distribute_mix_backward_fused: n > 32 not implemented\n"); }
    #undef DISPATCH_DIST_BWD
}

// ============================================================================
// StreamMixTC Class (using hipBLASLt for matmul)
// ============================================================================

class StreamMixTC {
  public:
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t matmulDesc;
    hipblasLtMatrixLayout_t Mdesc, Xdesc, Ydesc;
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;
    int B, n, C;
    bool initialized = false;

    void init(int B_, int n_, int C_) {
        B = B_;
        n = n_;
        C = C_;
        workspace_size = 4 * 1024 * 1024;

        CHECK_HIPBLAS(hipblasLtCreate(&handle));
        CHECK_HIPBLAS(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        hipblasOperation_t trans_a = HIPBLAS_OP_N;
        hipblasOperation_t trans_b = HIPBLAS_OP_T;
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

        hipblasLtOrder_t row_order = HIPBLASLT_ORDER_ROW;
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&Xdesc, HIP_R_32F, B * C, n, n));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(Xdesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&Mdesc, HIP_R_32F, n, n, n));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(Mdesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&Ydesc, HIP_R_32F, B * C, n, n));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(Ydesc, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

        CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&preference));
        CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspace_size, sizeof(workspace_size)));

        int returned_results = 0;
        CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Xdesc, Mdesc, Ydesc, Ydesc, preference,
                                                     1, &heuristic, &returned_results));

        CHECK_HIP(hipMalloc(&workspace, workspace_size));
        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(Mdesc);
        hipblasLtMatrixLayoutDestroy(Xdesc);
        hipblasLtMatrixLayoutDestroy(Ydesc);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(handle);
        hipFree(workspace);
        initialized = false;
    }

    void forward(float* out, const float* inp, const float* M, hipStream_t stream = nullptr) {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_HIPBLAS(hipblasLtMatmul(handle, matmulDesc, &alpha, inp, Xdesc, M, Mdesc, &beta, out, Ydesc, out,
                                      Ydesc, &heuristic.algo, workspace, workspace_size, stream));
    }
};

} // namespace mhc

