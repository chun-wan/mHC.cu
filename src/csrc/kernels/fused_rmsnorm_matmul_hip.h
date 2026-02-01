#pragma once

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "../include/mhc_types_hip.h"

namespace cg = cooperative_groups;

namespace mhc {

// ============================================================================
// Compute RMS Kernels
// ============================================================================

template<int BLOCK_SIZE>
__global__ void compute_rms_kernel(float* __restrict__ rms_out, const floatX* __restrict__ inp,
                                   int N, int C, float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const floatX* x = inp + idx * C;

    extern __shared__ float shared[];

    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = static_cast<float>(x[i]);
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    int num_warps = BLOCK_SIZE / 64;

    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

template<int BLOCK_SIZE>
__global__ void compute_rms_kernel_vectorized(float* __restrict__ rms_out,
                                              const floatX* __restrict__ inp, int N, int C,
                                              float eps) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);

    int idx = blockIdx.x;
    if (idx >= N)
        return;

    const floatX* x = inp + idx * C;

    extern __shared__ float shared[];

    constexpr int VEC_SIZE = 4;
    int C_vec = C / VEC_SIZE;

    float thread_sum_sq = 0.0f;

    using vec_t = float2;
    const vec_t* x_vec = reinterpret_cast<const vec_t*>(x);

    for (int i = threadIdx.x; i < C_vec; i += BLOCK_SIZE) {
        vec_t v = x_vec[i];
        hip_bfloat16* bf_v = reinterpret_cast<hip_bfloat16*>(&v);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float f = static_cast<float>(bf_v[j]);
            thread_sum_sq += f * f;
        }
    }

    int remainder_start = C_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = static_cast<float>(x[i]);
        thread_sum_sq += val * val;
    }

    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());

    int warp_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    int num_warps = BLOCK_SIZE / 64;

    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());

        if (lane_id == 0) {
            float rms = sqrtf(block_sum / (float)C + eps);
            rms_out[idx] = rms;
        }
    }
}

inline void compute_rms(float* rms_out, const floatX* inp, int N, int C, float eps,
                        hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_warps = BLOCK_SIZE / 64;
    size_t shared_mem = num_warps * sizeof(float);

    if (C % 4 == 0 && C >= 64) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_rms_kernel_vectorized<BLOCK_SIZE>),
                           dim3(N), dim3(BLOCK_SIZE), shared_mem, stream, rms_out, inp, N, C, eps);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(compute_rms_kernel<BLOCK_SIZE>),
                           dim3(N), dim3(BLOCK_SIZE), shared_mem, stream, rms_out, inp, N, C, eps);
    }
}

// ============================================================================
// Divide by RMS Kernels
// ============================================================================

template<int BLOCK_SIZE>
__global__ void divide_by_rms_kernel(float* __restrict__ out, const float* __restrict__ rms, int M,
                                     int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    float r_inv = 1.0f / rms[row];
    float* out_row = out + row * N;

    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        out_row[i] *= r_inv;
    }
}

template<int BLOCK_SIZE>
__global__ void divide_by_rms_kernel_vectorized(float* __restrict__ out,
                                                const float* __restrict__ rms, int M, int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    float r_inv = 1.0f / rms[row];
    float* out_row = out + row * N;

    constexpr int VEC_SIZE = 4;
    int N_vec = N / VEC_SIZE;

    float4* out_vec = reinterpret_cast<float4*>(out_row);

    for (int i = threadIdx.x; i < N_vec; i += BLOCK_SIZE) {
        float4 v = out_vec[i];
        v.x *= r_inv;
        v.y *= r_inv;
        v.z *= r_inv;
        v.w *= r_inv;
        out_vec[i] = v;
    }

    int remainder_start = N_vec * VEC_SIZE;
    for (int i = remainder_start + threadIdx.x; i < N; i += BLOCK_SIZE) {
        out_row[i] *= r_inv;
    }
}

inline void divide_by_rms(float* out, const float* rms, int M, int N,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;

    if (N % 4 == 0 && N >= 16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(divide_by_rms_kernel_vectorized<BLOCK_SIZE>),
                           dim3(M), dim3(BLOCK_SIZE), 0, stream, out, rms, M, N);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(divide_by_rms_kernel<BLOCK_SIZE>),
                           dim3(M), dim3(BLOCK_SIZE), 0, stream, out, rms, M, N);
    }
}

// ============================================================================
// Matmul Descriptors for hipBLASLt
// ============================================================================

struct MatmulDescriptors {
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t A_desc;
    hipblasLtMatrixLayout_t B_desc;
    hipblasLtMatrixLayout_t C_desc;
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    size_t workspace_size;
};

inline void init_matmul_descriptors(MatmulDescriptors& desc, int M, int N, int K,
                                    size_t workspace_size = 32 * 1024 * 1024) {
    CHECK_HIPBLAS(hipblasLtCreate(&desc.handle));

    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
    hipDataType ab_type = HIP_R_16BF;
    hipDataType c_type = HIP_R_32F;
    hipDataType scale_type = HIP_R_32F;

    CHECK_HIPBLAS(hipblasLtMatmulDescCreate(&desc.matmul_desc, compute_type, scale_type));

    hipblasOperation_t trans_a = HIPBLAS_OP_N;
    hipblasOperation_t trans_b = HIPBLAS_OP_T;
    CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(desc.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                  &trans_a, sizeof(trans_a)));
    CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(desc.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                  &trans_b, sizeof(trans_b)));

    hipblasLtOrder_t row_order = HIPBLASLT_ORDER_ROW;
    CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&desc.A_desc, ab_type, M, K, K));
    CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(desc.A_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                    &row_order, sizeof(row_order)));
    CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&desc.B_desc, ab_type, N, K, K));
    CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(desc.B_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                    &row_order, sizeof(row_order)));
    CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&desc.C_desc, c_type, M, N, N));
    CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(desc.C_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                    &row_order, sizeof(row_order)));

    CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&desc.preference));
    CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(desc.preference,
                                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace_size, sizeof(workspace_size)));

    int returned_results = 0;
    CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(
        desc.handle, desc.matmul_desc, desc.A_desc, desc.B_desc, desc.C_desc, desc.C_desc,
        desc.preference, 1, &desc.heuristic, &returned_results));

    if (returned_results == 0) {
        fprintf(stderr, "No hipBLASLt algorithm found for row-major matmul\n");
        exit(EXIT_FAILURE);
    }

    desc.workspace_size = workspace_size;
    CHECK_HIP(hipMalloc(&desc.workspace, workspace_size));
}

inline void destroy_matmul_descriptors(MatmulDescriptors& desc) {
    hipblasLtMatmulPreferenceDestroy(desc.preference);
    hipblasLtMatrixLayoutDestroy(desc.A_desc);
    hipblasLtMatrixLayoutDestroy(desc.B_desc);
    hipblasLtMatrixLayoutDestroy(desc.C_desc);
    hipblasLtMatmulDescDestroy(desc.matmul_desc);
    hipblasLtDestroy(desc.handle);
    hipFree(desc.workspace);
}

inline void matmul_forward(MatmulDescriptors& desc, float* out, const floatX* A, const floatX* B,
                           float alpha, float beta, hipStream_t stream = nullptr) {
    CHECK_HIPBLAS(hipblasLtMatmul(desc.handle, desc.matmul_desc, &alpha, A, desc.A_desc, B,
                                  desc.B_desc, &beta, out, desc.C_desc, out, desc.C_desc,
                                  &desc.heuristic.algo, desc.workspace, desc.workspace_size, stream));
}

// ============================================================================
// FusedRMSNormMatmul Class
// ============================================================================

struct FusedRMSNormMatmul {
    MatmulDescriptors matmul_desc;
    float* rms_buffer;
    int M, N, K;
    float eps;
    bool initialized;

    FusedRMSNormMatmul() : rms_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;
        eps = epsilon;

        init_matmul_descriptors(matmul_desc, M, N, K);
        CHECK_HIP(hipMalloc(&rms_buffer, M * sizeof(float)));
        initialized = true;
    }

    void destroy() {
        if (initialized) {
            destroy_matmul_descriptors(matmul_desc);
            hipFree(rms_buffer);
            initialized = false;
        }
    }

    void forward(float* out, const floatX* inp, const floatX* proj_weight,
                 hipStream_t stream = nullptr) {
        compute_rms(rms_buffer, inp, M, K, eps, stream);
        matmul_forward(matmul_desc, out, inp, proj_weight, 1.0f, 0.0f, stream);
        divide_by_rms(out, rms_buffer, M, N, stream);
    }

    float* get_rms_values() { return rms_buffer; }
};

// ============================================================================
// BF16 to FP32 and Backward Kernels
// ============================================================================

template<int BLOCK_SIZE>
__global__ void bf16_to_fp32_kernel(float* __restrict__ out, const floatX* __restrict__ inp,
                                    int total) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total)
        return;
    out[idx] = static_cast<float>(inp[idx]);
}

template<int BLOCK_SIZE>
__global__ void scale_grad_by_rms_kernel(float* __restrict__ grad_scaled,
                                         const float* __restrict__ grad,
                                         const float* __restrict__ rms, int M, int N) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = M * N;
    if (idx >= total)
        return;

    int row = idx / N;
    float r_inv = 1.0f / rms[row];
    grad_scaled[idx] = grad[idx] * r_inv;
}

template<int BLOCK_SIZE>
__global__ void rms_correction_kernel(float* __restrict__ dx, const float* __restrict__ K_buf,
                                      const floatX* __restrict__ x, const float* __restrict__ rms,
                                      int M, int K_dim) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);

    int row = blockIdx.x;
    if (row >= M)
        return;

    extern __shared__ float shared[];

    const floatX* x_row = x + row * K_dim;
    const float* K_row = K_buf + row * K_dim;
    float* dx_row = dx + row * K_dim;
    float r = rms[row];

    int warp_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    int num_warps = BLOCK_SIZE / 64;

    float thread_dot = 0.0f;
    for (int i = threadIdx.x; i < K_dim; i += BLOCK_SIZE) {
        float xi = static_cast<float>(x_row[i]);
        thread_dot += K_row[i] * xi;
    }

    float warp_dot = cg::reduce(warp, thread_dot, cg::plus<float>());
    if (lane_id == 0) {
        shared[warp_id] = warp_dot;
    }
    __syncthreads();

    float K_dot_x = 0.0f;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        K_dot_x = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            shared[0] = K_dot_x;
        }
    }
    __syncthreads();
    K_dot_x = shared[0];

    float correction_scale = K_dot_x / ((float)K_dim * r * r);

    for (int i = threadIdx.x; i < K_dim; i += BLOCK_SIZE) {
        float xi = static_cast<float>(x_row[i]);
        dx_row[i] = K_row[i] - correction_scale * xi;
    }
}

// ============================================================================
// FusedRMSNormMatmulBackward Class
// ============================================================================

struct FusedRMSNormMatmulBackward {
    hipblasLtHandle_t handle;
    hipblasLtMatmulDesc_t dW_matmul_desc;
    hipblasLtMatrixLayout_t dW_grad_desc;
    hipblasLtMatrixLayout_t dW_x_desc;
    hipblasLtMatrixLayout_t dW_out_desc;
    hipblasLtMatmulPreference_t dW_pref;
    hipblasLtMatmulHeuristicResult_t dW_heuristic;

    hipblasLtMatmulDesc_t dx_matmul_desc;
    hipblasLtMatrixLayout_t dx_grad_desc;
    hipblasLtMatrixLayout_t dx_W_desc;
    hipblasLtMatrixLayout_t dx_out_desc;
    hipblasLtMatmulPreference_t dx_pref;
    hipblasLtMatmulHeuristicResult_t dx_heuristic;

    void* workspace;
    size_t workspace_size;
    float* grad_scaled_buffer;
    float* K_buffer;
    float* x_fp32_buffer;
    float* W_fp32_buffer;

    int M, N, K;
    bool initialized;

    FusedRMSNormMatmulBackward()
        : workspace(nullptr), grad_scaled_buffer(nullptr), K_buffer(nullptr),
          x_fp32_buffer(nullptr), W_fp32_buffer(nullptr), initialized(false) {}

    void init(int m, int n, int k, float epsilon = 1e-5f) {
        M = m;
        N = n;
        K = k;

        workspace_size = 32 * 1024 * 1024;
        CHECK_HIP(hipMalloc(&workspace, workspace_size));
        CHECK_HIP(hipMalloc(&grad_scaled_buffer, M * N * sizeof(float)));
        CHECK_HIP(hipMalloc(&K_buffer, M * K * sizeof(float)));
        CHECK_HIP(hipMalloc(&x_fp32_buffer, M * K * sizeof(float)));
        CHECK_HIP(hipMalloc(&W_fp32_buffer, N * K * sizeof(float)));

        CHECK_HIPBLAS(hipblasLtCreate(&handle));

        hipblasLtOrder_t row_order = HIPBLASLT_ORDER_ROW;

        CHECK_HIPBLAS(hipblasLtMatmulDescCreate(&dW_matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t trans_a = HIPBLAS_OP_T;
        hipblasOperation_t trans_b = HIPBLAS_OP_N;
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(dW_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                      &trans_a, sizeof(trans_a)));
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(dW_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                      &trans_b, sizeof(trans_b)));

        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&dW_grad_desc, HIP_R_32F, M, N, N));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(dW_grad_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&dW_x_desc, HIP_R_32F, M, K, K));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(dW_x_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&dW_out_desc, HIP_R_32F, N, K, K));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(dW_out_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));

        CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&dW_pref));
        CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(dW_pref,
                                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                            &workspace_size, sizeof(workspace_size)));

        int returned = 0;
        CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(handle, dW_matmul_desc, dW_grad_desc, dW_x_desc,
                                                      dW_out_desc, dW_out_desc, dW_pref, 1,
                                                      &dW_heuristic, &returned));
        if (returned == 0) {
            fprintf(stderr, "No hipBLASLt algorithm found for dW backward matmul\n");
            exit(EXIT_FAILURE);
        }

        CHECK_HIPBLAS(hipblasLtMatmulDescCreate(&dx_matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        trans_a = HIPBLAS_OP_N;
        trans_b = HIPBLAS_OP_N;
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(dx_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                      &trans_a, sizeof(trans_a)));
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(dx_matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                      &trans_b, sizeof(trans_b)));

        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&dx_grad_desc, HIP_R_32F, M, N, N));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(dx_grad_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&dx_W_desc, HIP_R_32F, N, K, K));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(dx_W_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&dx_out_desc, HIP_R_32F, M, K, K));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(dx_out_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));

        CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&dx_pref));
        CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(dx_pref,
                                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                            &workspace_size, sizeof(workspace_size)));

        returned = 0;
        CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(handle, dx_matmul_desc, dx_grad_desc, dx_W_desc,
                                                      dx_out_desc, dx_out_desc, dx_pref, 1,
                                                      &dx_heuristic, &returned));
        if (returned == 0) {
            fprintf(stderr, "No hipBLASLt algorithm found for dx backward matmul\n");
            exit(EXIT_FAILURE);
        }

        initialized = true;
    }

    void destroy() {
        if (initialized) {
            hipblasLtMatmulPreferenceDestroy(dW_pref);
            hipblasLtMatrixLayoutDestroy(dW_grad_desc);
            hipblasLtMatrixLayoutDestroy(dW_x_desc);
            hipblasLtMatrixLayoutDestroy(dW_out_desc);
            hipblasLtMatmulDescDestroy(dW_matmul_desc);

            hipblasLtMatmulPreferenceDestroy(dx_pref);
            hipblasLtMatrixLayoutDestroy(dx_grad_desc);
            hipblasLtMatrixLayoutDestroy(dx_W_desc);
            hipblasLtMatrixLayoutDestroy(dx_out_desc);
            hipblasLtMatmulDescDestroy(dx_matmul_desc);

            hipblasLtDestroy(handle);
            hipFree(workspace);
            hipFree(grad_scaled_buffer);
            hipFree(K_buffer);
            hipFree(x_fp32_buffer);
            hipFree(W_fp32_buffer);
            initialized = false;
        }
    }

    void backward(float* dW, float* dx_out, const float* grad_output, const floatX* x,
                  const floatX* weight, const float* rms, hipStream_t stream = nullptr) {
        constexpr int BLOCK_SIZE = 256;

        int total_x = M * K;
        int num_blocks_x = (total_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(bf16_to_fp32_kernel<BLOCK_SIZE>),
                           dim3(num_blocks_x), dim3(BLOCK_SIZE), 0, stream, x_fp32_buffer, x, total_x);

        int total_w = N * K;
        int num_blocks_w = (total_w + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(bf16_to_fp32_kernel<BLOCK_SIZE>),
                           dim3(num_blocks_w), dim3(BLOCK_SIZE), 0, stream, W_fp32_buffer, weight, total_w);

        int total = M * N;
        int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(scale_grad_by_rms_kernel<BLOCK_SIZE>),
                           dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream, grad_scaled_buffer, grad_output, rms, M, N);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_HIPBLAS(hipblasLtMatmul(handle, dW_matmul_desc, &alpha, grad_scaled_buffer,
                                      dW_grad_desc, x_fp32_buffer, dW_x_desc, &beta, dW, dW_out_desc,
                                      dW, dW_out_desc, &dW_heuristic.algo, workspace, workspace_size,
                                      stream));

        CHECK_HIPBLAS(hipblasLtMatmul(handle, dx_matmul_desc, &alpha, grad_scaled_buffer,
                                      dx_grad_desc, W_fp32_buffer, dx_W_desc, &beta, K_buffer,
                                      dx_out_desc, K_buffer, dx_out_desc, &dx_heuristic.algo,
                                      workspace, workspace_size, stream));

        int num_warps = BLOCK_SIZE / 64;
        size_t shared_mem = num_warps * sizeof(float);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rms_correction_kernel<BLOCK_SIZE>),
                           dim3(M), dim3(BLOCK_SIZE), shared_mem, stream, dx_out, K_buffer, x, rms, M, K);
    }
};

} // namespace mhc

