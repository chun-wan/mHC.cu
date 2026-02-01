#pragma once

#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "mhc_types_hip.h"

namespace mhc {

// ============================================================================
// Type Conversion Kernels
// ============================================================================

template<int BLOCK_SIZE>
__global__ void float_to_bf16_kernel(floatX* __restrict__ out, const float* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<floatX>(inp[idx]);
    }
}

template<int BLOCK_SIZE>
__global__ void bf16_to_float_kernel(float* __restrict__ out, const floatX* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<float>(inp[idx]);
    }
}

inline void float_to_bf16(floatX* out, const float* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(float_to_bf16_kernel<BLOCK_SIZE>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       out, inp, size);
}

inline void bf16_to_float(float* out, const floatX* inp, int size, hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(bf16_to_float_kernel<BLOCK_SIZE>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                       out, inp, size);
}

// ============================================================================
// Fast Math Intrinsics for AMD GPUs
// ============================================================================

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

// Reciprocal approximation - AMD equivalent
__device__ __forceinline__ float fast_rcp(float x) {
    return __frcp_rn(x);
}

// ============================================================================
// Fused H Activations Kernel
// ============================================================================

template<int BLOCK_SIZE>
__global__ void fused_h_activations_kernel(
    float* __restrict__ H_pre_out, float* __restrict__ H_post_out, float* __restrict__ H_res_out,
    const float* __restrict__ H_proj_concat, const float* __restrict__ rms, float alpha_pre,
    float alpha_post, float alpha_res, const float* __restrict__ b_pre,
    const float* __restrict__ b_post, const float* __restrict__ b_res, int B, int n) {
    int n_sq = n * n;
    int total_pre = B * n;
    int total_post = B * n;
    int total_res = B * n_sq;
    int stride = n + n + n_sq;

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < total_pre) {
        int b = idx / n;
        int j = idx % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + j];
        val = alpha_pre * val * r_inv + b_pre[j];
        H_pre_out[idx] = fast_sigmoid(val);
    }

    int idx2 = idx;
    if (idx2 < total_post) {
        int b = idx2 / n;
        int j = idx2 % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + n + j];
        val = alpha_post * val * r_inv + b_post[j];
        H_post_out[idx2] = 2.0f * fast_sigmoid(val);
    }

    int idx3 = idx;
    if (idx3 < total_res) {
        int b = idx3 / n_sq;
        int local = idx3 % n_sq;
        int i = local / n;
        int j = local % n;
        float r_inv = 1.0f / rms[b];
        float val = H_proj_concat[b * stride + n + n + local];
        val = alpha_res * val * r_inv + b_res[i * n + j];
        H_res_out[idx3] = fast_exp(val);
    }
}

inline void fused_h_activations(float* H_pre_out, float* H_post_out, float* H_res_out,
                                const float* H_proj_concat, const float* rms, float alpha_pre,
                                float alpha_post, float alpha_res, const float* b_pre,
                                const float* b_post, const float* b_res, int B, int n,
                                hipStream_t stream = nullptr) {
    constexpr int BLOCK = 256;
    int n_sq = n * n;
    int max_total = B * n_sq;
    int blocks = (max_total + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(fused_h_activations_kernel<BLOCK>, dim3(blocks), dim3(BLOCK), 0, stream,
                       H_pre_out, H_post_out, H_res_out, H_proj_concat, rms, alpha_pre, alpha_post,
                       alpha_res, b_pre, b_post, b_res, B, n);
}

// ============================================================================
// L2 Cache Flusher for Benchmarking
// ============================================================================

__global__ void flush_l2_kernel(float* buf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] = buf[idx] + 1.0f;
    }
}

struct L2Flusher {
    // MI300X has 256MB L3 cache, but we use L2 equivalent
    static constexpr int L2_SIZE_BYTES = 96 * 1024 * 1024;  // 96MB for MI300X L2
    static constexpr int FLUSH_SIZE = L2_SIZE_BYTES / sizeof(float) * 2;
    float* buf;

    L2Flusher() : buf(nullptr) {
        hipMalloc(&buf, FLUSH_SIZE * sizeof(float));
        hipMemset(buf, 0, FLUSH_SIZE * sizeof(float));
    }

    ~L2Flusher() {
        if (buf)
            hipFree(buf);
    }

    void flush() {
        int block_size = 256;
        int num_blocks = (FLUSH_SIZE + block_size - 1) / block_size;
        hipLaunchKernelGGL(flush_l2_kernel, dim3(num_blocks), dim3(block_size), 0, 0, buf, FLUSH_SIZE);
        hipDeviceSynchronize();
    }
};

// ============================================================================
// Test Utilities
// ============================================================================

inline float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

inline bool check_test(float max_diff, float tolerance, const char* test_name = nullptr) {
    if (test_name) {
        printf("%s: ", test_name);
    }
    printf("max diff = %e, ", max_diff);
    if (max_diff < tolerance) {
        printf("PASSED (tol: %e)\n", tolerance);
        return true;
    } else {
        printf("FAILED (tol: %e)\n", tolerance);
        return false;
    }
}

// ============================================================================
// Benchmark Timer
// ============================================================================

struct BenchTimer {
    hipEvent_t start, stop;

    BenchTimer() {
        hipEventCreate(&start);
        hipEventCreate(&stop);
    }

    ~BenchTimer() {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    void record_start() { hipEventRecord(start); }

    void record_stop() { hipEventRecord(stop); }

    float elapsed_ms() {
        hipEventSynchronize(stop);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// Profiler (simplified for HIP - no PTX inline asm)
// ============================================================================

enum ProfilerTag {
    TagSetup = 0,
    TagLoad,
    TagCompute,
    TagReduce,
    TagStore,
    TagSync,
    TagOther,
    TagCount
};

inline const char* profiler_tag_name(ProfilerTag tag) {
    switch (tag) {
    case TagSetup:
        return "Setup";
    case TagLoad:
        return "Load";
    case TagCompute:
        return "Compute";
    case TagReduce:
        return "Reduce";
    case TagStore:
        return "Store";
    case TagSync:
        return "Sync";
    case TagOther:
        return "Other";
    default:
        return "Unknown";
    }
}

// Note: AMD GPUs use different timing mechanisms
// globaltimer() equivalent not directly available in HIP
// Use wall_clock64() for AMD GPUs if available in ROCm 6.x+

} // namespace mhc

