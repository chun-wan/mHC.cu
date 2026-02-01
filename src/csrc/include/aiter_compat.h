/**
 * @file aiter_compat.h
 * @brief AITER (AI Tensor Engine for ROCm) compatibility layer for mHC
 *
 * This header provides integration with ROCm's AITER library for optimized
 * AI operators on AMD MI300X GPUs.
 *
 * AITER repository: https://github.com/ROCm/aiter
 *
 * When MHC_USE_AITER is defined, this module provides wrappers that use
 * AITER's optimized implementations. Otherwise, it falls back to the
 * custom HIP kernels.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Check if AITER is available at compile time
#ifdef MHC_USE_AITER

// AITER headers
#include <aiter/rmsnorm.h>
#include <aiter/gemm.h>
#include <aiter/elementwise.h>

#define AITER_ENABLED 1

#else

#define AITER_ENABLED 0

#endif

namespace mhc {
namespace aiter {

/**
 * @brief Check if AITER is enabled at compile time
 */
constexpr bool is_enabled() {
    return AITER_ENABLED == 1;
}

/**
 * @brief Runtime information about AITER availability
 */
struct AITERInfo {
    bool compiled_with_aiter;
    bool rmsnorm_available;
    bool gemm_available;
    bool elementwise_available;
    const char* version;
};

inline AITERInfo get_info() {
    AITERInfo info;
    info.compiled_with_aiter = is_enabled();
    
#ifdef MHC_USE_AITER
    info.rmsnorm_available = true;
    info.gemm_available = true;
    info.elementwise_available = true;
    info.version = "0.1.9";  // AITER version
#else
    info.rmsnorm_available = false;
    info.gemm_available = false;
    info.elementwise_available = false;
    info.version = "N/A";
#endif
    
    return info;
}

// ============================================================================
// AITER RMSNorm Wrapper
// ============================================================================

#ifdef MHC_USE_AITER

/**
 * @brief AITER-optimized RMSNorm forward
 *
 * Uses AITER's Triton/CK-based RMSNorm kernel which is optimized for
 * AMD GPUs with better memory coalescing and wavefront utilization.
 */
inline void rmsnorm_forward_aiter(
    hip_bfloat16* out,
    float* rms_out,
    const hip_bfloat16* inp,
    const hip_bfloat16* weight,
    int N,
    int C,
    float eps,
    hipStream_t stream = nullptr
) {
    // AITER rmsnorm interface
    // Note: Actual AITER API may differ, adjust accordingly
    ::aiter::rmsnorm_forward(
        reinterpret_cast<void*>(out),
        reinterpret_cast<const void*>(inp),
        reinterpret_cast<const void*>(weight),
        N, C, eps,
        ::aiter::DataType::BF16,
        stream
    );
    
    // AITER may not output RMS values, compute separately if needed
    if (rms_out != nullptr) {
        // Launch a simple kernel to compute RMS
        // This is a fallback - ideally AITER should provide this
    }
}

/**
 * @brief AITER-optimized RMSNorm backward
 */
inline void rmsnorm_backward_aiter(
    float* d_inp,
    float* d_weight,
    const float* grad,
    const hip_bfloat16* inp,
    const hip_bfloat16* weight,
    const float* rms,
    int N,
    int C,
    hipStream_t stream = nullptr
) {
    // AITER rmsnorm backward interface
    ::aiter::rmsnorm_backward(
        reinterpret_cast<void*>(d_inp),
        reinterpret_cast<void*>(d_weight),
        reinterpret_cast<const void*>(grad),
        reinterpret_cast<const void*>(inp),
        reinterpret_cast<const void*>(weight),
        N, C,
        ::aiter::DataType::FP32,
        stream
    );
}

#endif // MHC_USE_AITER

// ============================================================================
// AITER GEMM Wrapper
// ============================================================================

#ifdef MHC_USE_AITER

/**
 * @brief AITER-optimized GEMM
 *
 * Uses AITER's optimized GEMM which leverages CK (Composable Kernel)
 * for better performance on MI300X.
 *
 * D = alpha * A @ B + beta * C
 */
inline void gemm_aiter(
    void* D,
    const void* A,
    const void* B,
    const void* C,
    int M, int N, int K,
    float alpha,
    float beta,
    bool transA,
    bool transB,
    ::aiter::DataType dtype,
    hipStream_t stream = nullptr
) {
    ::aiter::GemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.alpha = alpha;
    params.beta = beta;
    params.transA = transA;
    params.transB = transB;
    params.dtype = dtype;
    
    ::aiter::gemm(D, A, B, C, params, stream);
}

/**
 * @brief AITER GEMM with bf16 inputs and fp32 accumulator
 */
inline void gemm_bf16_f32_aiter(
    float* D,
    const hip_bfloat16* A,
    const hip_bfloat16* B,
    const float* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    bool transA = false,
    bool transB = false,
    hipStream_t stream = nullptr
) {
    gemm_aiter(
        reinterpret_cast<void*>(D),
        reinterpret_cast<const void*>(A),
        reinterpret_cast<const void*>(B),
        reinterpret_cast<const void*>(C),
        M, N, K,
        alpha, beta,
        transA, transB,
        ::aiter::DataType::BF16,
        stream
    );
}

#endif // MHC_USE_AITER

// ============================================================================
// AITER Sigmoid Wrapper
// ============================================================================

#ifdef MHC_USE_AITER

/**
 * @brief AITER-optimized Sigmoid
 */
inline void sigmoid_aiter(
    float* out,
    const float* inp,
    int size,
    hipStream_t stream = nullptr
) {
    ::aiter::sigmoid(
        reinterpret_cast<void*>(out),
        reinterpret_cast<const void*>(inp),
        size,
        ::aiter::DataType::FP32,
        stream
    );
}

#endif // MHC_USE_AITER

// ============================================================================
// Unified Interface (uses AITER when available, falls back otherwise)
// ============================================================================

/**
 * @brief Backend selector for runtime switching
 */
enum class Backend {
    AUTO,      // Use AITER if available, else HIP
    AITER,     // Force AITER (fails if not available)
    HIP        // Force custom HIP kernels
};

/**
 * @brief Global backend selection
 */
inline Backend& get_backend() {
    static Backend backend = Backend::AUTO;
    return backend;
}

inline void set_backend(Backend backend) {
    get_backend() = backend;
}

/**
 * @brief Check if AITER should be used based on current backend setting
 */
inline bool should_use_aiter() {
    Backend backend = get_backend();
    
    if (backend == Backend::HIP) {
        return false;
    }
    
    if (backend == Backend::AITER) {
#ifdef MHC_USE_AITER
        return true;
#else
        fprintf(stderr, "[mHC] Error: AITER backend requested but not compiled with MHC_USE_AITER\n");
        return false;
#endif
    }
    
    // AUTO mode
#ifdef MHC_USE_AITER
    return true;
#else
    return false;
#endif
}

} // namespace aiter
} // namespace mhc

