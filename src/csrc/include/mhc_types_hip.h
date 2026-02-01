#pragma once

// AMD ROCm / HIP compatible types for MI300X
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

namespace mhc {

// Use hip_bfloat16 for MI300X
using floatX = hip_bfloat16;
using floatN = float;

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;  // PDL not available on MI300X, kept for API compatibility
};

struct RMSNormParams {
    int n;
    float eps;
};

inline void check_hip(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d: %s\n", file, line, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void check_hipblas(hipblasStatus_t status, const char* file, int line) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "hipBLAS error at %s:%d: %d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_HIP(call) mhc::check_hip((call), __FILE__, __LINE__)
#define CHECK_HIPBLAS(call) mhc::check_hipblas((call), __FILE__, __LINE__)

// Compatibility macros for CUDA code
#define CHECK_CUDA(call) CHECK_HIP(call)
#define CHECK_CUBLAS(call) CHECK_HIPBLAS(call)
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetErrorString hipGetErrorString
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess

} // namespace mhc

