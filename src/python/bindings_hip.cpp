// HIP/ROCm compatible Python bindings for mHC kernels on AMD MI300X

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include "../csrc/include/mhc_types_hip.h"
#include "../csrc/kernels/rmsnorm_hip.h"
#include "../csrc/kernels/sinkhorn_knopp_hip.h"
#include "../csrc/kernels/stream_ops_hip.h"
#include "../csrc/kernels/fused_rmsnorm_matmul_hip.h"
#include "../csrc/include/utils_hip.h"

using namespace mhc;

#define CHECK_TENSOR_HIP(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a HIP tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_TENSOR_HIP(x); \
    CHECK_CONTIGUOUS(x)

// ============================================================================
// hipBLASLt Cache for efficient matmul
// ============================================================================

struct HipblasLtCache {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t matmul_desc = nullptr;
    hipblasLtMatrixLayout_t A_desc = nullptr;
    hipblasLtMatrixLayout_t B_concat_desc = nullptr;
    hipblasLtMatrixLayout_t C_concat_desc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulHeuristicResult_t heuristic_concat;
    void* workspace = nullptr;
    size_t workspace_size = 4 * 1024 * 1024;
    int cached_B = 0, cached_n = 0, cached_nC = 0;
    bool initialized = false;

    void init(int B, int n, int nC) {
        if (initialized && cached_B == B && cached_n == n && cached_nC == nC) {
            return;
        }
        destroy();

        cached_B = B;
        cached_n = n;
        cached_nC = nC;

        int out_dim = n + n + n * n;

        CHECK_HIPBLAS(hipblasLtCreate(&handle));
        CHECK_HIP(hipMalloc(&workspace, workspace_size));

        CHECK_HIPBLAS(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t trans_a = HIPBLAS_OP_N;
        hipblasOperation_t trans_b = HIPBLAS_OP_T;
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                      &trans_a, sizeof(trans_a)));
        CHECK_HIPBLAS(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                      &trans_b, sizeof(trans_b)));

        hipblasLtOrder_t row_order = HIPBLASLT_ORDER_ROW;

        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&A_desc, HIP_R_16BF, B, nC, nC));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(A_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));

        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&B_concat_desc, HIP_R_16BF, out_dim, nC, nC));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(B_concat_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutCreate(&C_concat_desc, HIP_R_32F, B, out_dim, out_dim));
        CHECK_HIPBLAS(hipblasLtMatrixLayoutSetAttribute(C_concat_desc, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                        &row_order, sizeof(row_order)));

        CHECK_HIPBLAS(hipblasLtMatmulPreferenceCreate(&pref));
        CHECK_HIPBLAS(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                            &workspace_size, sizeof(workspace_size)));

        int returned = 0;
        CHECK_HIPBLAS(hipblasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_concat_desc,
                                                      C_concat_desc, C_concat_desc, pref, 1,
                                                      &heuristic_concat, &returned));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (A_desc) hipblasLtMatrixLayoutDestroy(A_desc);
        if (B_concat_desc) hipblasLtMatrixLayoutDestroy(B_concat_desc);
        if (C_concat_desc) hipblasLtMatrixLayoutDestroy(C_concat_desc);
        if (matmul_desc) hipblasLtMatmulDescDestroy(matmul_desc);
        if (handle) hipblasLtDestroy(handle);
        if (workspace) hipFree(workspace);
        pref = nullptr;
        A_desc = nullptr;
        B_concat_desc = nullptr;
        C_concat_desc = nullptr;
        matmul_desc = nullptr;
        handle = nullptr;
        workspace = nullptr;
        initialized = false;
    }

    ~HipblasLtCache() { destroy(); }
};

static HipblasLtCache g_hipblas_cache;

// ============================================================================
// MHC Workspace
// ============================================================================

struct MHCWorkspace {
    void* buffer = nullptr;
    size_t buffer_size = 0;
    int cached_B = 0, cached_n = 0, cached_C = 0;
    size_t off_x_agg = 0, off_y_norm = 0, off_rms = 0, off_M = 0, off_H_pre = 0, off_H_post = 0;

    hipStream_t sinkhorn_stream = nullptr;
    hipEvent_t sinkhorn_done = nullptr;
    bool streams_initialized = false;

    void ensure_streams() {
        if (streams_initialized)
            return;
        CHECK_HIP(hipStreamCreate(&sinkhorn_stream));
        CHECK_HIP(hipEventCreate(&sinkhorn_done));
        streams_initialized = true;
    }

    void ensure_size(int B, int n, int C) {
        ensure_streams();
        if (cached_B >= B && cached_n == n && cached_C == C && buffer)
            return;
        if (buffer)
            hipFree(buffer);
        cached_B = B;
        cached_n = n;
        cached_C = C;
        auto align = [](size_t s) { return ((s + 255) / 256) * 256; };
        off_x_agg = 0;
        off_y_norm = align(B * C * 2);
        off_rms = off_y_norm + align(B * C * 2);
        off_M = off_rms + align(B * 4);
        off_H_pre = off_M + align(n * n * 4);
        off_H_post = off_H_pre + align(n * 4);
        buffer_size = off_H_post + align(n * 4);
        CHECK_HIP(hipMalloc(&buffer, buffer_size));
    }

    floatX* x_agg() { return (floatX*)((char*)buffer + off_x_agg); }
    floatX* y_norm() { return (floatX*)((char*)buffer + off_y_norm); }
    float* rms() { return (float*)((char*)buffer + off_rms); }
    float* M() { return (float*)((char*)buffer + off_M); }
    float* H_pre() { return (float*)((char*)buffer + off_H_pre); }
    float* H_post() { return (float*)((char*)buffer + off_H_post); }

    ~MHCWorkspace() {
        if (buffer) hipFree(buffer);
        if (sinkhorn_stream) hipStreamDestroy(sinkhorn_stream);
        if (sinkhorn_done) hipEventDestroy(sinkhorn_done);
    }
};

static MHCWorkspace g_ws;

// ============================================================================
// Sinkhorn-Knopp Functions
// ============================================================================

torch::Tensor sinkhorn_knopp_fwd(torch::Tensor inp, int iters, float eps) {
    CHECK_INPUT(inp);
    auto out = torch::empty_like(inp);
    sinkhorn_knopp_forward(out.data_ptr<float>(), inp.data_ptr<float>(), inp.size(0), inp.size(1),
                           iters, eps);
    return out;
}

torch::Tensor sinkhorn_knopp_bwd(torch::Tensor grad, torch::Tensor out, torch::Tensor inp,
                                 int iters, float eps) {
    CHECK_INPUT(grad);
    CHECK_INPUT(out);
    CHECK_INPUT(inp);
    auto d_inp = torch::empty_like(inp);
    sinkhorn_knopp_backward(d_inp.data_ptr<float>(), grad.data_ptr<float>(), out.data_ptr<float>(),
                            inp.data_ptr<float>(), inp.size(0), iters, eps);
    return d_inp;
}

// ============================================================================
// RMSNorm Functions
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_fwd(torch::Tensor inp, torch::Tensor weight,
                                                     float eps) {
    CHECK_INPUT(inp);
    CHECK_INPUT(weight);
    int B = inp.size(0), C = inp.size(1);
    auto out = torch::empty_like(inp);
    auto rms = torch::empty({B}, inp.options().dtype(torch::kFloat32));
    rmsnorm_forward_with_rms(
        reinterpret_cast<floatX*>(out.data_ptr<at::BFloat16>()), rms.data_ptr<float>(),
        reinterpret_cast<const floatX*>(inp.data_ptr<at::BFloat16>()),
        reinterpret_cast<const floatX*>(weight.data_ptr<at::BFloat16>()), B, C, eps);
    return {out, rms};
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_bwd(torch::Tensor grad, torch::Tensor inp,
                                                     torch::Tensor weight, torch::Tensor rms) {
    CHECK_INPUT(grad);
    CHECK_INPUT(inp);
    CHECK_INPUT(weight);
    CHECK_INPUT(rms);
    int B = inp.size(0), C = inp.size(1);
    auto grad_f32 = grad.to(torch::kFloat32).contiguous();
    auto d_inp_f32 = torch::empty({B, C}, inp.options().dtype(torch::kFloat32));
    auto d_weight = torch::zeros({C}, inp.options().dtype(torch::kFloat32));
    rmsnorm_backward(d_inp_f32.data_ptr<float>(), d_weight.data_ptr<float>(),
                     grad_f32.data_ptr<float>(),
                     reinterpret_cast<const floatX*>(inp.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const floatX*>(weight.data_ptr<at::BFloat16>()),
                     rms.data_ptr<float>(), B, C);
    return {d_inp_f32.to(torch::kBFloat16), d_weight};
}

// ============================================================================
// MHC Layer Forward (Static H)
// ============================================================================

torch::Tensor mhc_layer_fwd_inference(torch::Tensor x_expanded, torch::Tensor rmsnorm_weight,
                                      torch::Tensor H_pre, torch::Tensor H_post,
                                      torch::Tensor H_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(x_expanded);
    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    g_ws.ensure_size(B, n, C);
    auto output = torch::empty({B, n, C}, x_expanded.options().dtype(torch::kFloat32));

    const float* x_ptr = x_expanded.scalar_type() == torch::kFloat32
                             ? x_expanded.data_ptr<float>()
                             : (x_expanded = x_expanded.to(torch::kFloat32)).data_ptr<float>();

    bool use_pipeline = (n >= 16);

    if (use_pipeline) {
        sinkhorn_knopp_forward_fused_exp(g_ws.M(), nullptr, H_res.data_ptr<float>(), n, n,
                                         sinkhorn_iters, eps, g_ws.sinkhorn_stream);
        CHECK_HIP(hipEventRecord(g_ws.sinkhorn_done, g_ws.sinkhorn_stream));
    }

    stream_aggregate_bf16_fused_sigmoid(g_ws.x_agg(), g_ws.H_pre(), x_ptr, H_pre.data_ptr<float>(),
                                        B, n, C, stream);
    rmsnorm_forward_with_rms(
        g_ws.y_norm(), g_ws.rms(), g_ws.x_agg(),
        reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()), B, C, eps,
        stream);

    if (use_pipeline) {
        CHECK_HIP(hipStreamWaitEvent(stream, g_ws.sinkhorn_done, 0));
    } else {
        sinkhorn_knopp_forward_fused_exp(g_ws.M(), nullptr, H_res.data_ptr<float>(), n, n,
                                         sinkhorn_iters, eps, stream);
    }

    stream_distribute_mix_add_fused(output.data_ptr<float>(), g_ws.H_post(), x_ptr, g_ws.y_norm(),
                                    H_post.data_ptr<float>(), g_ws.M(), B, n, C, stream);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
mhc_layer_fwd(torch::Tensor x_expanded, torch::Tensor rmsnorm_weight, torch::Tensor H_pre,
              torch::Tensor H_post, torch::Tensor H_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(H_pre);
    CHECK_INPUT(H_post);
    CHECK_INPUT(H_res);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    g_ws.ensure_streams();

    auto opts = x_expanded.options();
    auto output = torch::empty({B, n, C}, opts.dtype(torch::kFloat32));
    auto x_agg_bf16 = torch::empty({B, C}, opts.dtype(torch::kBFloat16));
    auto rms = torch::empty({B}, opts.dtype(torch::kFloat32));
    auto y_norm_bf16 = torch::empty({B, C}, opts.dtype(torch::kBFloat16));
    auto M = torch::empty({n, n}, opts.dtype(torch::kFloat32));
    auto H_pre_activated = torch::empty({n}, opts.dtype(torch::kFloat32));
    auto H_post_activated = torch::empty({n}, opts.dtype(torch::kFloat32));

    const float* x_ptr;
    torch::Tensor x_holder;
    if (x_expanded.scalar_type() == torch::kFloat32) {
        x_ptr = x_expanded.data_ptr<float>();
    } else {
        x_holder = x_expanded.to(torch::kFloat32);
        x_ptr = x_holder.data_ptr<float>();
    }

    bool use_pipeline = (n >= 16);

    if (use_pipeline) {
        sinkhorn_knopp_forward_fused_exp(M.data_ptr<float>(), nullptr, H_res.data_ptr<float>(), n,
                                         n, sinkhorn_iters, eps, g_ws.sinkhorn_stream);
        CHECK_HIP(hipEventRecord(g_ws.sinkhorn_done, g_ws.sinkhorn_stream));
    }

    stream_aggregate_bf16_fused_sigmoid(
        reinterpret_cast<floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
        H_pre_activated.data_ptr<float>(), x_ptr, H_pre.data_ptr<float>(), B, n, C, stream);

    rmsnorm_forward_with_rms(
        reinterpret_cast<floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()), rms.data_ptr<float>(),
        reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()), B, C, eps,
        stream);

    if (use_pipeline) {
        CHECK_HIP(hipStreamWaitEvent(stream, g_ws.sinkhorn_done, 0));
    } else {
        sinkhorn_knopp_forward_fused_exp(M.data_ptr<float>(), nullptr, H_res.data_ptr<float>(), n,
                                         n, sinkhorn_iters, eps, stream);
    }

    stream_distribute_mix_add_fused(
        output.data_ptr<float>(), H_post_activated.data_ptr<float>(), x_ptr,
        reinterpret_cast<const floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()),
        H_post.data_ptr<float>(), M.data_ptr<float>(), B, n, C, stream);

    return {output, rms, x_agg_bf16, H_pre_activated, H_post_activated, M, y_norm_bf16};
}

// ============================================================================
// MHC Layer Backward (Static H)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mhc_layer_bwd(torch::Tensor grad_output, torch::Tensor x_expanded, torch::Tensor rmsnorm_weight,
              torch::Tensor rms, torch::Tensor x_agg_bf16, torch::Tensor H_pre_activated,
              torch::Tensor H_post_activated, torch::Tensor M, torch::Tensor y_norm_bf16,
              torch::Tensor H_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(rms);
    CHECK_INPUT(x_agg_bf16);
    CHECK_INPUT(H_pre_activated);
    CHECK_INPUT(H_post_activated);
    CHECK_INPUT(M);
    CHECK_INPUT(y_norm_bf16);
    CHECK_INPUT(H_res);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK_SIZE = 256;
    int workspace_num_blocks = std::min(128, (B * C + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto workspace_dH = torch::empty({workspace_num_blocks * n}, grad_output.options().dtype(torch::kFloat32));
    auto workspace_dM = torch::empty({workspace_num_blocks * n * n}, grad_output.options().dtype(torch::kFloat32));

    auto grad_f32 = grad_output.to(torch::kFloat32).contiguous();
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto y_norm_f32 = y_norm_bf16.to(torch::kFloat32).contiguous();

    auto d_x_mix = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_y_norm = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_M = torch::empty({n, n}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_post_activated = torch::empty({n}, grad_output.options().dtype(torch::kFloat32));

    stream_distribute_mix_backward_fused(
        d_x_mix.data_ptr<float>(), d_y_norm.data_ptr<float>(), d_M.data_ptr<float>(),
        d_H_post_activated.data_ptr<float>(), grad_f32.data_ptr<float>(), x_f32.data_ptr<float>(),
        y_norm_f32.data_ptr<float>(), M.data_ptr<float>(), H_post_activated.data_ptr<float>(), B, n,
        C, workspace_dM.data_ptr<float>(), workspace_dH.data_ptr<float>(), workspace_num_blocks,
        stream);

    auto d_H_post = d_H_post_activated * H_post_activated * (1.0f - H_post_activated / 2.0f);

    auto H_res_exp = torch::exp(H_res);
    auto d_H_res_exp = torch::empty({n, n}, grad_output.options().dtype(torch::kFloat32));
    sinkhorn_knopp_backward(d_H_res_exp.data_ptr<float>(), d_M.data_ptr<float>(),
                            M.data_ptr<float>(), H_res_exp.data_ptr<float>(), n, sinkhorn_iters,
                            eps, stream);
    auto d_H_res = d_H_res_exp * H_res_exp;

    auto d_x_agg = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_rmsnorm_weight = torch::zeros({C}, grad_output.options().dtype(torch::kFloat32));
    rmsnorm_backward(d_x_agg.data_ptr<float>(), d_rmsnorm_weight.data_ptr<float>(),
                     d_y_norm.data_ptr<float>(),
                     reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()),
                     rms.data_ptr<float>(), B, C, stream);

    auto d_x_from_agg = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_pre_activated = torch::empty({n}, grad_output.options().dtype(torch::kFloat32));
    stream_aggregate_backward(d_x_from_agg.data_ptr<float>(), d_H_pre_activated.data_ptr<float>(),
                              d_x_agg.data_ptr<float>(), x_f32.data_ptr<float>(),
                              H_pre_activated.data_ptr<float>(), B, n, C,
                              workspace_dH.data_ptr<float>(), workspace_num_blocks, stream);

    auto d_H_pre = d_H_pre_activated * H_pre_activated * (1.0f - H_pre_activated);

    auto d_x_expanded = d_x_mix + d_x_from_agg;

    return {d_x_expanded, d_rmsnorm_weight, d_H_pre, d_H_post, d_H_res};
}

// ============================================================================
// MHC Layer Dynamic H Functions
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
mhc_layer_fwd_dynamic(torch::Tensor x_expanded, torch::Tensor rmsnorm_weight,
                      torch::Tensor phi_concat_bf16, float alpha_pre, float alpha_post,
                      float alpha_res, torch::Tensor b_pre, torch::Tensor b_post,
                      torch::Tensor b_res, int sinkhorn_iters, float eps) {
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(phi_concat_bf16);
    CHECK_INPUT(b_pre);
    CHECK_INPUT(b_post);
    CHECK_INPUT(b_res);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    int nC = n * C;
    int out_dim = n + n + n * n;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto x_flat = x_f32.reshape({B, nC});
    auto x_flat_bf16 = x_flat.to(torch::kBFloat16).contiguous();

    auto rms_h = torch::empty({B}, x_expanded.options().dtype(torch::kFloat32));
    compute_rms(rms_h.data_ptr<float>(),
                reinterpret_cast<const floatX*>(x_flat_bf16.data_ptr<at::BFloat16>()), B, nC, eps,
                stream);

    auto H_proj_concat = torch::empty({B, out_dim}, x_expanded.options().dtype(torch::kFloat32));

    g_hipblas_cache.init(B, n, nC);

    float alpha = 1.0f, beta = 0.0f;

    CHECK_HIPBLAS(hipblasLtMatmul(g_hipblas_cache.handle, g_hipblas_cache.matmul_desc, &alpha,
                                  x_flat_bf16.data_ptr<at::BFloat16>(), g_hipblas_cache.A_desc,
                                  phi_concat_bf16.data_ptr<at::BFloat16>(),
                                  g_hipblas_cache.B_concat_desc, &beta,
                                  H_proj_concat.data_ptr<float>(), g_hipblas_cache.C_concat_desc,
                                  H_proj_concat.data_ptr<float>(), g_hipblas_cache.C_concat_desc,
                                  &g_hipblas_cache.heuristic_concat.algo, g_hipblas_cache.workspace,
                                  g_hipblas_cache.workspace_size, stream));

    auto H_pre_activated = torch::empty({B, n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_post_activated = torch::empty({B, n}, x_expanded.options().dtype(torch::kFloat32));
    auto H_res_exp = torch::empty({B, n, n}, x_expanded.options().dtype(torch::kFloat32));

    fused_h_activations(H_pre_activated.data_ptr<float>(), H_post_activated.data_ptr<float>(),
                        H_res_exp.data_ptr<float>(), H_proj_concat.data_ptr<float>(),
                        rms_h.data_ptr<float>(), alpha_pre, alpha_post, alpha_res,
                        b_pre.data_ptr<float>(), b_post.data_ptr<float>(), b_res.data_ptr<float>(),
                        B, n, stream);

    auto M = torch::empty({B, n, n}, x_expanded.options().dtype(torch::kFloat32));
    sinkhorn_knopp_forward_batched(M.data_ptr<float>(), H_res_exp.data_ptr<float>(), B, n,
                                   sinkhorn_iters, eps, stream);

    auto x_agg_bf16 = torch::empty({B, C}, x_expanded.options().dtype(torch::kBFloat16));
    auto rms = torch::empty({B}, x_expanded.options().dtype(torch::kFloat32));
    auto y_norm_bf16 = torch::empty({B, C}, x_expanded.options().dtype(torch::kBFloat16));
    auto output = torch::empty({B, n, C}, x_expanded.options().dtype(torch::kFloat32));

    stream_aggregate_bf16_dynamic(reinterpret_cast<floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
                                  x_f32.data_ptr<float>(), H_pre_activated.data_ptr<float>(), B, n,
                                  C, stream);

    rmsnorm_forward_with_rms(
        reinterpret_cast<floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()), rms.data_ptr<float>(),
        reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()), B, C, eps,
        stream);

    stream_distribute_mix_add_fused_dynamic(
        output.data_ptr<float>(), x_f32.data_ptr<float>(),
        reinterpret_cast<const floatX*>(y_norm_bf16.data_ptr<at::BFloat16>()),
        H_post_activated.data_ptr<float>(), M.data_ptr<float>(), B, n, C, stream);

    return {output, rms, x_agg_bf16, H_pre_activated, H_post_activated, M,
            y_norm_bf16, x_flat_bf16, rms_h};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mhc_layer_bwd_dynamic(torch::Tensor grad_output, torch::Tensor x_expanded,
                      torch::Tensor rmsnorm_weight, torch::Tensor rms, torch::Tensor x_agg_bf16,
                      torch::Tensor H_pre_activated, torch::Tensor H_post_activated,
                      torch::Tensor M, torch::Tensor y_norm_bf16, float eps) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(x_expanded);
    CHECK_INPUT(rmsnorm_weight);
    CHECK_INPUT(rms);
    CHECK_INPUT(x_agg_bf16);
    CHECK_INPUT(H_pre_activated);
    CHECK_INPUT(H_post_activated);
    CHECK_INPUT(M);
    CHECK_INPUT(y_norm_bf16);

    int B = x_expanded.size(0), n = x_expanded.size(1), C = x_expanded.size(2);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK_SIZE = 256;
    int workspace_num_blocks = std::min(128, (B * C + BLOCK_SIZE - 1) / BLOCK_SIZE);
    auto workspace_dH = torch::empty({workspace_num_blocks * n}, grad_output.options().dtype(torch::kFloat32));
    auto workspace_dM = torch::empty({workspace_num_blocks * n * n}, grad_output.options().dtype(torch::kFloat32));

    auto grad_f32 = grad_output.to(torch::kFloat32).contiguous();
    auto x_f32 = x_expanded.to(torch::kFloat32).contiguous();
    auto y_norm_f32 = y_norm_bf16.to(torch::kFloat32).contiguous();

    auto d_x_mix = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_y_norm = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_M = torch::empty({B, n, n}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_post = torch::empty({B, n}, grad_output.options().dtype(torch::kFloat32));

    stream_distribute_mix_backward_fused(
        d_x_mix.data_ptr<float>(), d_y_norm.data_ptr<float>(), d_M.data_ptr<float>(),
        d_H_post.data_ptr<float>(), grad_f32.data_ptr<float>(), x_f32.data_ptr<float>(),
        y_norm_f32.data_ptr<float>(), M.data_ptr<float>(), H_post_activated.data_ptr<float>(), B, n,
        C, workspace_dM.data_ptr<float>(), workspace_dH.data_ptr<float>(), workspace_num_blocks,
        stream);

    auto d_x_agg = torch::empty({B, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_rmsnorm_weight = torch::zeros({C}, grad_output.options().dtype(torch::kFloat32));
    rmsnorm_backward(d_x_agg.data_ptr<float>(), d_rmsnorm_weight.data_ptr<float>(),
                     d_y_norm.data_ptr<float>(),
                     reinterpret_cast<const floatX*>(x_agg_bf16.data_ptr<at::BFloat16>()),
                     reinterpret_cast<const floatX*>(rmsnorm_weight.data_ptr<at::BFloat16>()),
                     rms.data_ptr<float>(), B, C, stream);

    auto d_x_from_agg = torch::empty({B, n, C}, grad_output.options().dtype(torch::kFloat32));
    auto d_H_pre = torch::empty({B, n}, grad_output.options().dtype(torch::kFloat32));
    stream_aggregate_backward(d_x_from_agg.data_ptr<float>(), d_H_pre.data_ptr<float>(),
                              d_x_agg.data_ptr<float>(), x_f32.data_ptr<float>(),
                              H_pre_activated.data_ptr<float>(), B, n, C,
                              workspace_dH.data_ptr<float>(), workspace_num_blocks, stream);

    auto d_x_expanded = d_x_mix + d_x_from_agg;

    return {d_x_expanded, d_rmsnorm_weight, d_H_pre, d_H_post, d_M};
}

// ============================================================================
// Python Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sinkhorn_knopp_fwd", &sinkhorn_knopp_fwd, "Sinkhorn-Knopp forward (HIP)");
    m.def("sinkhorn_knopp_bwd", &sinkhorn_knopp_bwd, "Sinkhorn-Knopp backward (HIP)");
    m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm forward (HIP)");
    m.def("rmsnorm_bwd", &rmsnorm_bwd, "RMSNorm backward (HIP)");
    m.def("mhc_layer_fwd", &mhc_layer_fwd, "MHC Layer forward (HIP)");
    m.def("mhc_layer_fwd_inference", &mhc_layer_fwd_inference, "MHC Layer inference forward (HIP)");
    m.def("mhc_layer_bwd", &mhc_layer_bwd, "MHC Layer backward (HIP)");
    m.def("mhc_layer_fwd_dynamic", &mhc_layer_fwd_dynamic, "MHC Layer dynamic H forward (HIP)");
    m.def("mhc_layer_bwd_dynamic", &mhc_layer_bwd_dynamic, "MHC Layer dynamic H backward (HIP)");
}

