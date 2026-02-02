#!/usr/bin/env python3
"""
mHC HIP 精度驗證腳本
比較 HIP Kernel 與 PyTorch 參考實現的數值精度
"""

import sys
import os

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

import torch
import numpy as np

# Try to import mhc_hip
try:
    import mhc_hip
    HIP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  mhc_hip 模組未載入: {e}")
    print("請確保已設定 LD_LIBRARY_PATH")
    HIP_AVAILABLE = False

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_result(name, max_diff, tolerance, passed):
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  {name}: max_diff={max_diff:.2e}, tol={tolerance:.2e} {status}")

# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def rmsnorm_pytorch(x, weight, eps=1e-6):
    """PyTorch reference RMSNorm."""
    rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype), rms.squeeze(-1)

def sinkhorn_knopp_pytorch(inp, iters=20, eps=1e-6):
    """PyTorch reference Sinkhorn-Knopp."""
    P = inp.clone()
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P

def mhc_forward_pytorch(x, H_pre, H_post, H_res, rmsnorm_weight, sinkhorn_iters=3, eps=1e-6):
    """PyTorch reference mHC forward pass."""
    B, n, C = x.shape
    x_f32 = x.float()
    
    # 1. Stream aggregation
    H_pre_act = torch.sigmoid(H_pre)
    x_agg = torch.einsum('bnc,n->bc', x_f32, H_pre_act)
    
    # 2. RMSNorm
    rms = torch.sqrt((x_agg ** 2).mean(dim=-1, keepdim=True) + eps)
    y_norm = (x_agg / rms) * rmsnorm_weight.float()
    
    # 3. Sinkhorn-Knopp
    M = sinkhorn_knopp_pytorch(torch.exp(H_res), sinkhorn_iters, eps)
    
    # 4. Distribution
    H_post_act = 2.0 * torch.sigmoid(H_post)
    mixed = torch.einsum('ij,bjc->bic', M, x_f32)
    output = mixed + H_post_act.view(1, n, 1) * y_norm.unsqueeze(1)
    
    return output

# ============================================================================
# Verification Tests
# ============================================================================

def verify_rmsnorm():
    """驗證 RMSNorm 精度"""
    print_header("RMSNorm 精度驗證")
    
    if not HIP_AVAILABLE:
        print("  跳過 (mhc_hip 不可用)")
        return True
    
    test_configs = [
        (8, 64),
        (32, 128),
        (128, 512),
        (256, 1280),
        (512, 2048),
    ]
    
    all_passed = True
    tolerance = 1e-3  # bf16 tolerance
    
    for B, C in test_configs:
        # Create inputs
        x = torch.randn(B, C, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(C, device='cuda', dtype=torch.bfloat16)
        
        # PyTorch reference
        out_ref, rms_ref = rmsnorm_pytorch(x, weight, eps=1e-5)
        
        # HIP kernel
        out_hip, rms_hip = mhc_hip.rmsnorm_fwd(x.contiguous(), weight.contiguous(), 1e-5)
        
        # Compare
        max_diff_out = (out_ref.float() - out_hip.float()).abs().max().item()
        max_diff_rms = (rms_ref.float() - rms_hip.float()).abs().max().item()
        
        passed = max_diff_out < tolerance and max_diff_rms < tolerance
        all_passed = all_passed and passed
        
        print_result(f"B={B:4d}, C={C:4d} (out)", max_diff_out, tolerance, max_diff_out < tolerance)
        print_result(f"B={B:4d}, C={C:4d} (rms)", max_diff_rms, tolerance, max_diff_rms < tolerance)
    
    return all_passed

def verify_sinkhorn_knopp():
    """驗證 Sinkhorn-Knopp 精度"""
    print_header("Sinkhorn-Knopp 精度驗證")
    
    if not HIP_AVAILABLE:
        print("  跳過 (mhc_hip 不可用)")
        return True
    
    test_configs = [
        (4, 4, 3),
        (4, 4, 10),
        (8, 8, 10),
        (16, 16, 20),
        (32, 32, 20),
    ]
    
    all_passed = True
    tolerance = 1e-4
    
    for M, N, iters in test_configs:
        # Create positive input (exp of random)
        inp = torch.exp(torch.randn(M, N, device='cuda', dtype=torch.float32))
        
        # PyTorch reference
        out_ref = sinkhorn_knopp_pytorch(inp.clone(), iters, eps=1e-5)
        
        # HIP kernel
        out_hip = mhc_hip.sinkhorn_knopp_fwd(inp.contiguous(), iters, 1e-5)
        
        # Compare
        max_diff = (out_ref - out_hip).abs().max().item()
        
        passed = max_diff < tolerance
        all_passed = all_passed and passed
        
        print_result(f"{M}x{N}, iters={iters:2d}", max_diff, tolerance, passed)
        
        # Verify doubly stochastic property
        row_sums = out_hip.sum(dim=1)
        col_sums = out_hip.sum(dim=0)
        row_err = (row_sums - 1.0).abs().max().item()
        col_err = (col_sums - 1.0).abs().max().item()
        
        ds_passed = row_err < 0.01 and col_err < 0.01
        all_passed = all_passed and ds_passed
        print(f"    雙隨機性: row_err={row_err:.2e}, col_err={col_err:.2e} {'✅' if ds_passed else '❌'}")
    
    return all_passed

def verify_mhc_layer():
    """驗證完整 mHC Layer 精度"""
    print_header("mHC Layer 完整精度驗證")
    
    # Import mhc_aiter
    try:
        from mhc_aiter import MHCLayerSuperFused, MHCLayer
        LAYER_AVAILABLE = True
    except ImportError as e:
        print(f"  跳過 (無法導入 mhc_aiter: {e})")
        return True
    
    test_configs = [
        (16, 4, 512),
        (32, 4, 1280),
        (64, 4, 1920),
        (128, 4, 1280),
    ]
    
    all_passed = True
    tolerance = 5e-2  # Higher tolerance for full layer (bf16 accumulation errors)
    
    for B, n, C in test_configs:
        # Create layer and inputs
        layer = MHCLayerSuperFused(hidden_dim=C, n_streams=n, sinkhorn_iters=3).cuda()
        x = torch.randn(B, n, C, device='cuda', dtype=torch.float32)
        
        # Get HIP output
        with torch.no_grad():
            out_hip = layer(x)
        
        # PyTorch reference
        with torch.no_grad():
            out_ref = mhc_forward_pytorch(
                x, 
                layer.H_pre, 
                layer.H_post, 
                layer.H_res,
                layer.rmsnorm_weight,
                sinkhorn_iters=3,
                eps=1e-6
            )
        
        # Compare
        max_diff = (out_ref - out_hip).abs().max().item()
        mean_diff = (out_ref - out_hip).abs().mean().item()
        
        passed = max_diff < tolerance
        all_passed = all_passed and passed
        
        print_result(f"B={B:3d}, n={n}, C={C:4d}", max_diff, tolerance, passed)
        print(f"    mean_diff={mean_diff:.2e}")
    
    return all_passed

def verify_backward():
    """驗證反向傳播精度"""
    print_header("反向傳播精度驗證")
    
    if not HIP_AVAILABLE:
        print("  跳過 (mhc_hip 不可用)")
        return True
    
    # RMSNorm backward
    print("\n  RMSNorm Backward:")
    
    B, C = 32, 256
    x = torch.randn(B, C, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(C, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    
    # PyTorch reference with autograd
    x_ref = x.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach().requires_grad_(True)
    out_ref, rms_ref = rmsnorm_pytorch(x_ref, weight_ref, eps=1e-5)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    # HIP forward
    out_hip, rms_hip = mhc_hip.rmsnorm_fwd(x.contiguous(), weight.contiguous(), 1e-5)
    
    # HIP backward
    grad_out = torch.ones_like(out_hip)
    d_x, d_weight = mhc_hip.rmsnorm_bwd(
        grad_out.contiguous(), 
        x.contiguous(), 
        weight.contiguous(), 
        rms_hip.contiguous()
    )
    
    # Compare gradients
    tolerance = 5e-2
    dx_diff = (x_ref.grad.float() - d_x.float()).abs().max().item()
    dw_diff = (weight_ref.grad.float() - d_weight.float()).abs().max().item()
    
    dx_passed = dx_diff < tolerance
    dw_passed = dw_diff < tolerance
    
    print_result("d_x gradient", dx_diff, tolerance, dx_passed)
    print_result("d_weight gradient", dw_diff, tolerance, dw_passed)
    
    return dx_passed and dw_passed

def verify_numerical_stability():
    """驗證數值穩定性"""
    print_header("數值穩定性驗證")
    
    if not HIP_AVAILABLE:
        print("  跳過 (mhc_hip 不可用)")
        return True
    
    all_passed = True
    
    # Test with extreme values
    print("\n  極端值測試:")
    
    # Very small values
    x_small = torch.randn(32, 256, device='cuda', dtype=torch.bfloat16) * 1e-4
    weight = torch.ones(256, device='cuda', dtype=torch.bfloat16)
    out, rms = mhc_hip.rmsnorm_fwd(x_small.contiguous(), weight.contiguous(), 1e-5)
    
    has_nan = torch.isnan(out).any().item() or torch.isnan(rms).any().item()
    has_inf = torch.isinf(out).any().item() or torch.isinf(rms).any().item()
    small_passed = not has_nan and not has_inf
    all_passed = all_passed and small_passed
    print(f"    小數值 (1e-4): {'✅ PASSED' if small_passed else '❌ FAILED (NaN/Inf)'}")
    
    # Very large values
    x_large = torch.randn(32, 256, device='cuda', dtype=torch.bfloat16) * 1e3
    out, rms = mhc_hip.rmsnorm_fwd(x_large.contiguous(), weight.contiguous(), 1e-5)
    
    has_nan = torch.isnan(out).any().item() or torch.isnan(rms).any().item()
    has_inf = torch.isinf(out).any().item() or torch.isinf(rms).any().item()
    large_passed = not has_nan and not has_inf
    all_passed = all_passed and large_passed
    print(f"    大數值 (1e3): {'✅ PASSED' if large_passed else '❌ FAILED (NaN/Inf)'}")
    
    # Mixed precision consistency
    print("\n  混合精度一致性:")
    x_f32 = torch.randn(32, 256, device='cuda', dtype=torch.float32)
    x_bf16 = x_f32.to(torch.bfloat16)
    weight_f32 = torch.ones(256, device='cuda', dtype=torch.float32)
    weight_bf16 = weight_f32.to(torch.bfloat16)
    
    out_bf16, _ = mhc_hip.rmsnorm_fwd(x_bf16.contiguous(), weight_bf16.contiguous(), 1e-5)
    out_ref, _ = rmsnorm_pytorch(x_bf16, weight_bf16, eps=1e-5)
    
    diff = (out_bf16.float() - out_ref.float()).abs().max().item()
    precision_passed = diff < 1e-3
    all_passed = all_passed and precision_passed
    print(f"    BF16 一致性: max_diff={diff:.2e} {'✅ PASSED' if precision_passed else '❌ FAILED'}")
    
    return all_passed

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print(" mHC HIP 精度驗證")
    print("=" * 60)
    print(f"\n設備: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP Kernel: {'可用' if HIP_AVAILABLE else '不可用'}")
    
    results = {}
    
    # Run all verifications
    results['RMSNorm'] = verify_rmsnorm()
    results['Sinkhorn-Knopp'] = verify_sinkhorn_knopp()
    results['mHC Layer'] = verify_mhc_layer()
    results['Backward'] = verify_backward()
    results['Numerical Stability'] = verify_numerical_stability()
    
    # Summary
    print_header("驗證結果總結")
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print(" 🎉 所有精度驗證通過！")
    else:
        print(" ⚠️  部分驗證失敗，請檢查上述結果")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

