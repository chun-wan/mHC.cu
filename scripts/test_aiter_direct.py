#!/usr/bin/env python3
"""
Direct AITER benchmark test for mHC operations.

This script tests AITER's optimized operators directly without requiring
the full C++ extension compilation.

Usage:
    python scripts/test_aiter_direct.py
"""

import sys
import time
import torch
import torch.nn as nn

# Check environment
print("=" * 60)
print("Environment Check")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
print()

# Try importing AITER
try:
    import aiter
    AITER_AVAILABLE = True
    print("✓ AITER is available")
    
    # Check specific ops
    aiter_ops = {
        'rms_norm': hasattr(aiter, 'rms_norm'),
        'rmsnorm2d_fwd': hasattr(aiter, 'rmsnorm2d_fwd'),
        'layer_norm': hasattr(aiter, 'layer_norm'),
        'sigmoid': hasattr(aiter, 'sigmoid'),
    }
    print(f"  AITER ops: {aiter_ops}")
except ImportError as e:
    AITER_AVAILABLE = False
    print(f"✗ AITER not available: {e}")

print()

# ============================================================================
# PyTorch Reference Implementations
# ============================================================================

def pytorch_rmsnorm(x, weight, eps=1e-6):
    """PyTorch RMSNorm implementation."""
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight, rms.squeeze(-1)


def pytorch_sinkhorn_knopp(inp, iters=3, eps=1e-6):
    """PyTorch Sinkhorn-Knopp implementation."""
    P = inp.clone()
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P


def pytorch_mhc_forward(x, rmsnorm_weight, H_pre, H_post, H_res, sinkhorn_iters=3, eps=1e-6):
    """PyTorch mHC forward pass."""
    B, n, C = x.shape
    
    # 1. Stream aggregation with sigmoid
    H_pre_activated = torch.sigmoid(H_pre)  # [n]
    x_agg = torch.einsum('bnc,n->bc', x, H_pre_activated)  # [B, C]
    
    # 2. RMSNorm
    y_norm, rms = pytorch_rmsnorm(x_agg, rmsnorm_weight, eps)
    
    # 3. Sinkhorn-Knopp for mixing matrix
    M = pytorch_sinkhorn_knopp(torch.exp(H_res), sinkhorn_iters, eps)
    
    # 4. Stream distribution
    H_post_activated = 2.0 * torch.sigmoid(H_post)
    mixed = torch.einsum('ij,bjc->bic', M, x)
    output = mixed + H_post_activated.view(1, n, 1) * y_norm.unsqueeze(1)
    
    return output


# ============================================================================
# AITER-Accelerated Implementations
# ============================================================================

if AITER_AVAILABLE:
    def aiter_rmsnorm(x, weight, eps=1e-6):
        """AITER-accelerated RMSNorm."""
        if hasattr(aiter, 'rmsnorm2d_fwd'):
            # Use AITER's RMSNorm
            out = aiter.rmsnorm2d_fwd(x.contiguous(), weight.contiguous(), eps)
            rms = torch.sqrt((x ** 2).mean(dim=-1) + eps)
            return out, rms
        elif hasattr(aiter, 'rms_norm'):
            out = aiter.rms_norm(x, weight, eps)
            rms = torch.sqrt((x ** 2).mean(dim=-1) + eps)
            return out, rms
        else:
            # Fallback
            return pytorch_rmsnorm(x, weight, eps)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_fn(fn, *args, warmup=10, iters=100):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000 / iters  # ms


def run_benchmarks():
    """Run all benchmarks."""
    device = 'cuda'
    
    configs = [
        (128, 4, 1280),
        (256, 4, 1280),
        (320, 4, 1280),
        (512, 4, 1920),
        (1024, 4, 2560),
    ]
    
    print("=" * 80)
    print("BENCHMARK RESULTS - RMSNorm (bf16)")
    print("=" * 80)
    print(f"{'Config':<25} | {'PyTorch':<12} | {'AITER':<12} | {'Speedup':<10}")
    print("-" * 80)
    
    for B, n, C in configs:
        # Create test tensors in bf16 (AITER requires bf16/fp16)
        x_bf16 = torch.randn(B, C, device=device, dtype=torch.bfloat16)
        x_f32 = x_bf16.float()
        rmsnorm_weight_bf16 = torch.ones(C, device=device, dtype=torch.bfloat16)
        rmsnorm_weight_f32 = rmsnorm_weight_bf16.float()
        
        config_str = f"B={B}, C={C}"
        
        # Benchmark PyTorch (f32)
        def pytorch_rms():
            return pytorch_rmsnorm(x_f32, rmsnorm_weight_f32, 1e-6)
        
        pytorch_time = benchmark_fn(pytorch_rms, warmup=10, iters=100)
        
        # Benchmark AITER RMSNorm (bf16)
        if AITER_AVAILABLE:
            try:
                def aiter_rms():
                    return aiter.rmsnorm2d_fwd(x_bf16, rmsnorm_weight_bf16, 1e-6)
                
                aiter_time = benchmark_fn(aiter_rms, warmup=10, iters=100)
                speedup = pytorch_time / aiter_time if aiter_time > 0 else float('nan')
                aiter_str = f"{aiter_time:.3f}ms"
                speedup_str = f"{speedup:.2f}x"
            except Exception as e:
                aiter_str = f"Error"
                speedup_str = "N/A"
                print(f"  AITER error: {e}")
        else:
            aiter_str = "N/A"
            speedup_str = "N/A"
        
        print(f"{config_str:<25} | {pytorch_time:.3f}ms      | {aiter_str:<12} | {speedup_str:<10}")
    
    print("=" * 80)
    
    # Also benchmark full mHC forward pass
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS - Full mHC Forward (f32)")
    print("=" * 80)
    print(f"{'Config':<25} | {'PyTorch mHC':<12}")
    print("-" * 80)
    
    for B, n, C in configs:
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)
        rmsnorm_weight = torch.ones(C, device=device, dtype=torch.float32)
        H_pre = torch.zeros(n, device=device, dtype=torch.float32)
        H_post = torch.zeros(n, device=device, dtype=torch.float32)
        H_res = torch.eye(n, device=device, dtype=torch.float32)
        
        config_str = f"B={B}, n={n}, C={C}"
        
        pytorch_time = benchmark_fn(
            pytorch_mhc_forward,
            x, rmsnorm_weight, H_pre, H_post, H_res,
            warmup=10, iters=100
        )
        
        print(f"{config_str:<25} | {pytorch_time:.3f}ms")
    
    print("=" * 80)


def test_aiter_ops():
    """Test individual AITER ops."""
    if not AITER_AVAILABLE:
        print("AITER not available, skipping op tests")
        return
    
    print("\n" + "=" * 60)
    print("Testing Individual AITER Ops")
    print("=" * 60)
    
    device = 'cuda'
    B, C = 256, 1280
    
    # AITER requires bf16/fp16 for norm ops
    x_bf16 = torch.randn(B, C, device=device, dtype=torch.bfloat16)
    x_f32 = x_bf16.float()
    weight_bf16 = torch.ones(C, device=device, dtype=torch.bfloat16)
    weight_f32 = weight_bf16.float()
    eps = 1e-6
    
    # Test RMSNorm (bf16)
    print("\n1. RMSNorm Test (bf16):")
    try:
        if hasattr(aiter, 'rmsnorm2d_fwd'):
            out_aiter = aiter.rmsnorm2d_fwd(x_bf16, weight_bf16, eps)
            out_pytorch, _ = pytorch_rmsnorm(x_f32, weight_f32, eps)
            diff = (out_aiter.float() - out_pytorch).abs().max().item()
            print(f"   ✓ rmsnorm2d_fwd works, max diff: {diff:.6e}")
            print(f"     Input shape: {x_bf16.shape}, Output shape: {out_aiter.shape}")
        else:
            print("   ✗ No rmsnorm2d_fwd found in AITER")
    except Exception as e:
        print(f"   ✗ RMSNorm failed: {e}")
    
    # Test LayerNorm (bf16)
    print("\n2. LayerNorm Test (bf16):")
    try:
        if hasattr(aiter, 'layernorm2d_fwd'):
            bias_bf16 = torch.zeros(C, device=device, dtype=torch.bfloat16)
            out_aiter = aiter.layernorm2d_fwd(x_bf16, weight_bf16, bias_bf16, eps)
            # Compare with PyTorch LayerNorm
            ln = nn.LayerNorm(C, eps=eps, device=device, dtype=torch.float32)
            out_pytorch = ln(x_f32)
            diff = (out_aiter.float() - out_pytorch).abs().max().item()
            print(f"   ✓ layernorm2d_fwd works, max diff: {diff:.6e}")
        else:
            print("   ✗ No layernorm2d_fwd found in AITER")
    except Exception as e:
        print(f"   ✗ LayerNorm failed: {e}")
    
    # Test sigmoid (works with f32)
    print("\n3. Sigmoid Test (f32):")
    try:
        if hasattr(aiter, 'sigmoid'):
            out_aiter = aiter.sigmoid(x_f32)
            out_pytorch = torch.sigmoid(x_f32)
            diff = (out_aiter - out_pytorch).abs().max().item()
            print(f"   ✓ sigmoid works, max diff: {diff:.6e}")
        else:
            print("   ✗ No sigmoid op found in AITER")
    except Exception as e:
        print(f"   ✗ Sigmoid failed: {e}")
    
    # Test add op
    print("\n4. Add Test:")
    try:
        if hasattr(aiter, 'add'):
            y_f32 = torch.randn(B, C, device=device, dtype=torch.float32)
            out_aiter = aiter.add(x_f32, y_f32)
            out_pytorch = x_f32 + y_f32
            diff = (out_aiter - out_pytorch).abs().max().item()
            print(f"   ✓ add works, max diff: {diff:.6e}")
        else:
            print("   ✗ No add op found in AITER")
    except Exception as e:
        print(f"   ✗ Add failed: {e}")
    
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available")
        sys.exit(1)
    
    test_aiter_ops()
    run_benchmarks()
    
    print("\nDone!")

