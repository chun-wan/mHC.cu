#!/usr/bin/env python3
"""
Fused Operations Benchmark for mHC.cu

This script implements and benchmarks fused operations:
1. Fused Stream Aggregation + RMSNorm
2. Fused Sinkhorn-Knopp (optimized iterations)
3. Fused Stream Distribution + Mix + Add
4. Full Fused mHC Layer

Compares against non-fused PyTorch and AITER implementations.
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Check environment
print("=" * 70)
print("mHC.cu Fused Operations Benchmark")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Check AITER
try:
    import aiter
    AITER_AVAILABLE = True
    print("✓ AITER available")
except ImportError:
    AITER_AVAILABLE = False
    print("✗ AITER not available")

# Check Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    print("✓ Triton available")
except ImportError:
    TRITON_AVAILABLE = False
    print("✗ Triton not available")

print()

# ============================================================================
# Non-Fused Reference Implementations
# ============================================================================

def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard PyTorch RMSNorm."""
    rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)


def sinkhorn_knopp_pytorch(inp: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """Standard Sinkhorn-Knopp."""
    P = inp.clone()
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P


class MHCLayerNonFused(nn.Module):
    """Non-fused mHC Layer (baseline)."""
    
    def __init__(self, C: int, n: int = 4, sinkhorn_iters: int = 3, eps: float = 1e-6):
        super().__init__()
        self.C = C
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        
        self.rmsnorm_weight = nn.Parameter(torch.ones(C, dtype=torch.bfloat16))
        self.H_pre = nn.Parameter(torch.zeros(n))
        self.H_post = nn.Parameter(torch.zeros(n))
        self.H_res = nn.Parameter(torch.eye(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, C = x.shape
        x_f32 = x.float()
        
        # Step 1: Stream aggregation
        H_pre_act = torch.sigmoid(self.H_pre)
        x_agg = torch.einsum('bnc,n->bc', x_f32, H_pre_act)
        
        # Step 2: RMSNorm
        y_norm = rmsnorm_pytorch(x_agg.to(torch.bfloat16), self.rmsnorm_weight, self.eps)
        
        # Step 3: Sinkhorn-Knopp
        M = sinkhorn_knopp_pytorch(torch.exp(self.H_res), self.sinkhorn_iters, self.eps)
        
        # Step 4: Stream distribution
        H_post_act = 2.0 * torch.sigmoid(self.H_post)
        mixed = torch.einsum('ij,bjc->bic', M, x_f32)
        output = mixed + H_post_act.view(1, n, 1) * y_norm.float().unsqueeze(1)
        
        return output


# ============================================================================
# Fused Operations - Triton Kernels
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def fused_agg_rmsnorm_kernel(
        x_ptr, weight_ptr, out_ptr, H_pre_ptr,
        B, n, C,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused stream aggregation + RMSNorm kernel."""
        # Get batch index
        batch_idx = tl.program_id(0)
        
        # Load H_pre and compute sigmoid
        h_pre_offsets = tl.arange(0, 32)  # Assume n <= 32
        h_pre_mask = h_pre_offsets < n
        h_pre = tl.load(H_pre_ptr + h_pre_offsets, mask=h_pre_mask, other=0.0)
        h_pre_sigmoid = 1.0 / (1.0 + tl.exp(-h_pre))
        
        # Process in blocks
        acc_sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        agg_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for c_start in range(0, C, BLOCK_SIZE):
            c_offsets = c_start + tl.arange(0, BLOCK_SIZE)
            c_mask = c_offsets < C
            
            # Aggregate across streams
            agg = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for i in range(n):
                x_offset = batch_idx * n * C + i * C + c_offsets
                x_val = tl.load(x_ptr + x_offset, mask=c_mask, other=0.0)
                h_val = tl.load(H_pre_ptr + i)
                h_sigmoid = 1.0 / (1.0 + tl.exp(-h_val))
                agg += x_val * h_sigmoid
            
            # Accumulate for RMS
            acc_sum_sq += agg * agg
            
            # Store aggregated values temporarily
            if c_start == 0:
                agg_vals = agg
        
        # Compute RMS (simplified - full version would need reduction)
        # This is a simplified version for demonstration
        
    @triton.jit  
    def fused_sinkhorn_kernel(
        inp_ptr, out_ptr,
        n, iters, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused Sinkhorn-Knopp kernel with all iterations in one kernel."""
        row = tl.program_id(0)
        
        # Load row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < n
        
        P = tl.load(inp_ptr + row * n + col_offsets, mask=col_mask, other=0.0)
        
        # Sinkhorn iterations
        for _ in range(iters):
            # Row normalization
            row_sum = tl.sum(P, axis=0) + eps
            P = P / row_sum
            
            # Column normalization (simplified - actual impl needs sync)
            # This is a simplified version
            
        tl.store(out_ptr + row * n + col_offsets, P, mask=col_mask)


# ============================================================================
# Fused Operations - PyTorch JIT
# ============================================================================

@torch.jit.script
def fused_stream_agg_rmsnorm_jit(
    x: torch.Tensor,
    H_pre: torch.Tensor, 
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused stream aggregation + RMSNorm using JIT.
    
    Combines:
    1. H_pre_act = sigmoid(H_pre)
    2. x_agg = einsum('bnc,n->bc', x, H_pre_act)  
    3. y_norm = rmsnorm(x_agg, weight)
    """
    B, n, C = x.shape
    
    # Fused sigmoid + weighted sum
    H_pre_act = torch.sigmoid(H_pre)
    
    # Optimized aggregation using matmul instead of einsum
    # x: [B, n, C] -> [B, C, n] for matmul
    x_t = x.transpose(1, 2)  # [B, C, n]
    x_agg = torch.matmul(x_t, H_pre_act.unsqueeze(-1)).squeeze(-1)  # [B, C]
    
    # Fused RMSNorm
    x_agg_f32 = x_agg.float()
    rms = torch.sqrt((x_agg_f32 ** 2).mean(dim=-1, keepdim=True) + eps)
    y_norm = (x_agg_f32 / rms) * weight.float()
    
    return y_norm.to(x.dtype)


@torch.jit.script
def fused_sinkhorn_jit(inp: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    Optimized Sinkhorn-Knopp using JIT.
    
    Uses in-place operations to reduce memory traffic.
    """
    P = inp.clone()
    
    for _ in range(iters):
        # Row normalization (in-place)
        row_sum = P.sum(dim=-1, keepdim=True) + eps
        P = P / row_sum
        
        # Column normalization (in-place)
        col_sum = P.sum(dim=-2, keepdim=True) + eps
        P = P / col_sum
    
    return P


@torch.jit.script
def fused_distribute_mix_add_jit(
    x: torch.Tensor,
    y_norm: torch.Tensor,
    H_post: torch.Tensor,
    M: torch.Tensor
) -> torch.Tensor:
    """
    Fused stream distribution + mixing + addition.
    
    Combines:
    1. H_post_act = 2 * sigmoid(H_post)
    2. mixed = einsum('ij,bjc->bic', M, x)
    3. output = mixed + H_post_act * y_norm
    """
    B, n, C = x.shape
    
    # Fused sigmoid
    H_post_act = 2.0 * torch.sigmoid(H_post)
    
    # Optimized mixing using bmm
    # M: [n, n], x: [B, n, C]
    # mixed[b,i,c] = sum_j M[i,j] * x[b,j,c]
    M_expanded = M.unsqueeze(0).expand(B, -1, -1)  # [B, n, n]
    mixed = torch.bmm(M_expanded, x)  # [B, n, C]
    
    # Fused add
    output = mixed + H_post_act.view(1, n, 1) * y_norm.unsqueeze(1)
    
    return output


# ============================================================================
# Fully Fused mHC Layer
# ============================================================================

class MHCLayerFused(nn.Module):
    """Fully fused mHC Layer using JIT-compiled operations."""
    
    def __init__(self, C: int, n: int = 4, sinkhorn_iters: int = 3, eps: float = 1e-6):
        super().__init__()
        self.C = C
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        
        self.rmsnorm_weight = nn.Parameter(torch.ones(C, dtype=torch.bfloat16))
        self.H_pre = nn.Parameter(torch.zeros(n))
        self.H_post = nn.Parameter(torch.zeros(n))
        self.H_res = nn.Parameter(torch.eye(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, C = x.shape
        
        # Fused aggregation + RMSNorm
        y_norm = fused_stream_agg_rmsnorm_jit(
            x.float(), self.H_pre, self.rmsnorm_weight.float(), self.eps
        )
        
        # Optimized Sinkhorn-Knopp
        M = fused_sinkhorn_jit(torch.exp(self.H_res), self.sinkhorn_iters, self.eps)
        
        # Fused distribution + mix + add
        output = fused_distribute_mix_add_jit(x.float(), y_norm, self.H_post, M)
        
        return output


# ============================================================================
# Super Fused - Single Forward Pass
# ============================================================================

@torch.jit.script
def mhc_forward_superfused(
    x: torch.Tensor,
    H_pre: torch.Tensor,
    H_post: torch.Tensor,
    H_res: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    sinkhorn_iters: int,
    eps: float
) -> torch.Tensor:
    """
    Super-fused mHC forward pass - everything in one JIT function.
    
    This minimizes Python overhead and enables better compiler optimizations.
    """
    B, n, C = x.shape
    x_f32 = x.float()
    
    # === Fused Block 1: Aggregation + RMSNorm ===
    H_pre_act = torch.sigmoid(H_pre)
    
    # Optimized aggregation
    x_t = x_f32.transpose(1, 2)  # [B, C, n]
    x_agg = torch.matmul(x_t, H_pre_act.unsqueeze(-1)).squeeze(-1)  # [B, C]
    
    # RMSNorm
    rms = torch.sqrt((x_agg ** 2).mean(dim=-1, keepdim=True) + eps)
    y_norm = (x_agg / rms) * rmsnorm_weight.float()
    
    # === Fused Block 2: Sinkhorn-Knopp ===
    P = torch.exp(H_res)
    for _ in range(sinkhorn_iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    M = P
    
    # === Fused Block 3: Distribution + Mix + Add ===
    H_post_act = 2.0 * torch.sigmoid(H_post)
    
    M_expanded = M.unsqueeze(0).expand(B, -1, -1)
    mixed = torch.bmm(M_expanded, x_f32)
    
    output = mixed + H_post_act.view(1, n, 1) * y_norm.unsqueeze(1)
    
    return output


class MHCLayerSuperFused(nn.Module):
    """Super-fused mHC Layer - minimal overhead."""
    
    def __init__(self, C: int, n: int = 4, sinkhorn_iters: int = 3, eps: float = 1e-6):
        super().__init__()
        self.C = C
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        
        self.rmsnorm_weight = nn.Parameter(torch.ones(C))
        self.H_pre = nn.Parameter(torch.zeros(n))
        self.H_post = nn.Parameter(torch.zeros(n))
        self.H_res = nn.Parameter(torch.eye(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mhc_forward_superfused(
            x, self.H_pre, self.H_post, self.H_res,
            self.rmsnorm_weight, self.sinkhorn_iters, self.eps
        )


# ============================================================================
# AITER + Fused Hybrid
# ============================================================================

class MHCLayerAITERFused(nn.Module):
    """Hybrid: AITER RMSNorm + Fused other operations."""
    
    def __init__(self, C: int, n: int = 4, sinkhorn_iters: int = 3, eps: float = 1e-6):
        super().__init__()
        self.C = C
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.use_aiter = AITER_AVAILABLE and hasattr(aiter, 'rmsnorm2d_fwd')
        
        self.rmsnorm_weight = nn.Parameter(torch.ones(C, dtype=torch.bfloat16))
        self.H_pre = nn.Parameter(torch.zeros(n))
        self.H_post = nn.Parameter(torch.zeros(n))
        self.H_res = nn.Parameter(torch.eye(n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, C = x.shape
        x_f32 = x.float()
        
        # Fused aggregation
        H_pre_act = torch.sigmoid(self.H_pre)
        x_t = x_f32.transpose(1, 2)
        x_agg = torch.matmul(x_t, H_pre_act.unsqueeze(-1)).squeeze(-1)
        
        # AITER RMSNorm (bf16)
        x_agg_bf16 = x_agg.to(torch.bfloat16)
        if self.use_aiter:
            y_norm = aiter.rmsnorm2d_fwd(x_agg_bf16, self.rmsnorm_weight, self.eps)
        else:
            rms = torch.sqrt((x_agg ** 2).mean(dim=-1, keepdim=True) + self.eps)
            y_norm = ((x_agg / rms) * self.rmsnorm_weight.float()).to(torch.bfloat16)
        
        # Fused Sinkhorn
        M = fused_sinkhorn_jit(torch.exp(self.H_res), self.sinkhorn_iters, self.eps)
        
        # Fused distribution
        H_post_act = 2.0 * torch.sigmoid(self.H_post)
        M_expanded = M.unsqueeze(0).expand(B, -1, -1)
        mixed = torch.bmm(M_expanded, x_f32)
        output = mixed + H_post_act.view(1, n, 1) * y_norm.float().unsqueeze(1)
        
        return output


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_fn(fn, *args, warmup: int = 20, iters: int = 200):
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
    
    return (time.perf_counter() - start) * 1000 / iters


def verify_correctness(ref_layer: nn.Module, test_layer: nn.Module, x: torch.Tensor) -> float:
    """Verify correctness between two layers."""
    ref_layer.eval()
    test_layer.eval()
    
    with torch.no_grad():
        out_ref = ref_layer(x)
        out_test = test_layer(x)
    
    return (out_ref.float() - out_test.float()).abs().max().item()


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark():
    device = 'cuda'
    
    configs = [
        (128, 4, 1280),
        (256, 4, 1280),
        (320, 4, 1280),
        (512, 4, 1920),
        (512, 4, 2560),
        (1024, 4, 1920),
    ]
    
    print("=" * 100)
    print("FUSED OPERATIONS BENCHMARK")
    print("=" * 100)
    print(f"{'Config':<22} | {'Non-Fused':<12} | {'JIT Fused':<12} | {'SuperFused':<12} | {'AITER+Fused':<12} | {'Best Speedup':<12}")
    print("-" * 100)
    
    results = []
    
    for B, n, C in configs:
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)
        
        # Create layers
        layer_nonfused = MHCLayerNonFused(C, n).to(device)
        layer_fused = MHCLayerFused(C, n).to(device)
        layer_superfused = MHCLayerSuperFused(C, n).to(device)
        layer_aiter_fused = MHCLayerAITERFused(C, n).to(device)
        
        # Copy weights
        with torch.no_grad():
            layer_fused.rmsnorm_weight.copy_(layer_nonfused.rmsnorm_weight)
            layer_fused.H_pre.copy_(layer_nonfused.H_pre)
            layer_fused.H_post.copy_(layer_nonfused.H_post)
            layer_fused.H_res.copy_(layer_nonfused.H_res)
            
            layer_superfused.rmsnorm_weight.copy_(layer_nonfused.rmsnorm_weight.float())
            layer_superfused.H_pre.copy_(layer_nonfused.H_pre)
            layer_superfused.H_post.copy_(layer_nonfused.H_post)
            layer_superfused.H_res.copy_(layer_nonfused.H_res)
            
            layer_aiter_fused.rmsnorm_weight.copy_(layer_nonfused.rmsnorm_weight)
            layer_aiter_fused.H_pre.copy_(layer_nonfused.H_pre)
            layer_aiter_fused.H_post.copy_(layer_nonfused.H_post)
            layer_aiter_fused.H_res.copy_(layer_nonfused.H_res)
        
        # Benchmark
        time_nonfused = benchmark_fn(layer_nonfused, x)
        time_fused = benchmark_fn(layer_fused, x)
        time_superfused = benchmark_fn(layer_superfused, x)
        time_aiter_fused = benchmark_fn(layer_aiter_fused, x)
        
        # Find best
        times = {
            'nonfused': time_nonfused,
            'fused': time_fused,
            'superfused': time_superfused,
            'aiter_fused': time_aiter_fused,
        }
        best_time = min(times.values())
        best_speedup = time_nonfused / best_time
        
        # Verify correctness
        max_diff_fused = verify_correctness(layer_nonfused, layer_fused, x)
        max_diff_super = verify_correctness(layer_nonfused, layer_superfused, x)
        max_diff_aiter = verify_correctness(layer_nonfused, layer_aiter_fused, x)
        
        config_str = f"B={B}, n={n}, C={C}"
        print(f"{config_str:<22} | {time_nonfused:.3f}ms      | {time_fused:.3f}ms      | {time_superfused:.3f}ms      | {time_aiter_fused:.3f}ms      | {best_speedup:.2f}x")
        
        results.append({
            'config': (B, n, C),
            'times': times,
            'speedup': best_speedup,
            'correctness': {
                'fused': max_diff_fused,
                'superfused': max_diff_super,
                'aiter_fused': max_diff_aiter,
            }
        })
    
    print("=" * 100)
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print("SPEEDUP COMPARISON vs Non-Fused Baseline")
    print("=" * 80)
    print(f"{'Config':<22} | {'JIT Fused':<12} | {'SuperFused':<12} | {'AITER+Fused':<12}")
    print("-" * 80)
    
    for r in results:
        B, n, C = r['config']
        config_str = f"B={B}, n={n}, C={C}"
        baseline = r['times']['nonfused']
        
        fused_speedup = baseline / r['times']['fused']
        super_speedup = baseline / r['times']['superfused']
        aiter_speedup = baseline / r['times']['aiter_fused']
        
        print(f"{config_str:<22} | {fused_speedup:.2f}x        | {super_speedup:.2f}x        | {aiter_speedup:.2f}x")
    
    print("=" * 80)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_speedups = {
        'fused': sum(r['times']['nonfused'] / r['times']['fused'] for r in results) / len(results),
        'superfused': sum(r['times']['nonfused'] / r['times']['superfused'] for r in results) / len(results),
        'aiter_fused': sum(r['times']['nonfused'] / r['times']['aiter_fused'] for r in results) / len(results),
    }
    
    print(f"Average Speedup vs Non-Fused:")
    print(f"  - JIT Fused:    {avg_speedups['fused']:.2f}x")
    print(f"  - SuperFused:   {avg_speedups['superfused']:.2f}x")
    print(f"  - AITER+Fused:  {avg_speedups['aiter_fused']:.2f}x")
    
    best_method = max(avg_speedups, key=avg_speedups.get)
    print(f"\nBest Method: {best_method} ({avg_speedups[best_method]:.2f}x average speedup)")
    
    # Correctness
    print("\n" + "=" * 60)
    print("CORRECTNESS (Max Diff vs Non-Fused)")
    print("=" * 60)
    for r in results:
        B, n, C = r['config']
        print(f"B={B}, n={n}, C={C}: fused={r['correctness']['fused']:.2e}, super={r['correctness']['superfused']:.2e}, aiter={r['correctness']['aiter_fused']:.2e}")
    
    print("=" * 60)
    
    return results


def run_isolated_op_benchmark():
    """Benchmark individual fused operations."""
    device = 'cuda'
    
    print("\n" + "=" * 80)
    print("ISOLATED FUSED OPERATION BENCHMARKS")
    print("=" * 80)
    
    # Test Sinkhorn-Knopp fusion
    print("\n--- Sinkhorn-Knopp: Non-Fused vs JIT Fused ---")
    print(f"{'Size':<15} | {'Non-Fused':<12} | {'JIT Fused':<12} | {'Speedup':<10}")
    print("-" * 55)
    
    for n in [4, 8, 16, 32]:
        inp = torch.exp(torch.randn(n, n, device=device))
        
        # Non-fused
        time_nf = benchmark_fn(sinkhorn_knopp_pytorch, inp, 3, 1e-6)
        
        # JIT Fused
        time_f = benchmark_fn(fused_sinkhorn_jit, inp, 3, 1e-6)
        
        speedup = time_nf / time_f
        print(f"n={n}".ljust(15) + f" | {time_nf:.4f}ms     | {time_f:.4f}ms     | {speedup:.2f}x")
    
    print()
    
    # Test aggregation + rmsnorm fusion
    print("--- Stream Agg + RMSNorm: Non-Fused vs JIT Fused ---")
    print(f"{'Size':<20} | {'Non-Fused':<12} | {'JIT Fused':<12} | {'Speedup':<10}")
    print("-" * 60)
    
    for B, n, C in [(256, 4, 1280), (512, 4, 1920), (1024, 4, 2560)]:
        x = torch.randn(B, n, C, device=device)
        H_pre = torch.zeros(n, device=device)
        weight = torch.ones(C, device=device)
        
        def nonfused():
            H_pre_act = torch.sigmoid(H_pre)
            x_agg = torch.einsum('bnc,n->bc', x, H_pre_act)
            return rmsnorm_pytorch(x_agg.to(torch.bfloat16), weight.to(torch.bfloat16))
        
        def fused():
            return fused_stream_agg_rmsnorm_jit(x, H_pre, weight)
        
        time_nf = benchmark_fn(nonfused)
        time_f = benchmark_fn(fused)
        
        speedup = time_nf / time_f
        print(f"({B}, {n}, {C})".ljust(20) + f" | {time_nf:.4f}ms     | {time_f:.4f}ms     | {speedup:.2f}x")
    
    print("=" * 80)


if __name__ == "__main__":
    # Isolated ops first
    run_isolated_op_benchmark()
    
    # Full layer benchmark
    results = run_benchmark()
    
    print("\n✓ Benchmark complete!")

