#!/usr/bin/env python3
"""
Complete AITER Integration Benchmark for mHC.cu

This script provides a comprehensive benchmark comparing:
1. PyTorch baseline (f32)
2. PyTorch with bf16
3. AITER-accelerated (bf16)

Usage:
    python scripts/benchmark_full_aiter.py
"""

import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

# ============================================================================
# Environment Check
# ============================================================================

print("=" * 70)
print("mHC.cu AITER Integration - Complete Benchmark")
print("=" * 70)

print(f"\nPython: {sys.version}")
print(f"PyTorch: {torch.__version__}")

if not torch.cuda.is_available():
    print("ERROR: CUDA/ROCm not available")
    sys.exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Device count: {torch.cuda.device_count()}")

# Check AITER
try:
    import aiter
    AITER_AVAILABLE = True
    print(f"\n✓ AITER is available")
except ImportError:
    AITER_AVAILABLE = False
    print(f"\n✗ AITER not available")

print()

# ============================================================================
# PyTorch Reference Implementations
# ============================================================================

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch RMSNorm."""
    rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    out = (x.float() / rms) * weight.float()
    return out.to(x.dtype), rms.squeeze(-1)


def pytorch_sinkhorn_knopp(inp: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch Sinkhorn-Knopp algorithm."""
    P = inp.clone()
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P


class MHCLayerPyTorch(nn.Module):
    """Pure PyTorch mHC Layer implementation."""
    
    def __init__(self, hidden_dim: int, n_streams: int = 4, sinkhorn_iters: int = 3, 
                 eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.dtype = dtype
        
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=dtype))
        self.H_pre = nn.Parameter(torch.zeros(n_streams, dtype=torch.float32))
        self.H_post = nn.Parameter(torch.zeros(n_streams, dtype=torch.float32))
        self.H_res = nn.Parameter(torch.eye(n_streams, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, C = x.shape
        x_f32 = x.float()
        
        # 1. Stream aggregation with sigmoid
        H_pre_activated = torch.sigmoid(self.H_pre)
        x_agg = torch.einsum('bnc,n->bc', x_f32, H_pre_activated)
        
        # 2. RMSNorm
        x_agg_typed = x_agg.to(self.dtype)
        y_norm, rms = pytorch_rmsnorm(x_agg_typed, self.rmsnorm_weight, self.eps)
        
        # 3. Sinkhorn-Knopp
        M = pytorch_sinkhorn_knopp(torch.exp(self.H_res), self.sinkhorn_iters, self.eps)
        
        # 4. Stream distribution
        H_post_activated = 2.0 * torch.sigmoid(self.H_post)
        mixed = torch.einsum('ij,bjc->bic', M, x_f32)
        output = mixed + H_post_activated.view(1, n, 1) * y_norm.float().unsqueeze(1)
        
        return output


# ============================================================================
# AITER-Accelerated Implementation
# ============================================================================

class MHCLayerAITER(nn.Module):
    """AITER-accelerated mHC Layer implementation."""
    
    def __init__(self, hidden_dim: int, n_streams: int = 4, sinkhorn_iters: int = 3, 
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        
        # Use bf16 for AITER compatibility
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))
        self.H_pre = nn.Parameter(torch.zeros(n_streams, dtype=torch.float32))
        self.H_post = nn.Parameter(torch.zeros(n_streams, dtype=torch.float32))
        self.H_res = nn.Parameter(torch.eye(n_streams, dtype=torch.float32))
        
        self.use_aiter_rmsnorm = AITER_AVAILABLE and hasattr(aiter, 'rmsnorm2d_fwd')
        self.use_aiter_sigmoid = AITER_AVAILABLE and hasattr(aiter, 'sigmoid')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n, C = x.shape
        x_f32 = x.float()
        
        # 1. Stream aggregation with sigmoid
        # Note: AITER sigmoid is slower for small 1D tensors, use torch.sigmoid
        H_pre_activated = torch.sigmoid(self.H_pre)
        x_agg = torch.einsum('bnc,n->bc', x_f32, H_pre_activated)
        
        # 2. RMSNorm (AITER rmsnorm with bf16) - main speedup here!
        x_agg_bf16 = x_agg.to(torch.bfloat16)
        if self.use_aiter_rmsnorm:
            y_norm = aiter.rmsnorm2d_fwd(x_agg_bf16, self.rmsnorm_weight, self.eps)
        else:
            y_norm, _ = pytorch_rmsnorm(x_agg_bf16, self.rmsnorm_weight, self.eps)
        
        # 3. Sinkhorn-Knopp (no AITER equivalent, use PyTorch)
        M = pytorch_sinkhorn_knopp(torch.exp(self.H_res), self.sinkhorn_iters, self.eps)
        
        # 4. Stream distribution with sigmoid
        H_post_activated = 2.0 * torch.sigmoid(self.H_post)
        
        mixed = torch.einsum('ij,bjc->bic', M, x_f32)
        output = mixed + H_post_activated.view(1, n, 1) * y_norm.float().unsqueeze(1)
        
        return output


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark_layer(layer: nn.Module, x: torch.Tensor, warmup: int = 20, iters: int = 200) -> Dict[str, float]:
    """Benchmark a layer."""
    layer.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)
            torch.cuda.synchronize()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    
    # Benchmark forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = layer(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    forward_ms = (end - start) * 1000 / iters
    mem_peak = torch.cuda.max_memory_allocated()
    
    # Benchmark backward
    layer.train()
    x_grad = x.clone().requires_grad_(True)
    
    # Warmup backward
    for _ in range(warmup // 2):
        y = layer(x_grad)
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters // 2):
        y = layer(x_grad)
        loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    backward_ms = (end - start) * 1000 / (iters // 2)
    
    return {
        'forward_ms': forward_ms,
        'backward_ms': backward_ms,
        'total_ms': forward_ms + backward_ms,
        'memory_mb': (mem_peak - mem_before) / (1024 * 1024),
    }


def verify_correctness(layer_ref: nn.Module, layer_test: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """Verify correctness between two layers."""
    layer_ref.eval()
    layer_test.eval()
    
    with torch.no_grad():
        out_ref = layer_ref(x)
        out_test = layer_test(x)
    
    diff = (out_ref.float() - out_test.float()).abs()
    
    return {
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'relative_diff': (diff / (out_ref.float().abs() + 1e-8)).mean().item(),
    }


# ============================================================================
# Main Benchmark
# ============================================================================

def run_full_benchmark():
    """Run complete benchmark suite."""
    device = 'cuda'
    
    configs = [
        # (batch_size, n_streams, hidden_dim)
        (128, 4, 1280),
        (256, 4, 1280),
        (320, 4, 1280),
        (512, 4, 1920),
        (512, 4, 2560),
        (1024, 4, 1280),
        (1024, 4, 1920),
    ]
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': torch.cuda.get_device_name(0),
            'pytorch_version': torch.__version__,
            'aiter_available': AITER_AVAILABLE,
        },
        'benchmarks': [],
    }
    
    print("=" * 90)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Config':<22} | {'PyTorch f32':<14} | {'PyTorch bf16':<14} | {'AITER bf16':<14} | {'Speedup':<10}")
    print("-" * 90)
    
    for B, n, C in configs:
        config_str = f"B={B}, n={n}, C={C}"
        config_result = {
            'batch_size': B,
            'n_streams': n,
            'hidden_dim': C,
            'results': {},
        }
        
        # Create input tensor
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)
        
        # 1. PyTorch f32 baseline
        layer_f32 = MHCLayerPyTorch(C, n, dtype=torch.float32).to(device)
        result_f32 = benchmark_layer(layer_f32, x)
        config_result['results']['pytorch_f32'] = result_f32
        
        # 2. PyTorch bf16
        layer_bf16 = MHCLayerPyTorch(C, n, dtype=torch.bfloat16).to(device)
        result_bf16 = benchmark_layer(layer_bf16, x)
        config_result['results']['pytorch_bf16'] = result_bf16
        
        # 3. AITER bf16 (if available)
        if AITER_AVAILABLE:
            layer_aiter = MHCLayerAITER(C, n).to(device)
            result_aiter = benchmark_layer(layer_aiter, x)
            config_result['results']['aiter_bf16'] = result_aiter
            
            # Verify correctness
            correctness = verify_correctness(layer_f32, layer_aiter, x)
            config_result['correctness'] = correctness
            
            aiter_str = f"{result_aiter['forward_ms']:.3f}ms"
            speedup = result_f32['forward_ms'] / result_aiter['forward_ms']
            speedup_str = f"{speedup:.2f}x"
        else:
            aiter_str = "N/A"
            speedup_str = "N/A"
        
        print(f"{config_str:<22} | {result_f32['forward_ms']:.3f}ms        | {result_bf16['forward_ms']:.3f}ms        | {aiter_str:<14} | {speedup_str:<10}")
        
        results['benchmarks'].append(config_result)
    
    print("=" * 90)
    
    # Print detailed results
    print("\n" + "=" * 90)
    print("DETAILED RESULTS (Forward + Backward)")
    print("=" * 90)
    print(f"{'Config':<22} | {'Backend':<14} | {'Forward':<10} | {'Backward':<10} | {'Total':<10} | {'Memory':<10}")
    print("-" * 90)
    
    for bench in results['benchmarks']:
        config_str = f"B={bench['batch_size']}, n={bench['n_streams']}, C={bench['hidden_dim']}"
        
        for backend, data in bench['results'].items():
            print(f"{config_str:<22} | {backend:<14} | {data['forward_ms']:.3f}ms    | {data['backward_ms']:.3f}ms    | {data['total_ms']:.3f}ms    | {data['memory_mb']:.1f}MB")
        print("-" * 90)
    
    print("=" * 90)
    
    # Print correctness results
    if AITER_AVAILABLE:
        print("\n" + "=" * 70)
        print("CORRECTNESS VERIFICATION (AITER vs PyTorch f32)")
        print("=" * 70)
        print(f"{'Config':<30} | {'Max Diff':<15} | {'Mean Diff':<15} | {'Rel Diff':<15}")
        print("-" * 70)
        
        for bench in results['benchmarks']:
            if 'correctness' in bench:
                config_str = f"B={bench['batch_size']}, n={bench['n_streams']}, C={bench['hidden_dim']}"
                c = bench['correctness']
                print(f"{config_str:<30} | {c['max_diff']:.6e}     | {c['mean_diff']:.6e}     | {c['relative_diff']:.6e}")
        
        print("=" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if AITER_AVAILABLE:
        speedups = []
        for bench in results['benchmarks']:
            if 'aiter_bf16' in bench['results']:
                f32_time = bench['results']['pytorch_f32']['forward_ms']
                aiter_time = bench['results']['aiter_bf16']['forward_ms']
                speedups.append(f32_time / aiter_time)
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            
            print(f"AITER vs PyTorch f32 Forward Speedup:")
            print(f"  - Average: {avg_speedup:.2f}x")
            print(f"  - Max: {max_speedup:.2f}x")
            print(f"  - Min: {min_speedup:.2f}x")
    else:
        print("AITER not available - showing PyTorch-only results")
    
    print("=" * 70)
    
    return results


def run_isolated_op_benchmark():
    """Benchmark individual operations."""
    device = 'cuda'
    
    print("\n" + "=" * 70)
    print("ISOLATED OPERATION BENCHMARKS")
    print("=" * 70)
    
    sizes = [(256, 1280), (512, 1920), (1024, 2560), (2048, 4096)]
    
    # RMSNorm benchmark
    print("\n--- RMSNorm Benchmark ---")
    print(f"{'Size':<20} | {'PyTorch f32':<12} | {'PyTorch bf16':<12} | {'AITER bf16':<12} | {'Speedup':<10}")
    print("-" * 70)
    
    for B, C in sizes:
        x_f32 = torch.randn(B, C, device=device, dtype=torch.float32)
        x_bf16 = x_f32.to(torch.bfloat16)
        w_f32 = torch.ones(C, device=device, dtype=torch.float32)
        w_bf16 = w_f32.to(torch.bfloat16)
        
        # PyTorch f32
        def pt_f32():
            return pytorch_rmsnorm(x_f32, w_f32)
        
        # PyTorch bf16
        def pt_bf16():
            return pytorch_rmsnorm(x_bf16, w_bf16)
        
        # Warmup & benchmark
        for _ in range(20):
            pt_f32()
            torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            pt_f32()
        torch.cuda.synchronize()
        pt_f32_time = (time.perf_counter() - start) * 1000 / 200
        
        for _ in range(20):
            pt_bf16()
            torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            pt_bf16()
        torch.cuda.synchronize()
        pt_bf16_time = (time.perf_counter() - start) * 1000 / 200
        
        if AITER_AVAILABLE and hasattr(aiter, 'rmsnorm2d_fwd'):
            def aiter_rms():
                return aiter.rmsnorm2d_fwd(x_bf16, w_bf16, 1e-6)
            
            for _ in range(20):
                aiter_rms()
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(200):
                aiter_rms()
            torch.cuda.synchronize()
            aiter_time = (time.perf_counter() - start) * 1000 / 200
            
            speedup = pt_f32_time / aiter_time
            aiter_str = f"{aiter_time:.4f}ms"
            speedup_str = f"{speedup:.2f}x"
        else:
            aiter_str = "N/A"
            speedup_str = "N/A"
        
        print(f"({B}, {C})".ljust(20) + f" | {pt_f32_time:.4f}ms     | {pt_bf16_time:.4f}ms     | {aiter_str:<12} | {speedup_str:<10}")
    
    print("=" * 70)
    
    # Sigmoid benchmark
    print("\n--- Sigmoid Benchmark ---")
    print(f"{'Size':<20} | {'PyTorch':<12} | {'AITER':<12} | {'Speedup':<10}")
    print("-" * 55)
    
    for B, C in sizes:
        x = torch.randn(B, C, device=device, dtype=torch.float32)
        
        # PyTorch
        for _ in range(20):
            torch.sigmoid(x)
            torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            torch.sigmoid(x)
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - start) * 1000 / 200
        
        if AITER_AVAILABLE and hasattr(aiter, 'sigmoid'):
            for _ in range(20):
                aiter.sigmoid(x)
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(200):
                aiter.sigmoid(x)
            torch.cuda.synchronize()
            aiter_time = (time.perf_counter() - start) * 1000 / 200
            
            speedup = pt_time / aiter_time
            aiter_str = f"{aiter_time:.4f}ms"
            speedup_str = f"{speedup:.2f}x"
        else:
            aiter_str = "N/A"
            speedup_str = "N/A"
        
        print(f"({B}, {C})".ljust(20) + f" | {pt_time:.4f}ms     | {aiter_str:<12} | {speedup_str:<10}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run isolated op benchmarks first
    run_isolated_op_benchmark()
    
    # Run full mHC layer benchmark
    results = run_full_benchmark()
    
    # Save results
    output_file = '/tmp/mhc_aiter_benchmark_results.json'
    with open(output_file, 'w') as f:
        # Convert non-serializable items
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float):
                return round(obj, 6)
            return obj
        
        json.dump(clean_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("\n✓ Benchmark complete!")

