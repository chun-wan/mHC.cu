"""
mHC Benchmark Script for DeepSeek-V3.2

This script benchmarks mHC (Manifold-Constrained Hyper-Connections) integration
with different configurations and compares against baseline.

Usage:
    python mhc_benchmark.py [--mode kernel|e2e|both]
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import Dict, List
import sys

sys.path.insert(0, '/opt/mHC.cu/src/python')

from mhc_vllm_integration import MHCResidualConnection, benchmark_mhc_residual


def benchmark_kernel_performance() -> Dict:
    """
    Benchmark mHC kernel-level performance with various configurations.
    """
    print("\n" + "=" * 70)
    print("mHC Kernel-Level Performance Benchmark")
    print("=" * 70)
    
    configs = [
        # (batch_size, seq_len, hidden_size)
        (1, 128, 7168),    # Small batch
        (4, 256, 7168),    # Medium batch
        (8, 512, 7168),    # Large batch
        (16, 1024, 7168),  # Very large batch
    ]
    
    results = []
    
    for batch_size, seq_len, hidden_size in configs:
        print(f"\n[Config] batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
        
        try:
            result = benchmark_mhc_residual(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=hidden_size,
                warmup_iters=10,
                bench_iters=50,
                device="cuda"
            )
            
            print(f"  mHC time:      {result['mhc_time_ms']:.3f} ms")
            print(f"  Standard time: {result['standard_time_ms']:.3f} ms")
            print(f"  Overhead:      {result['overhead']:.2f}x")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return results


def benchmark_superfused_mhc() -> Dict:
    """
    Benchmark SuperFused mHC implementation.
    """
    print("\n" + "=" * 70)
    print("SuperFused mHC Performance Benchmark")
    print("=" * 70)
    
    try:
        from mhc_aiter import MHCLayerSuperFused
        
        configs = [
            # (batch_size, n_streams, hidden_size)
            (128, 4, 1280),
            (256, 4, 1280),
            (512, 4, 1920),
            (1024, 4, 2560),
        ]
        
        results = []
        
        for batch_size, n_streams, hidden_size in configs:
            print(f"\n[Config] batch={batch_size}, n_streams={n_streams}, hidden={hidden_size}")
            
            layer = MHCLayerSuperFused(
                hidden_dim=hidden_size,
                n_streams=n_streams
            ).cuda()
            
            x = torch.randn(batch_size, n_streams, hidden_size, device="cuda", dtype=torch.float32)
            
            # Warmup
            for _ in range(10):
                _ = layer(x)
                torch.cuda.synchronize()
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                _ = layer(x)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000 / 50
            
            print(f"  SuperFused time: {elapsed_ms:.3f} ms")
            print(f"  Throughput: {batch_size / (elapsed_ms / 1000):.0f} samples/sec")
            
            results.append({
                "batch_size": batch_size,
                "n_streams": n_streams,
                "hidden_size": hidden_size,
                "time_ms": elapsed_ms,
            })
            
            torch.cuda.empty_cache()
        
        return results
        
    except ImportError as e:
        print(f"SuperFused mHC not available: {e}")
        return []


def benchmark_mhc_training_simulation() -> Dict:
    """
    Simulate mHC impact on training-like workload.
    """
    print("\n" + "=" * 70)
    print("mHC Training Simulation Benchmark")
    print("=" * 70)
    
    # Simulate DeepSeek-V3.2 dimensions
    hidden_size = 7168
    num_layers = 61
    batch_size = 4
    seq_len = 2048
    
    print(f"\n[Config] DeepSeek-V3.2 like:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: {seq_len}")
    
    # Create mHC layers
    mhc_layers = nn.ModuleList([
        MHCResidualConnection(hidden_size) for _ in range(num_layers)
    ]).cuda()
    
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    
    # Warmup
    print("\n[Warmup] Running warmup iterations...")
    for _ in range(3):
        h, r = hidden_states.clone(), residual.clone()
        for layer in mhc_layers:
            h, r = layer(h, r)
        torch.cuda.synchronize()
    
    # Benchmark mHC forward pass through all layers
    print("[Benchmark] Running mHC forward pass...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(10):
        h, r = hidden_states.clone(), residual.clone()
        for layer in mhc_layers:
            h, r = layer(h, r)
    
    torch.cuda.synchronize()
    mhc_time = (time.perf_counter() - start) * 1000 / 10
    
    # Benchmark standard residual
    print("[Benchmark] Running standard residual...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(10):
        h = hidden_states.clone()
        r = residual.clone()
        for _ in range(num_layers):
            h = h + r  # Standard residual
    
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) * 1000 / 10
    
    print(f"\n[Results]")
    print(f"  mHC total time (61 layers): {mhc_time:.1f} ms")
    print(f"  Standard total time (61 layers): {std_time:.3f} ms")
    print(f"  Overhead: {mhc_time / std_time:.1f}x")
    print(f"  mHC per-layer time: {mhc_time / num_layers:.3f} ms")
    
    return {
        "mhc_total_ms": mhc_time,
        "std_total_ms": std_time,
        "overhead": mhc_time / std_time,
        "per_layer_ms": mhc_time / num_layers,
    }


def print_summary(kernel_results: List, superfused_results: List, training_results: Dict):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\n## Kernel-Level Performance")
    print("| Batch | Seq | Hidden | mHC (ms) | Std (ms) | Overhead |")
    print("|-------|-----|--------|----------|----------|----------|")
    for r in kernel_results:
        print(f"| {r['batch_size']:5} | {r['seq_len']:3} | {r['hidden_size']:6} | {r['mhc_time_ms']:8.3f} | {r['standard_time_ms']:8.3f} | {r['overhead']:8.1f}x |")
    
    if superfused_results:
        print("\n## SuperFused mHC Performance")
        print("| Batch | Streams | Hidden | Time (ms) | Throughput |")
        print("|-------|---------|--------|-----------|------------|")
        for r in superfused_results:
            tput = r['batch_size'] / (r['time_ms'] / 1000)
            print(f"| {r['batch_size']:5} | {r['n_streams']:7} | {r['hidden_size']:6} | {r['time_ms']:9.3f} | {tput:10.0f}/s |")
    
    if training_results:
        print("\n## Training Simulation (DeepSeek-V3.2 like, 61 layers)")
        print(f"  - mHC total: {training_results['mhc_total_ms']:.1f} ms")
        print(f"  - Standard total: {training_results['std_total_ms']:.3f} ms")
        print(f"  - Overhead: {training_results['overhead']:.1f}x")
        print(f"  - Per-layer: {training_results['per_layer_ms']:.3f} ms")
    
    print("\n## Conclusions")
    print("  - mHC PyTorch fallback has significant overhead (~400-500x)")
    print("  - SuperFused JIT version is much faster but still has overhead")
    print("  - Native HIP kernels required for production use")
    print("  - mHC benefit is in model quality, not inference speed")
    print("  - Consider using mHC only during training, not inference")


def main():
    parser = argparse.ArgumentParser(description="mHC Benchmark for DeepSeek-V3.2")
    parser.add_argument("--mode", choices=["kernel", "superfused", "training", "all"], 
                        default="all", help="Benchmark mode")
    args = parser.parse_args()
    
    print("=" * 70)
    print("mHC (Manifold-Constrained Hyper-Connections) Benchmark")
    print("Target: DeepSeek-V3.2 Integration")
    print("=" * 70)
    
    kernel_results = []
    superfused_results = []
    training_results = {}
    
    if args.mode in ["kernel", "all"]:
        kernel_results = benchmark_kernel_performance()
    
    if args.mode in ["superfused", "all"]:
        superfused_results = benchmark_superfused_mhc()
    
    if args.mode in ["training", "all"]:
        training_results = benchmark_mhc_training_simulation()
    
    print_summary(kernel_results, superfused_results, training_results)


if __name__ == "__main__":
    main()

