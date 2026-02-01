#!/usr/bin/env python3
"""
Benchmark script for mHC.cu with AITER vs non-AITER comparison.

This script benchmarks the mHC layer performance on AMD MI300X GPUs,
comparing:
1. Custom HIP kernels (baseline)
2. AITER-accelerated kernels (when available)
3. PyTorch fallback (reference)

Usage:
    python scripts/benchmark_aiter.py [--batch-sizes 128,256,512] [--hidden-dims 1280,1920]

Container: 63015cccf5f

AITER repository: https://github.com/ROCm/aiter
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.nn as nn

# Try to import mHC modules
try:
    from mhc_aiter import MHCLayer, MHCLayerAITER, MHCLayerDynamic, benchmark_mhc
    from aiter_ops import is_aiter_available, get_aiter_info
    MHC_AVAILABLE = True
except ImportError as e:
    print(f"[Benchmark] Warning: mHC modules not available: {e}")
    print("[Benchmark] Run 'pip install -e .' first")
    MHC_AVAILABLE = False

# Try to import custom HIP kernels
try:
    import mhc_hip
    HIP_KERNELS_AVAILABLE = True
except ImportError:
    HIP_KERNELS_AVAILABLE = False


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    
    return info


def benchmark_layer(
    layer: nn.Module,
    batch_size: int,
    n_streams: int,
    hidden_dim: int,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Benchmark a layer with detailed metrics.
    
    Returns:
        Dictionary with benchmark results
    """
    layer = layer.to(device)
    layer.eval()
    
    x = torch.randn(batch_size, n_streams, hidden_dim, device=device, dtype=torch.float32)
    
    # Memory before
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024**2)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = layer(x)
            torch.cuda.synchronize()
    
    # Benchmark forward
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(bench_iters):
            _ = layer(x)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    forward_time_ms = (end_time - start_time) * 1000 / bench_iters
    
    # Memory peak
    mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
    
    # Benchmark backward (if training mode)
    backward_time_ms = None
    if hasattr(layer, 'parameters') and any(p.requires_grad for p in layer.parameters()):
        layer.train()
        x.requires_grad_(True)
        
        # Warmup backward
        for _ in range(warmup_iters):
            y = layer(x)
            loss = y.sum()
            loss.backward()
            torch.cuda.synchronize()
        
        # Benchmark backward
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(bench_iters):
            y = layer(x)
            loss = y.sum()
            loss.backward()
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        backward_time_ms = (end_time - start_time) * 1000 / bench_iters
    
    return {
        "forward_time_ms": forward_time_ms,
        "backward_time_ms": backward_time_ms,
        "total_time_ms": forward_time_ms + (backward_time_ms or 0),
        "throughput_samples_per_sec": batch_size / (forward_time_ms / 1000),
        "memory_mb": mem_peak - mem_before,
        "peak_memory_mb": mem_peak,
    }


def run_benchmark(
    batch_sizes: List[int],
    hidden_dims: List[int],
    n_streams_list: List[int],
    sinkhorn_iters: int = 3,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run full benchmark suite.
    
    Returns:
        Dictionary with all benchmark results
    """
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "gpu_info": get_gpu_info(),
            "aiter_available": is_aiter_available() if MHC_AVAILABLE else False,
            "aiter_info": get_aiter_info() if MHC_AVAILABLE else {},
            "hip_kernels_available": HIP_KERNELS_AVAILABLE,
            "warmup_iters": warmup_iters,
            "bench_iters": bench_iters,
        },
        "benchmarks": [],
    }
    
    total_configs = len(batch_sizes) * len(hidden_dims) * len(n_streams_list)
    current = 0
    
    for batch_size in batch_sizes:
        for hidden_dim in hidden_dims:
            for n_streams in n_streams_list:
                current += 1
                config_name = f"B={batch_size}, C={hidden_dim}, n={n_streams}"
                print(f"\n[{current}/{total_configs}] Benchmarking {config_name}")
                
                config_results = {
                    "batch_size": batch_size,
                    "hidden_dim": hidden_dim,
                    "n_streams": n_streams,
                    "sinkhorn_iters": sinkhorn_iters,
                    "backends": {},
                }
                
                # 1. PyTorch fallback (baseline)
                print("  - PyTorch fallback...")
                try:
                    layer_pytorch = MHCLayer(
                        hidden_dim=hidden_dim,
                        n_streams=n_streams,
                        sinkhorn_iters=sinkhorn_iters,
                        use_hip=False  # Force PyTorch
                    )
                    config_results["backends"]["pytorch"] = benchmark_layer(
                        layer_pytorch, batch_size, n_streams, hidden_dim,
                        warmup_iters, bench_iters, device
                    )
                    print(f"    Forward: {config_results['backends']['pytorch']['forward_time_ms']:.3f} ms")
                except Exception as e:
                    print(f"    Error: {e}")
                    config_results["backends"]["pytorch"] = {"error": str(e)}
                
                # 2. Custom HIP kernels
                if HIP_KERNELS_AVAILABLE:
                    print("  - HIP kernels...")
                    try:
                        layer_hip = MHCLayer(
                            hidden_dim=hidden_dim,
                            n_streams=n_streams,
                            sinkhorn_iters=sinkhorn_iters,
                            use_hip=True
                        )
                        config_results["backends"]["hip"] = benchmark_layer(
                            layer_hip, batch_size, n_streams, hidden_dim,
                            warmup_iters, bench_iters, device
                        )
                        print(f"    Forward: {config_results['backends']['hip']['forward_time_ms']:.3f} ms")
                    except Exception as e:
                        print(f"    Error: {e}")
                        config_results["backends"]["hip"] = {"error": str(e)}
                
                # 3. AITER-accelerated
                if is_aiter_available() if MHC_AVAILABLE else False:
                    print("  - AITER-accelerated...")
                    try:
                        layer_aiter = MHCLayerAITER(
                            hidden_dim=hidden_dim,
                            n_streams=n_streams,
                            sinkhorn_iters=sinkhorn_iters,
                            use_aiter=True
                        )
                        config_results["backends"]["aiter"] = benchmark_layer(
                            layer_aiter, batch_size, n_streams, hidden_dim,
                            warmup_iters, bench_iters, device
                        )
                        print(f"    Forward: {config_results['backends']['aiter']['forward_time_ms']:.3f} ms")
                    except Exception as e:
                        print(f"    Error: {e}")
                        config_results["backends"]["aiter"] = {"error": str(e)}
                
                # 4. Dynamic H (if HIP available)
                if HIP_KERNELS_AVAILABLE:
                    print("  - Dynamic H (HIP)...")
                    try:
                        layer_dynamic = MHCLayerDynamic(
                            hidden_dim=hidden_dim,
                            n_streams=n_streams,
                            sinkhorn_iters=sinkhorn_iters,
                            use_aiter=False
                        )
                        config_results["backends"]["dynamic_hip"] = benchmark_layer(
                            layer_dynamic, batch_size, n_streams, hidden_dim,
                            warmup_iters, bench_iters, device
                        )
                        print(f"    Forward: {config_results['backends']['dynamic_hip']['forward_time_ms']:.3f} ms")
                    except Exception as e:
                        print(f"    Error: {e}")
                        config_results["backends"]["dynamic_hip"] = {"error": str(e)}
                
                # Calculate speedups
                if "pytorch" in config_results["backends"] and "forward_time_ms" in config_results["backends"]["pytorch"]:
                    baseline = config_results["backends"]["pytorch"]["forward_time_ms"]
                    config_results["speedups"] = {}
                    
                    for backend, data in config_results["backends"].items():
                        if backend != "pytorch" and "forward_time_ms" in data:
                            speedup = baseline / data["forward_time_ms"]
                            config_results["speedups"][f"{backend}_vs_pytorch"] = speedup
                
                results["benchmarks"].append(config_results)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"{'Config':<30} | {'PyTorch':<12} | {'HIP':<12} | {'AITER':<12} | {'Speedup':<10}")
    print("-" * 80)
    
    for bench in results["benchmarks"]:
        config = f"B={bench['batch_size']}, C={bench['hidden_dim']}, n={bench['n_streams']}"
        
        pytorch_ms = bench["backends"].get("pytorch", {}).get("forward_time_ms", "N/A")
        hip_ms = bench["backends"].get("hip", {}).get("forward_time_ms", "N/A")
        aiter_ms = bench["backends"].get("aiter", {}).get("forward_time_ms", "N/A")
        
        # Best speedup
        speedup = "N/A"
        if "speedups" in bench:
            best_speedup = max(bench["speedups"].values()) if bench["speedups"] else None
            if best_speedup:
                speedup = f"{best_speedup:.2f}x"
        
        pytorch_str = f"{pytorch_ms:.3f}ms" if isinstance(pytorch_ms, float) else pytorch_ms
        hip_str = f"{hip_ms:.3f}ms" if isinstance(hip_ms, float) else hip_ms
        aiter_str = f"{aiter_ms:.3f}ms" if isinstance(aiter_ms, float) else aiter_ms
        
        print(f"{config:<30} | {pytorch_str:<12} | {hip_str:<12} | {aiter_str:<12} | {speedup:<10}")
    
    print("=" * 80)
    
    # AITER status
    if results["metadata"].get("aiter_available"):
        print("\n✓ AITER is available and was used for benchmarking")
        aiter_info = results["metadata"].get("aiter_info", {})
        print(f"  RMSNorm: {aiter_info.get('rmsnorm', 'N/A')}")
        print(f"  GEMM: {aiter_info.get('gemm', 'N/A')}")
        print(f"  Sigmoid: {aiter_info.get('sigmoid', 'N/A')}")
    else:
        print("\n✗ AITER is not available")
        print("  Install AITER for additional acceleration:")
        print("  git clone --recursive https://github.com/ROCm/aiter.git")
        print("  cd aiter && python3 setup.py develop")


def main():
    parser = argparse.ArgumentParser(description="Benchmark mHC.cu with AITER comparison")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="128,256,320,512",
        help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="1280,1920,2560",
        help="Comma-separated hidden dimensions"
    )
    parser.add_argument(
        "--n-streams",
        type=str,
        default="4",
        help="Comma-separated number of streams"
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=3,
        help="Number of Sinkhorn-Knopp iterations"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    if not MHC_AVAILABLE:
        print("Error: mHC modules not available. Install with 'pip install -e .'")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("Error: CUDA/ROCm not available")
        sys.exit(1)
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    n_streams_list = [int(x) for x in args.n_streams.split(",")]
    
    print("=" * 80)
    print("mHC.cu Benchmark - AITER Comparison")
    print("=" * 80)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"N streams: {n_streams_list}")
    print(f"Sinkhorn iters: {args.sinkhorn_iters}")
    print(f"Device: {args.device}")
    print(f"AITER available: {is_aiter_available()}")
    print(f"HIP kernels available: {HIP_KERNELS_AVAILABLE}")
    print("=" * 80)
    
    results = run_benchmark(
        batch_sizes=batch_sizes,
        hidden_dims=hidden_dims,
        n_streams_list=n_streams_list,
        sinkhorn_iters=args.sinkhorn_iters,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        device=args.device
    )
    
    print_summary(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

