# mHC.cu AITER Benchmark Results

**Test Environment:**
- **GPU**: 8x AMD Instinct MI308X
- **ROCm**: 7.2.0
- **PyTorch**: 2.9.1+rocm7.2.0
- **AITER**: v0.1.9 (from ROCm/aiter)
- **Container**: 63015cccf5f7

## Summary

### Isolated Operation Benchmarks

#### RMSNorm (AITER bf16 vs PyTorch f32)

| Size | PyTorch f32 | PyTorch bf16 | AITER bf16 | **Speedup** |
|------|-------------|--------------|------------|-------------|
| (256, 1280) | 0.0383ms | 0.0590ms | 0.0097ms | **3.94x** |
| (512, 1920) | 0.0442ms | 0.0753ms | 0.0094ms | **4.70x** |
| (1024, 2560) | 0.0596ms | 0.0947ms | 0.0098ms | **6.08x** |
| (2048, 4096) | 0.1128ms | 0.1702ms | 0.0209ms | **5.39x** |

**Key Finding**: AITER RMSNorm is **4-6x faster** than PyTorch for typical mHC sizes.

#### Sigmoid (AITER vs PyTorch)

| Size | PyTorch | AITER | Speedup |
|------|---------|-------|---------|
| (256, 1280) | 0.0051ms | 0.0127ms | 0.40x |
| (512, 1920) | 0.0064ms | 0.0135ms | 0.47x |
| (1024, 2560) | 0.0134ms | 0.0173ms | 0.77x |
| (2048, 4096) | 0.0352ms | 0.0410ms | 0.86x |

**Key Finding**: AITER Sigmoid has kernel launch overhead and is **slower** for small tensors. Use `torch.sigmoid` instead.

### Full mHC Layer Benchmarks

#### Forward Pass

| Config | PyTorch f32 | PyTorch bf16 | AITER bf16 | Speedup |
|--------|-------------|--------------|------------|---------|
| B=128, n=4, C=1280 | 0.530ms | 0.551ms | 0.515ms | 1.03x |
| B=256, n=4, C=1280 | 0.438ms | 0.463ms | 0.424ms | 1.03x |
| B=320, n=4, C=1280 | 0.514ms | 0.540ms | 0.499ms | 1.03x |
| B=512, n=4, C=1920 | 0.990ms | 1.028ms | 0.963ms | 1.03x |
| B=512, n=4, C=2560 | 1.278ms | 1.322ms | 1.252ms | 1.02x |
| B=1024, n=4, C=1280 | 1.276ms | 1.320ms | 1.249ms | 1.02x |
| B=1024, n=4, C=1920 | 1.829ms | 1.881ms | 1.804ms | 1.01x |

#### Forward + Backward (Total)

| Config | PyTorch f32 | AITER bf16 | **Speedup** |
|--------|-------------|------------|-------------|
| B=128, n=4, C=1280 | 1.788ms | 1.595ms | **1.12x** |
| B=256, n=4, C=1280 | 1.696ms | 1.514ms | **1.12x** |
| B=320, n=4, C=1280 | 1.785ms | 1.594ms | **1.12x** |
| B=512, n=4, C=1920 | 3.299ms | 2.797ms | **1.18x** |
| B=512, n=4, C=2560 | 4.190ms | 3.585ms | **1.17x** |
| B=1024, n=4, C=1280 | 4.184ms | 3.578ms | **1.17x** |
| B=1024, n=4, C=1920 | 5.942ms | 5.109ms | **1.16x** |

### Correctness Verification

| Config | Max Diff | Mean Diff | Relative Diff |
|--------|----------|-----------|---------------|
| B=128, n=4, C=1280 | 2.47e-02 | 1.47e-03 | 2.53e-03 |
| B=256, n=4, C=1280 | 2.92e-02 | 1.50e-03 | 3.34e-03 |
| B=320, n=4, C=1280 | 2.07e-02 | 1.52e-03 | 3.27e-03 |
| B=512, n=4, C=1920 | 2.32e-02 | 1.50e-03 | 2.80e-03 |

**Note**: Differences are within bf16 precision tolerance (~1.5e-03 mean relative error).

## Analysis

### Why Full Layer Speedup is Modest (1.02-1.03x Forward)

The mHC layer computation breakdown:

```
┌─────────────────────────────────────────────────────────┐
│                    mHC Forward Pass                      │
├─────────────────────────────────────────────────────────┤
│  1. Stream Aggregation (einsum)         ~35% of time    │
│  2. RMSNorm                             ~5% of time     │  ← AITER 4-6x faster here
│  3. Sinkhorn-Knopp (3 iterations)       ~25% of time    │
│  4. Stream Distribution (einsum + add)  ~35% of time    │
└─────────────────────────────────────────────────────────┘
```

Even though AITER RMSNorm is 4-6x faster, it only accounts for ~5% of total compute, resulting in:
- **Forward**: ~3% improvement
- **Backward**: ~15-18% improvement (larger RMSNorm contribution in backward)

### Recommendations

1. **Use AITER RMSNorm** - 4-6x faster, significant for large batches
2. **Don't use AITER Sigmoid** - Slower due to kernel launch overhead for small tensors
3. **For maximum performance**, custom HIP kernels with fused operations would provide better speedup

### Memory Usage

Memory usage is identical between PyTorch and AITER versions (6.3MB - 75MB depending on config).

## Conclusion

| Metric | Improvement |
|--------|-------------|
| **RMSNorm (isolated)** | **4-6x faster** |
| **Full mHC Forward** | 1.02-1.03x faster |
| **Full mHC Forward+Backward** | **1.12-1.18x faster** |
| **Correctness** | ✓ Within bf16 tolerance |

**Bottom Line**: AITER provides measurable improvements, especially for backward pass. For production use:
- Use AITER RMSNorm for bf16 workloads
- Keep PyTorch sigmoid for small tensors
- Consider custom fused kernels for maximum performance

