# AITER Integration for mHC.cu

This document describes how to use [AITER (AI Tensor Engine for ROCm)](https://github.com/ROCm/aiter) to accelerate mHC kernels on AMD MI300X GPUs.

## What is AITER?

AITER is AMD's centralized repository for high-performance AI operators optimized for ROCm. It provides:

- **RMSNorm**: Optimized root mean square normalization
- **GEMM**: High-performance matrix multiplications using CK (Composable Kernel)
- **Element-wise ops**: Optimized +, -, *, /, sigmoid, etc.
- **MHA/MLA**: Multi-head attention implementations
- **FusedMoE**: Mixture of Experts kernels

## Benefits of AITER Integration

| Feature | Custom HIP Kernels | AITER |
|---------|-------------------|-------|
| RMSNorm | Hand-tuned wavefront 64 | Triton/CK optimized |
| GEMM | hipBLASLt | CK-based, auto-tuned |
| Memory Layout | Manual optimization | Auto-optimized |
| Kernel Fusion | Manual | Automatic |

Expected speedup with AITER: **10-30%** depending on workload.

## Installation

### 1. Install AITER

```bash
# Clone AITER repository
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter

# Install in development mode
python3 setup.py develop

# Verify installation
python3 -c "import aiter; print('AITER installed successfully')"
```

### 2. Install mHC with AITER Support

```bash
cd /path/to/mHC.cu

# Option A: Auto-detect AITER
pip install -e .

# Option B: Explicitly enable AITER
MHC_USE_AITER=1 pip install -e .

# Option C: Disable AITER (use custom HIP kernels only)
pip install -e . --no-aiter
```

### 3. Verify Installation

```python
from mhc_aiter import MHCLayerAITER
from aiter_ops import is_aiter_available, get_aiter_info

print(f"AITER available: {is_aiter_available()}")
print(f"AITER info: {get_aiter_info()}")

# Create AITER-accelerated layer
layer = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=True)
```

## Usage

### Basic Usage

```python
import torch
from mhc_aiter import MHCLayer, MHCLayerAITER

# Standard HIP implementation
layer_hip = MHCLayer(hidden_dim=1280, n_streams=4)

# AITER-accelerated implementation
layer_aiter = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=True)

# Input tensor [B, n, C]
x = torch.randn(32, 4, 1280, device='cuda', dtype=torch.float32)

# Forward pass
output_hip = layer_hip(x)
output_aiter = layer_aiter(x)
```

### Dynamic H (Input-Dependent)

```python
from mhc_aiter import MHCLayerDynamic

layer_dynamic = MHCLayerDynamic(
    hidden_dim=1280,
    n_streams=4,
    alpha_pre=0.1,
    alpha_post=0.1,
    alpha_res=0.1,
    use_aiter=True
)

output = layer_dynamic(x)
```

### Backend Selection

```python
from mhc_aiter import MHCLayerAITER

# Force AITER (fails if not available)
layer = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=True, fallback_to_hip=False)

# AITER with HIP fallback
layer = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=True, fallback_to_hip=True)

# Force custom HIP kernels
layer = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=False)
```

## Benchmarking

Run the benchmark script to compare AITER vs non-AITER performance:

```bash
# Full benchmark
python scripts/benchmark_aiter.py

# Custom configurations
python scripts/benchmark_aiter.py \
    --batch-sizes 128,256,512 \
    --hidden-dims 1280,1920 \
    --n-streams 4 \
    --output results.json

# Quick test
python scripts/benchmark_aiter.py \
    --batch-sizes 128 \
    --hidden-dims 1280 \
    --warmup 5 \
    --iters 50
```

### Expected Results (MI300X)

| Config | PyTorch | HIP Kernels | AITER | Speedup |
|--------|---------|-------------|-------|---------|
| B=320, C=1280, n=4 | 2.5ms | 0.25ms | 0.20ms | 12.5x |
| B=512, C=1920, n=4 | 5.8ms | 0.82ms | 0.65ms | 8.9x |
| B=256, C=2560, n=4 | 4.2ms | 0.55ms | 0.42ms | 10.0x |

## Troubleshooting

### AITER not detected

```python
from aiter_ops import is_aiter_available, get_aiter_info

if not is_aiter_available():
    print("AITER not available. Check installation:")
    print("  1. Verify AITER is installed: python -c 'import aiter'")
    print("  2. Rebuild mHC: pip install -e . --force-reinstall")
```

### Performance regression with AITER

If AITER is slower than custom HIP kernels:

1. Check kernel launch overhead for small batch sizes
2. Verify AITER is using the correct architecture (gfx942 for MI300X)
3. Try different AITER configurations

```python
# Disable AITER for specific operations
from mhc_aiter import MHCLayerAITER

layer = MHCLayerAITER(
    hidden_dim=1280,
    n_streams=4,
    use_aiter=False  # Use HIP kernels instead
)
```

### Build errors with AITER

```bash
# Clean build
rm -rf build/ dist/ *.egg-info
pip cache purge

# Rebuild with verbose output
MHC_USE_AITER=1 pip install -e . -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        mHC Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Stream    │    │   RMSNorm   │    │  Sinkhorn   │          │
│  │ Aggregation │───▶│             │───▶│   Knopp     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        │                  │                  │                   │
│        ▼                  ▼                  ▼                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Custom HIP │    │ Custom HIP/ │    │  Custom HIP │          │
│  │   Kernel    │    │   AITER     │    │   Kernel    │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐                              │
│  │   Stream    │    │  Mix Add    │                              │
│  │Distribution │───▶│             │───▶ Output                   │
│  └─────────────┘    └─────────────┘                              │
│        │                  │                                      │
│        ▼                  ▼                                      │
│  ┌─────────────┐    ┌─────────────┐                              │
│  │  Custom HIP │    │ Custom HIP/ │                              │
│  │   Kernel    │    │   AITER     │                              │
│  └─────────────┘    └─────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Contributing

To add new AITER operator integrations:

1. Check if AITER provides the operator: https://github.com/ROCm/aiter
2. Add wrapper in `src/python/aiter_ops.py`
3. Integrate in `src/python/mhc_aiter.py`
4. Update benchmarks in `scripts/benchmark_aiter.py`
5. Document in this file

## References

- [AITER GitHub](https://github.com/ROCm/aiter)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [mHC Paper (DeepSeek-V3)](https://arxiv.org/abs/2501.12948)
- [Original CUDA Implementation](https://github.com/andreslavescu/mHC.cu)

