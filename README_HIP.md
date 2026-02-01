# mHC.cu for AMD MI300X (HIP/ROCm)

This is the AMD MI300X port of the mHC (Manifold-Constrained Hyper-Connections) CUDA kernels, using HIP/ROCm for AMD GPU acceleration.

## Overview

mHC implements the kernel operations from the DeepSeek-AI paper "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880). This port converts the original NVIDIA CUDA implementation to AMD HIP for use on MI300X and other CDNA3 GPUs.

### Key Changes from CUDA Version

| CUDA | HIP |
|------|-----|
| `cuda_runtime.h` | `hip/hip_runtime.h` |
| `nv_bfloat16` | `hip_bfloat16` |
| `cuBLASLt` | `hipBLASLt` |
| Warp size: 32 | Wavefront size: 64 |
| `cudaLaunchKernelEx` | `hipLaunchKernelGGL` |
| PDL (SM_90+) | Not available |

## Requirements

- **ROCm 6.0+** (tested with ROCm 6.1)
- **AMD MI300X** (gfx942) or compatible CDNA3 GPU
- **PyTorch 2.0+** with ROCm support
- **hipBLASLt** for matrix operations
- **Python 3.10+**
- **AITER** (optional, for additional acceleration) - [https://github.com/ROCm/aiter](https://github.com/ROCm/aiter)

## Installation

### 1. Install ROCm

Follow the [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html).

```bash
# Ubuntu 22.04
sudo apt update
sudo apt install rocm-dkms rocm-libs rocm-dev
```

### 2. Install PyTorch with ROCm

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### 3. Install mHC

```bash
# Clone repository
git clone https://github.com/chun-wan/mHC.cu
cd mHC.cu

# Install using HIP build script
./scripts/build_hip.sh --install

# Or using make
make -f Makefile.hip install
```

## Usage

```python
import torch
from mhc import MHCLayer

# Dynamic H path (default)
layer = MHCLayer(hidden_dim=4096, expansion_rate=4).cuda()
x = torch.randn(8, 4, 4096, device="cuda")  # [B, n, C]
y = layer(x)  # [B, n, C]

# Static H path (faster for inference)
layer_static = MHCLayer(hidden_dim=4096, expansion_rate=4, use_dynamic_h=False).cuda()
y = layer_static(x)
```

## Architecture Support

| GPU | Architecture | Supported |
|-----|-------------|-----------|
| MI300X | gfx942 | ✅ Primary target |
| MI300A | gfx942 | ✅ |
| MI250X | gfx90a | ✅ |
| MI210 | gfx90a | ✅ |
| MI100 | gfx908 | ⚠️ Untested |

Set architecture via environment variable:

```bash
export HIP_ARCH=gfx942  # MI300X (default)
export HIP_ARCH=gfx90a  # MI250X
```

## Building

### Using Make

```bash
# Build for MI300X (default)
make -f Makefile.hip

# Build for MI250X
make -f Makefile.hip HIP_ARCH=gfx90a

# Install Python extension
make -f Makefile.hip install

# Run tests
make -f Makefile.hip test

# Run benchmarks
make -f Makefile.hip bench
```

### Using Build Script

```bash
# Full build with tests and benchmarks
./scripts/build_hip.sh --all

# Just install
./scripts/build_hip.sh --install

# Run benchmarks
./scripts/build_hip.sh --bench
```

## AITER Integration

[AITER (AI Tensor Engine for ROCm)](https://github.com/ROCm/aiter) provides additional acceleration for mHC kernels. When available, AITER's optimized operators are used for RMSNorm, GEMM, and element-wise operations.

### Installing AITER

```bash
# Clone AITER repository
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter

# Install in development mode
python3 setup.py develop
```

### Using AITER with mHC

```python
from mhc_aiter import MHCLayerAITER

# AITER-accelerated layer
layer = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=True)

# Check AITER status
from aiter_ops import is_aiter_available, get_aiter_info
print(f"AITER available: {is_aiter_available()}")
print(f"AITER info: {get_aiter_info()}")
```

### Benchmarking AITER vs HIP

```bash
# Run benchmark comparison
python scripts/benchmark_aiter.py --batch-sizes 128,256,512 --hidden-dims 1280,1920

# Save results to JSON
python scripts/benchmark_aiter.py --output benchmark_results.json
```

For detailed AITER documentation, see [docs/AITER_INTEGRATION.md](docs/AITER_INTEGRATION.md).

## Performance Notes

### MI300X Optimizations

1. **Wavefront Size**: AMD uses 64-thread wavefronts vs NVIDIA's 32-thread warps. Kernels are optimized accordingly.

2. **No PDL**: NVIDIA's Programmatic Dependent Launch (PDL) for H100/B200 is not available. Pipelining uses standard stream synchronization.

3. **hipBLASLt**: Uses AMD's hipBLASLt for matrix multiplications, which is optimized for MI300X.

4. **L2 Cache**: MI300X has 256MB L3 cache. L2 flusher adjusted for benchmarking.

5. **AITER**: When available, AITER provides 10-30% additional speedup through optimized Triton/CK kernels.

### Expected Performance

Based on the original CUDA benchmarks on H100, expect similar relative speedups on MI300X:

| Config | Batch | Hidden | n | Expected Speedup |
|--------|-------|--------|---|------------------|
| Static | 320 | 1280 | 4 | ~10-13x forward |
| Static | 512 | 1920 | 4 | ~7-9x forward |
| Dynamic | 320 | 1280 | 4 | ~6-7x forward |

*Actual results may vary. Run benchmarks to get accurate numbers for your workload.*

## File Structure

```
mHC.cu/
├── src/
│   ├── csrc/
│   │   ├── include/
│   │   │   ├── mhc_types_hip.h      # HIP type definitions
│   │   │   ├── utils_hip.h          # HIP utilities
│   │   │   └── aiter_compat.h       # AITER compatibility layer
│   │   └── kernels/
│   │       ├── rmsnorm_hip.h        # RMSNorm kernels
│   │       ├── sinkhorn_knopp_hip.h # Sinkhorn-Knopp kernels
│   │       ├── stream_ops_hip.h     # Stream operations
│   │       ├── fused_rmsnorm_matmul_hip.h
│   │       └── mhc_layer_hip.h      # Main MHC layer
│   └── python/
│       ├── bindings_hip.cpp         # Python bindings for HIP
│       ├── aiter_ops.py             # AITER operators wrapper
│       ├── mhc_aiter.py             # AITER-enabled MHC layers
│       └── mhc/                     # Python package
├── docs/
│   └── AITER_INTEGRATION.md         # AITER integration guide
├── scripts/
│   ├── build_hip.sh                 # Build script
│   └── benchmark_aiter.py           # AITER benchmark script
├── Makefile.hip                     # HIP-specific Makefile
├── setup_hip.py                     # HIP-specific setup.py
└── README_HIP.md                    # This file
```

## Troubleshooting

### Common Issues

1. **ROCm not found**
   ```bash
   export ROCM_PATH=/opt/rocm
   ```

2. **hipBLASLt not available**
   ```bash
   sudo apt install hipblaslt
   ```

3. **PyTorch not detecting ROCm**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.version.hip)          # Should show HIP version
   ```

4. **Wrong GPU architecture**
   ```bash
   rocm-smi --showproductname  # Check your GPU
   export HIP_ARCH=gfx942      # Set correct arch
   ```

## References

- **Paper**: [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- **Original CUDA Implementation**: [AndreSlavescu/mHC.cu](https://github.com/AndreSlavescu/mHC.cu)
- **AITER**: [ROCm/aiter](https://github.com/ROCm/aiter) - AI Tensor Engine for ROCm
- **ROCm Documentation**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)
- **HIP Programming Guide**: [ROCm HIP Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## License

MIT License - see [LICENSE](LICENSE) file.

