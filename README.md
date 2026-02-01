# mHC.cu

Unofficial CUDA/HIP implementation of mHC: Manifold-Constrained Hyper-Connections by DeepSeek-AI

**Supports both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs**

## Supported Platforms

| Platform | GPUs | Status |
|----------|------|--------|
| NVIDIA CUDA | H100, B200, A100 | ✅ Full support |
| AMD ROCm/HIP | MI300X, MI300A, MI250X | ✅ Full support |

## Running on Modal (NVIDIA)

Once the image builds the first time, it will be cached and will not require a rebuild.

### Supported GPUs

```bash
--gpu h100 # H100 80GB HBM3 SXM5 model
--gpu b200 # B200
```

### Benchmark

```bash
# python bench
modal run runmodal.py --gpu h100 --mode bench --scope python

# c++ / cuda bench
modal run runmodal.py --gpu h100 --mode bench --scope native

# run all benches
modal run runmodal.py --gpu h100 --mode bench --scope all
```

### Test

```bash
# python bench
modal run runmodal.py --gpu h100 --mode test --scope python

# cpp / cuda bench
modal run runmodal.py --gpu h100 --mode test --scope native

# run all benches
modal run runmodal.py --gpu h100 --mode test --scope all
```

## Local 

### Installation

```bash
make install      # install PyTorch extension
make install-dev  # install with dev dependencies
```

### Build

```bash
make              # build C++ / CUDA source for all architectures
make CUDA_ARCH=90 # build for specific arch (H100)
make clean        # clean build
```

### Test

```bash
make test         # C++ / CUDA tests
make test-python  # Python tests
```

### Benchmark

```bash
make bench        # run all C++ / CUDA benchmarks
make bench-python # run all Python benchmarks
```

---

## AMD ROCm/HIP (MI300X)

Full HIP/ROCm support for AMD Instinct GPUs with [AITER](https://github.com/ROCm/aiter) integration.

### Requirements

- ROCm 6.0+ (tested with ROCm 7.2)
- AMD MI300X/MI300A/MI250X GPU
- PyTorch 2.0+ with ROCm support

### Installation

```bash
# Install PyTorch extension for AMD
pip install -e . --config-settings="setup-file=setup_hip.py"

# Or use the HIP Makefile
make -f Makefile.hip install
```

### Build

```bash
make -f Makefile.hip                    # build for MI300X (default)
make -f Makefile.hip HIP_ARCH=gfx90a    # build for MI250X
make -f Makefile.hip clean              # clean build
```

### Build Script

```bash
./scripts/build_hip.sh --install        # build and install
./scripts/build_hip.sh --test           # run tests
./scripts/build_hip.sh --bench          # run benchmarks
```

### AMD Benchmark Results (MI300X)

#### SuperFused Implementation (Best Performance)

| Batch | Hidden | n  | Non-Fused | SuperFused | Speedup |
|-------|--------|----|-----------|------------|---------|
| 128   | 1280   | 4  | 0.534ms   | 0.240ms    | **2.22x** |
| 256   | 1280   | 4  | 0.442ms   | 0.244ms    | **1.81x** |
| 512   | 1920   | 4  | 1.014ms   | 0.383ms    | **2.65x** |
| 512   | 2560   | 4  | 1.308ms   | 0.471ms    | **2.77x** |
| 1024  | 1920   | 4  | 1.871ms   | 0.692ms    | **2.70x** |

#### AITER RMSNorm (Isolated)

| Size | PyTorch | AITER | Speedup |
|------|---------|-------|---------|
| (256, 1280) | 0.038ms | 0.010ms | **3.94x** |
| (512, 1920) | 0.044ms | 0.009ms | **4.70x** |
| (1024, 2560) | 0.060ms | 0.010ms | **6.08x** |

### AMD Usage

```python
import torch
from mhc_aiter import MHCLayerSuperFused  # 2.35x faster

# SuperFused implementation (recommended)
layer = MHCLayerSuperFused(hidden_dim=1280, n_streams=4).cuda()
x = torch.randn(256, 4, 1280, device="cuda")
y = layer(x)

# Or with AITER acceleration
from mhc_aiter import MHCLayerAITER
layer_aiter = MHCLayerAITER(hidden_dim=1280, n_streams=4, use_aiter=True).cuda()
y = layer_aiter(x)
```

### AMD Documentation

- [README_HIP.md](README_HIP.md) - Full HIP/ROCm documentation
- [docs/AITER_INTEGRATION.md](docs/AITER_INTEGRATION.md) - AITER integration guide
- [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) - Detailed benchmarks
- [docs/FUSED_OPS_RESULTS.md](docs/FUSED_OPS_RESULTS.md) - Fusion optimization results

---

### Pytorch Benchmark Results (benchmarked on H100 SXM5)

Fused mHC vs naive PyTorch mHC implementation (configs from paper Appendix A 
in section A.1):

**Static H Path** (shared H across batch):

| Batch | Hidden | n  | Forward | Backward |
|-------|--------|----|---------|----------|
| 320   | 1280   | 4  | 13.2x   | 11.1x    |
| 512   | 1920   | 4  | 9.0x    | 7.7x     |
| 1280  | 2560   | 4  | 5.1x    | 3.6x     |
| 2560  | 1280   | 4  | 5.0x    | 3.5x     |
| 128   | 1280   | 8  | 13.6x   | 11.5x    |
| 256   | 1280   | 8  | 10.3x   | 9.8x     |
| 32    | 1280   | 32 | 5.8x    | 2.9x     |
| 64    | 1280   | 32 | 4.7x    | 2.2x     |
| 128   | 1280   | 32 | 3.5x    | 1.5x     |

**Dynamic H Path** (per-batch H values computed via Equations 7-9 from paper):

| Batch | Hidden | n  | Forward | Backward |
|-------|--------|----|---------|----------|
| 320   | 1280   | 4  | 6.7x    | 11.0x    |
| 512   | 1920   | 4  | 6.8x    | 9.0x     |
| 1280  | 2560   | 4  | 4.6x    | 5.1x     |
| 2560  | 1280   | 4  | 4.6x    | 5.0x     |
| 128   | 1280   | 8  | 6.4x    | 11.2x    |
| 256   | 1280   | 8  | 6.1x    | 10.6x    |
| 32    | 1280   | 32 | 1.9x    | 3.1x     |
| 64    | 1280   | 32 | 1.8x    | 2.5x     |
| 128   | 1280   | 32 | 1.7x    | 1.9x     |


## Format

```bash
make format       # clang-format + python black formatting
```

## Usage

```python
import torch
from mhc import MHCLayer

# Dynamic H path (default, matches paper architecture)
# H values are computed from x via learned projections
layer = MHCLayer(hidden_dim=4096, expansion_rate=4).cuda()
x = torch.randn(8, 4, 4096, device="cuda")  # [B, n, C]
y = layer(x)  # [B, n, C]

# Static H path (shared H across batch, faster for inference)
layer_static = MHCLayer(hidden_dim=4096, expansion_rate=4, use_dynamic_h=False).cuda()
y = layer_static(x)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for directions on how to contribute, including testing, formatting, and code style requirements.

## Paper

**mHC: Manifold-Constrained Hyper-Connections**  
https://arxiv.org/abs/2512.24880

DeepSeek-AI

## Citation

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```
