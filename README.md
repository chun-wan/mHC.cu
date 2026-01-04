# mHC.cu

unofficial CUDA implementation of mHC: Manifold-Constrained Hyper-Connections by DeepSeek-AI

## Installation

```bash
make install      # install PyTorch extension
make install-dev  # install with dev dependencies
```

## Build

```bash
make              # build C++ / CUDA source for all architectures
make CUDA_ARCH=90 # build for specific arch (H100)
make clean        # clean build
```

## Test

```bash
make test         # C++ / CUDA tests
make test-python  # Python tests
```

## Benchmark

```bash
make bench        # run all C++ / CUDA benchmarks
make bench-python # run all Python benchmarks
```

### Pytorch Benchmark Results (benchmarked on H100 SXM5)

Fused mHC vs naive PyTorch mHC implementation (configs from paper Appendix A 
in section A.1):

**Static H Path** (shared H across batch):

| Batch | Hidden | n | Forward | Backward |
|-------|--------|---|---------|----------|
| 320   | 1280   | 4 | 14.5x   | 10.9x    |
| 512   | 1920   | 4 | 11.7x   | 7.9x     |
| 1280  | 2560   | 4 | 8.0x    | 3.7x     |
| 2560  | 1280   | 4 | 7.9x    | 3.6x     |

**Dynamic H Path** (per-batch H values -> this matches paper architecture as presented in Equations 7-9):

| Batch | Hidden | n | Forward | Backward |
|-------|--------|---|---------|----------|
| 320   | 1280   | 4 | 6.3x    | 10.3x    |
| 512   | 1920   | 4 | 5.7x    | 8.2x     |
| 1280  | 2560   | 4 | 3.8x    | 4.5x     |
| 2560  | 1280   | 4 | 3.8x    | 4.4x     |

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
