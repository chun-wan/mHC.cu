"""
mHC.cu - Manifold-Constrained Hyper-Connections for AMD MI300X

This package provides HIP/ROCm implementations of the mHC kernel operations
from DeepSeek-V3, with optional AITER acceleration.

Usage:
    from mhc_aiter import MHCLayer, MHCLayerAITER, MHCLayerDynamic
    
    # Standard HIP implementation
    layer = MHCLayer(hidden_dim=1280, n_streams=4)
    
    # AITER-accelerated (when available)
    layer_aiter = MHCLayerAITER(hidden_dim=1280, n_streams=4)
    
    # Dynamic H parameters
    layer_dynamic = MHCLayerDynamic(hidden_dim=1280, n_streams=4)

AITER repository: https://github.com/ROCm/aiter
"""

__version__ = "0.1.0"

from .aiter_ops import (
    is_aiter_available,
    get_aiter_info,
    create_aiter_ops,
    AITERRMSNorm,
    AITERSigmoid,
    AITERGEMM,
    AITERStreamOps,
    aiter_rmsnorm,
)

from .mhc_aiter import (
    MHCLayer,
    MHCLayerAITER,
    MHCLayerSuperFused,
    MHCLayerDynamic,
    benchmark_mhc,
    mhc_forward_superfused,
)

__all__ = [
    # Version
    "__version__",
    # AITER ops
    "is_aiter_available",
    "get_aiter_info",
    "create_aiter_ops",
    "AITERRMSNorm",
    "AITERSigmoid",
    "AITERGEMM",
    "AITERStreamOps",
    "aiter_rmsnorm",
    # MHC layers
    "MHCLayer",
    "MHCLayerAITER",
    "MHCLayerSuperFused",  # Best performance - 2.35x speedup
    "MHCLayerDynamic",
    "benchmark_mhc",
    "mhc_forward_superfused",
]

