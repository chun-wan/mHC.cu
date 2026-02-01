"""
mHC (Manifold-Constrained Hyper-Connections) Integration for vLLM/DeepSeek-V3.2

This module provides mHC layer integration for DeepSeek-V3.2 models in vLLM.
Based on: https://github.com/chun-wan/mHC.cu

mHC replaces/enhances residual connections with multi-stream manifold-constrained
connections, providing better gradient flow and representation learning.

Reference: mHC: Manifold-Constrained Hyper-Connections (https://arxiv.org/abs/2512.24880)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import AITER for ROCm acceleration
try:
    from aiter import rmsnorm_fwd
    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False


def sinkhorn_knopp(inp: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for computing doubly stochastic matrices.
    
    Args:
        inp: Input matrix [n, n] or [B, n, n]
        iters: Number of iterations
        eps: Epsilon for numerical stability
        
    Returns:
        Doubly stochastic matrix
    """
    P = inp.clone()
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    return P


@torch.jit.script
def mhc_residual_forward(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    H_pre: torch.Tensor,
    H_post: torch.Tensor,
    H_res: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    sinkhorn_iters: int,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mHC-enhanced residual connection forward pass.
    
    This replaces the standard residual: out = hidden_states + residual
    With mHC multi-stream mixing.
    
    Args:
        hidden_states: Current layer output [B, C]
        residual: Accumulated residual [B, C]
        H_pre: Pre-aggregation weights [2]
        H_post: Post-distribution weights [2]
        H_res: Residual mixing matrix [2, 2]
        rmsnorm_weight: RMSNorm weight [C]
        sinkhorn_iters: Number of Sinkhorn iterations
        eps: Epsilon for numerical stability
        
    Returns:
        Tuple of (new_hidden_states, new_residual)
    """
    B, C = hidden_states.shape
    
    # Create 2-stream representation: [hidden_states, residual]
    # Shape: [B, 2, C]
    x = torch.stack([hidden_states.float(), residual.float()], dim=1)
    
    # 1. Stream aggregation with sigmoid activation
    H_pre_activated = torch.sigmoid(H_pre)  # [2]
    x_agg = torch.einsum('bnc,n->bc', x, H_pre_activated)  # [B, C]
    
    # 2. RMSNorm
    rms = torch.sqrt((x_agg ** 2).mean(dim=-1, keepdim=True) + eps)
    y_norm = (x_agg / rms) * rmsnorm_weight.float()
    
    # 3. Sinkhorn-Knopp for mixing matrix
    P = torch.exp(H_res)
    for _ in range(sinkhorn_iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    M = P
    
    # 4. Stream distribution with mixing and addition
    H_post_activated = 2.0 * torch.sigmoid(H_post)  # [2]
    
    # Mix streams: [B, 2, C]
    M_expanded = M.unsqueeze(0).expand(B, -1, -1)
    mixed = torch.bmm(M_expanded, x)  # [B, 2, C]
    output = mixed + H_post_activated.view(1, 2, 1) * y_norm.unsqueeze(1)
    
    # Extract new hidden_states and residual
    new_hidden_states = output[:, 0, :]  # [B, C]
    new_residual = output[:, 1, :]  # [B, C]
    
    return new_hidden_states, new_residual


class MHCResidualConnection(nn.Module):
    """
    mHC-enhanced Residual Connection for transformer layers.
    
    This module replaces standard residual connections with mHC's
    manifold-constrained multi-stream mixing mechanism.
    
    Args:
        hidden_size: Model hidden dimension
        n_streams: Number of streams (default: 2 for hidden + residual)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        eps: Epsilon for numerical stability
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_streams: int = 2,
        sinkhorn_iters: int = 3,
        eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        
        # Learnable parameters
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_size))
        self.H_pre = nn.Parameter(torch.zeros(n_streams))
        self.H_post = nn.Parameter(torch.zeros(n_streams))
        self.H_res = nn.Parameter(torch.eye(n_streams))
        
        # Initialize for identity-like behavior initially
        with torch.no_grad():
            self.H_pre[0] = 0.0  # hidden_states weight
            self.H_pre[1] = 0.0  # residual weight
            self.H_post[0] = 0.0
            self.H_post[1] = 0.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mHC residual connection.
        
        Args:
            hidden_states: Current layer output [B, C] or [B, S, C]
            residual: Accumulated residual [B, C] or [B, S, C]
            
        Returns:
            Tuple of (new_hidden_states, new_residual)
        """
        # Handle sequence dimension
        if hidden_states.dim() == 3:
            B, S, C = hidden_states.shape
            hidden_states = hidden_states.reshape(B * S, C)
            residual = residual.reshape(B * S, C)
            reshape_back = True
        else:
            B, C = hidden_states.shape
            S = 1
            reshape_back = False
        
        # Apply mHC
        new_hidden, new_residual = mhc_residual_forward(
            hidden_states,
            residual,
            self.H_pre,
            self.H_post,
            self.H_res,
            self.rmsnorm_weight,
            self.sinkhorn_iters,
            self.eps
        )
        
        # Restore shape if needed
        if reshape_back:
            new_hidden = new_hidden.reshape(B, S, C)
            new_residual = new_residual.reshape(B, S, C)
        
        return new_hidden.to(hidden_states.dtype), new_residual.to(residual.dtype)


class MHCDecoderLayerWrapper(nn.Module):
    """
    Wrapper to add mHC connections to existing DeepSeek decoder layers.
    
    This wrapper can be applied to DeepseekV2DecoderLayer to enhance
    its residual connections with mHC.
    
    Args:
        decoder_layer: Original decoder layer
        hidden_size: Model hidden dimension
        use_mhc_attention: Whether to use mHC for attention residual
        use_mhc_mlp: Whether to use mHC for MLP residual
    """
    
    def __init__(
        self,
        decoder_layer: nn.Module,
        hidden_size: int,
        use_mhc_attention: bool = True,
        use_mhc_mlp: bool = True
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.hidden_size = hidden_size
        
        if use_mhc_attention:
            self.mhc_attention = MHCResidualConnection(hidden_size)
        else:
            self.mhc_attention = None
            
        if use_mhc_mlp:
            self.mhc_mlp = MHCResidualConnection(hidden_size)
        else:
            self.mhc_mlp = None
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with mHC-enhanced residual connections.
        """
        # Use original layer's forward but intercept residual handling
        # This is a simplified version - full implementation would need
        # to modify the internal residual handling
        
        output, residual = self.decoder_layer(positions, hidden_states, residual, **kwargs)
        
        # Apply mHC to final output if enabled
        if self.mhc_mlp is not None and residual is not None:
            output, residual = self.mhc_mlp(output, residual)
        
        return output, residual


def apply_mhc_to_model(model: nn.Module, hidden_size: int) -> nn.Module:
    """
    Apply mHC to all decoder layers in a model.
    
    Args:
        model: The model to enhance
        hidden_size: Hidden dimension
        
    Returns:
        Enhanced model with mHC connections
    """
    for name, module in model.named_modules():
        if 'DecoderLayer' in type(module).__name__:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            wrapped = MHCDecoderLayerWrapper(module, hidden_size)
            setattr(parent, child_name, wrapped)
    
    return model


# Benchmark utilities
def benchmark_mhc_residual(
    batch_size: int = 32,
    seq_len: int = 1024,
    hidden_size: int = 7168,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    device: str = "cuda"
) -> dict:
    """
    Benchmark mHC residual connection vs standard residual.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        device: Device to use
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Setup
    mhc_layer = MHCResidualConnection(hidden_size).to(device)
    hidden_states = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    
    # Warmup mHC
    for _ in range(warmup_iters):
        _ = mhc_layer(hidden_states.clone(), residual.clone())
        torch.cuda.synchronize()
    
    # Benchmark mHC
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = mhc_layer(hidden_states, residual)
    torch.cuda.synchronize()
    mhc_time = (time.perf_counter() - start) * 1000 / bench_iters
    
    # Benchmark standard residual
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = hidden_states + residual
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) * 1000 / bench_iters
    
    return {
        "mhc_time_ms": mhc_time,
        "standard_time_ms": std_time,
        "overhead": mhc_time / std_time,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
    }


if __name__ == "__main__":
    print("mHC-vLLM Integration Module")
    print("=" * 50)
    
    # Test mHC residual connection
    print("\n[TEST] MHCResidualConnection")
    mhc = MHCResidualConnection(hidden_size=7168)
    mhc = mhc.cuda()
    
    hidden = torch.randn(4, 1024, 7168, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(4, 1024, 7168, device="cuda", dtype=torch.bfloat16)
    
    new_hidden, new_residual = mhc(hidden, residual)
    print(f"  Input shapes: hidden={hidden.shape}, residual={residual.shape}")
    print(f"  Output shapes: hidden={new_hidden.shape}, residual={new_residual.shape}")
    
    # Benchmark
    print("\n[BENCHMARK] mHC vs Standard Residual")
    results = benchmark_mhc_residual(batch_size=4, seq_len=1024, hidden_size=7168)
    print(f"  mHC time: {results['mhc_time_ms']:.3f} ms")
    print(f"  Standard time: {results['standard_time_ms']:.3f} ms")
    print(f"  Overhead: {results['overhead']:.2f}x")

