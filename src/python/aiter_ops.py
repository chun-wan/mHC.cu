"""
AITER (AI Tensor Engine for ROCm) integration for mHC kernels.

This module provides a unified interface that uses AITER's optimized operators
when available, falling back to custom HIP kernels otherwise.

AITER repository: https://github.com/ROCm/aiter
"""

import torch
from typing import Optional, Tuple, Callable
from functools import wraps

# Global flags for AITER availability
_AITER_AVAILABLE = False
_AITER_RMSNORM = None
_AITER_GEMM = None
_AITER_SIGMOID = None

def _try_import_aiter():
    """Try to import AITER components."""
    global _AITER_AVAILABLE, _AITER_RMSNORM, _AITER_GEMM, _AITER_SIGMOID
    
    try:
        # Import AITER's optimized RMSNorm
        from aiter import rmsnorm_fwd as aiter_rmsnorm_fwd
        from aiter import rmsnorm_bwd as aiter_rmsnorm_bwd
        _AITER_RMSNORM = (aiter_rmsnorm_fwd, aiter_rmsnorm_bwd)
        
        # Import AITER's optimized GEMM
        from aiter import gemm as aiter_gemm
        _AITER_GEMM = aiter_gemm
        
        # Import AITER's optimized sigmoid
        from aiter import sigmoid as aiter_sigmoid
        _AITER_SIGMOID = aiter_sigmoid
        
        _AITER_AVAILABLE = True
        print("[mHC] AITER operators loaded successfully")
        return True
    except ImportError as e:
        print(f"[mHC] AITER not available, using custom HIP kernels: {e}")
        return False

# Try to import at module load time
_try_import_aiter()


def is_aiter_available() -> bool:
    """Check if AITER is available."""
    return _AITER_AVAILABLE


def get_aiter_info() -> dict:
    """Get information about available AITER components."""
    return {
        "available": _AITER_AVAILABLE,
        "rmsnorm": _AITER_RMSNORM is not None,
        "gemm": _AITER_GEMM is not None,
        "sigmoid": _AITER_SIGMOID is not None,
    }


class AITERRMSNorm:
    """
    AITER-optimized RMSNorm wrapper.
    
    Falls back to custom HIP kernel if AITER is not available.
    """
    
    def __init__(self, use_aiter: bool = True):
        self.use_aiter = use_aiter and _AITER_RMSNORM is not None
        self._custom_fwd = None
        self._custom_bwd = None
    
    def set_custom_impl(self, fwd_fn: Callable, bwd_fn: Callable):
        """Set custom implementation for fallback."""
        self._custom_fwd = fwd_fn
        self._custom_bwd = bwd_fn
    
    def forward(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RMSNorm forward pass.
        
        Args:
            inp: Input tensor [B, C] in bf16
            weight: Weight tensor [C] in bf16
            eps: Epsilon for numerical stability
            
        Returns:
            Tuple of (normalized output, rms values)
        """
        if self.use_aiter and _AITER_RMSNORM is not None:
            aiter_fwd, _ = _AITER_RMSNORM
            # AITER rmsnorm signature may differ, adapt as needed
            try:
                out = aiter_fwd(inp, weight, eps)
                # AITER may not return rms, compute it if needed
                rms = torch.sqrt((inp.float() ** 2).mean(dim=-1) + eps)
                return out, rms
            except Exception as e:
                print(f"[mHC] AITER RMSNorm forward failed, using fallback: {e}")
        
        if self._custom_fwd is not None:
            return self._custom_fwd(inp, weight, eps)
        
        raise RuntimeError("No RMSNorm implementation available")
    
    def backward(
        self,
        grad: torch.Tensor,
        inp: torch.Tensor,
        weight: torch.Tensor,
        rms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RMSNorm backward pass.
        
        Args:
            grad: Gradient from upstream
            inp: Original input
            weight: Weight tensor
            rms: RMS values from forward pass
            
        Returns:
            Tuple of (d_inp, d_weight)
        """
        if self.use_aiter and _AITER_RMSNORM is not None:
            _, aiter_bwd = _AITER_RMSNORM
            try:
                return aiter_bwd(grad, inp, weight, rms)
            except Exception as e:
                print(f"[mHC] AITER RMSNorm backward failed, using fallback: {e}")
        
        if self._custom_bwd is not None:
            return self._custom_bwd(grad, inp, weight, rms)
        
        raise RuntimeError("No RMSNorm backward implementation available")


class AITERSigmoid:
    """
    AITER-optimized Sigmoid wrapper.
    
    Falls back to torch.sigmoid if AITER is not available.
    """
    
    def __init__(self, use_aiter: bool = True):
        self.use_aiter = use_aiter and _AITER_SIGMOID is not None
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid activation."""
        if self.use_aiter and _AITER_SIGMOID is not None:
            try:
                return _AITER_SIGMOID(x)
            except Exception as e:
                print(f"[mHC] AITER sigmoid failed, using torch.sigmoid: {e}")
        
        return torch.sigmoid(x)


class AITERGEMM:
    """
    AITER-optimized GEMM wrapper.
    
    Falls back to torch.matmul if AITER is not available.
    """
    
    def __init__(self, use_aiter: bool = True):
        self.use_aiter = use_aiter and _AITER_GEMM is not None
    
    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
        C: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        General Matrix Multiplication: D = alpha * A @ B + beta * C
        
        Args:
            A: First matrix
            B: Second matrix
            alpha: Scalar multiplier for A @ B
            beta: Scalar multiplier for C
            C: Optional accumulator matrix
            
        Returns:
            Result matrix D
        """
        if self.use_aiter and _AITER_GEMM is not None:
            try:
                if C is not None and beta != 0.0:
                    return _AITER_GEMM(A, B, alpha=alpha, beta=beta, C=C)
                else:
                    result = _AITER_GEMM(A, B)
                    if alpha != 1.0:
                        result = result * alpha
                    if C is not None and beta != 0.0:
                        result = result + beta * C
                    return result
            except Exception as e:
                print(f"[mHC] AITER GEMM failed, using torch.matmul: {e}")
        
        # Fallback to PyTorch
        result = alpha * torch.matmul(A, B)
        if C is not None and beta != 0.0:
            result = result + beta * C
        return result


class AITERStreamOps:
    """
    AITER-optimized stream operations for mHC.
    
    Combines element-wise operations with optimized AITER primitives.
    """
    
    def __init__(self, use_aiter: bool = True):
        self.use_aiter = use_aiter and _AITER_AVAILABLE
        self.gemm = AITERGEMM(use_aiter)
        self.sigmoid = AITERSigmoid(use_aiter)
    
    def stream_aggregate_bf16_fused_sigmoid(
        self,
        inp: torch.Tensor,
        H_pre_raw: torch.Tensor,
        B: int,
        n: int,
        C: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused stream aggregation with sigmoid activation.
        
        Args:
            inp: Input tensor [B, n, C] in float32
            H_pre_raw: Pre-activation H tensor [n] in float32
            B: Batch size
            n: Number of streams
            C: Channel dimension
            
        Returns:
            Tuple of (aggregated output [B, C] in bf16, activated H_pre [n])
        """
        # Apply sigmoid activation to H_pre
        H_pre_activated = self.sigmoid(H_pre_raw)
        
        # Weighted sum: out[b, c] = sum_i(H_pre[i] * inp[b, i, c])
        # This is essentially a batched matrix-vector multiplication
        # inp: [B, n, C] -> [B, C, n] for matmul
        # H_pre: [n] -> [n, 1]
        inp_transposed = inp.permute(0, 2, 1)  # [B, C, n]
        H_pre_expanded = H_pre_activated.unsqueeze(1)  # [n, 1]
        
        # [B, C, n] @ [n, 1] -> [B, C, 1] -> [B, C]
        out = torch.bmm(inp_transposed, H_pre_expanded.expand(B, n, 1)).squeeze(-1)
        
        return out.to(torch.bfloat16), H_pre_activated
    
    def stream_distribute_mix_add_fused(
        self,
        x_inp: torch.Tensor,
        y_norm: torch.Tensor,
        H_post_raw: torch.Tensor,
        M: torch.Tensor,
        B: int,
        n: int,
        C: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused stream distribution with mixing and addition.
        
        Args:
            x_inp: Input tensor [B, n, C] in float32
            y_norm: Normalized y tensor [B, C] in bf16
            H_post_raw: Post-activation H tensor [n] in float32
            M: Mixing matrix [n, n] in float32
            B: Batch size
            n: Number of streams
            C: Channel dimension
            
        Returns:
            Tuple of (output [B, n, C] in float32, activated H_post [n])
        """
        # Apply 2*sigmoid activation to H_post
        H_post_activated = 2.0 * self.sigmoid(H_post_raw)
        
        # Mix operation: mixed[b, i, c] = sum_j(M[i, j] * x_inp[b, j, c])
        # Reshape for batch matmul
        x_reshaped = x_inp.permute(0, 2, 1)  # [B, C, n]
        mixed = torch.bmm(x_reshaped, M.T.unsqueeze(0).expand(B, n, n))  # [B, C, n]
        mixed = mixed.permute(0, 2, 1)  # [B, n, C]
        
        # Add: out[b, i, c] = mixed[b, i, c] + H_post[i] * y_norm[b, c]
        y_norm_f32 = y_norm.float()  # [B, C]
        H_post_expanded = H_post_activated.unsqueeze(0).unsqueeze(-1)  # [1, n, 1]
        y_expanded = y_norm_f32.unsqueeze(1)  # [B, 1, C]
        
        out = mixed + H_post_expanded * y_expanded
        
        return out, H_post_activated


# Convenience function to create an AITER-enabled mHC operator set
def create_aiter_ops(use_aiter: bool = True) -> dict:
    """
    Create a set of AITER-enabled operators for mHC.
    
    Args:
        use_aiter: Whether to use AITER when available
        
    Returns:
        Dictionary of operator instances
    """
    return {
        "rmsnorm": AITERRMSNorm(use_aiter),
        "sigmoid": AITERSigmoid(use_aiter),
        "gemm": AITERGEMM(use_aiter),
        "stream_ops": AITERStreamOps(use_aiter),
    }


# PyTorch integration for autograd
class AITERRMSNormFunction(torch.autograd.Function):
    """
    PyTorch autograd function for AITER RMSNorm.
    """
    
    @staticmethod
    def forward(ctx, inp, weight, eps, use_aiter):
        rmsnorm = AITERRMSNorm(use_aiter)
        out, rms = rmsnorm.forward(inp, weight, eps)
        ctx.save_for_backward(inp, weight, rms)
        ctx.eps = eps
        ctx.use_aiter = use_aiter
        return out, rms
    
    @staticmethod
    def backward(ctx, grad_out, grad_rms):
        inp, weight, rms = ctx.saved_tensors
        rmsnorm = AITERRMSNorm(ctx.use_aiter)
        d_inp, d_weight = rmsnorm.backward(grad_out, inp, weight, rms)
        return d_inp, d_weight, None, None


def aiter_rmsnorm(
    inp: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_aiter: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    AITER-enabled RMSNorm with autograd support.
    
    Args:
        inp: Input tensor [B, C]
        weight: Weight tensor [C]
        eps: Epsilon for numerical stability
        use_aiter: Whether to use AITER when available
        
    Returns:
        Tuple of (normalized output, rms values)
    """
    return AITERRMSNormFunction.apply(inp, weight, eps, use_aiter)

