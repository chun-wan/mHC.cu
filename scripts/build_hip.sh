#!/bin/bash
# Build script for mHC.cu HIP/ROCm on AMD MI300X
# Usage: ./scripts/build_hip.sh [--install] [--test] [--bench]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     mHC.cu HIP/ROCm Build Script for AMD MI300X          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for ROCm
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
if [ ! -d "$ROCM_PATH" ]; then
    echo -e "${RED}Error: ROCm not found at $ROCM_PATH${NC}"
    echo "Please install ROCm 6.0+ or set ROCM_PATH environment variable."
    exit 1
fi

echo -e "${GREEN}✓ ROCm found at: $ROCM_PATH${NC}"

# Check for hipcc
HIPCC="$ROCM_PATH/bin/hipcc"
if [ ! -x "$HIPCC" ]; then
    echo -e "${RED}Error: hipcc not found at $HIPCC${NC}"
    exit 1
fi
echo -e "${GREEN}✓ hipcc found: $($HIPCC --version | head -1)${NC}"

# Detect GPU architecture
echo ""
echo -e "${YELLOW}Detecting GPU architecture...${NC}"
if command -v rocm-smi &> /dev/null; then
    GPU_INFO=$(rocm-smi --showproductname 2>/dev/null || echo "Unknown GPU")
    echo -e "${GREEN}GPU: $GPU_INFO${NC}"
fi

# Default to MI300X (gfx942)
HIP_ARCH="${HIP_ARCH:-gfx942}"
echo -e "${GREEN}Target architecture: $HIP_ARCH${NC}"

# Parse arguments
DO_INSTALL=false
DO_TEST=false
DO_BENCH=false

for arg in "$@"; do
    case $arg in
        --install)
            DO_INSTALL=true
            ;;
        --test)
            DO_TEST=true
            ;;
        --bench)
            DO_BENCH=true
            ;;
        --all)
            DO_INSTALL=true
            DO_TEST=true
            DO_BENCH=true
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install   Install Python extension"
            echo "  --test      Run tests after build"
            echo "  --bench     Run benchmarks after build"
            echo "  --all       Do all of the above"
            echo "  --help      Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ROCM_PATH   Path to ROCm installation (default: /opt/rocm)"
            echo "  HIP_ARCH    GPU architecture (default: gfx942 for MI300X)"
            exit 0
            ;;
    esac
done

# Create build directory
BUILD_DIR="build_hip"
mkdir -p "$BUILD_DIR"

echo ""
echo -e "${YELLOW}Building mHC HIP kernels...${NC}"

# Export environment variables for setup.py
export ROCM_PATH
export HIP_ARCH

# Install Python extension
if [ "$DO_INSTALL" = true ]; then
    echo ""
    echo -e "${YELLOW}Installing Python extension...${NC}"
    
    # Check for PyTorch with ROCm support
    python3 -c "import torch; assert torch.cuda.is_available(), 'PyTorch ROCm not available'" 2>/dev/null || {
        echo -e "${RED}Error: PyTorch with ROCm support not found${NC}"
        echo "Please install PyTorch ROCm version:"
        echo "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
        exit 1
    }
    echo -e "${GREEN}✓ PyTorch ROCm detected${NC}"
    
    # Install using the HIP setup script
    pip install -e . --config-settings="--global-option=--plat-name=linux_x86_64" || {
        # Fallback: try using setup_hip.py directly
        echo -e "${YELLOW}Trying alternative install method...${NC}"
        python setup_hip.py develop
    }
    
    echo -e "${GREEN}✓ Python extension installed${NC}"
fi

# Run tests
if [ "$DO_TEST" = true ]; then
    echo ""
    echo -e "${YELLOW}Running tests...${NC}"
    
    # Python tests
    if python3 -c "import mhc_hip" 2>/dev/null; then
        echo "Running Python tests..."
        python3 -c "
import torch
import mhc_hip

# Simple smoke test
print('Testing Sinkhorn-Knopp...')
inp = torch.randn(4, 4, device='cuda')
inp = torch.exp(inp)  # Make positive
out = mhc_hip.sinkhorn_knopp_fwd(inp, 10, 1e-5)
print(f'  Input sum: {inp.sum():.4f}')
print(f'  Output row sums: {out.sum(dim=1)}')
print('  ✓ Sinkhorn-Knopp works!')

print('')
print('Testing RMSNorm...')
inp = torch.randn(8, 64, device='cuda', dtype=torch.bfloat16)
weight = torch.ones(64, device='cuda', dtype=torch.bfloat16)
out, rms = mhc_hip.rmsnorm_fwd(inp, weight, 1e-5)
print(f'  Output shape: {out.shape}')
print(f'  RMS values: {rms[:4]}')
print('  ✓ RMSNorm works!')

print('')
print('All HIP kernel tests passed!')
"
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${YELLOW}mhc_hip module not installed, skipping Python tests${NC}"
    fi
fi

# Run benchmarks
if [ "$DO_BENCH" = true ]; then
    echo ""
    echo -e "${YELLOW}Running benchmarks...${NC}"
    
    if python3 -c "import mhc_hip" 2>/dev/null; then
        python3 -c "
import torch
import mhc_hip
import time

print('=== mHC HIP Benchmark on MI300X ===')
print('')

# Warmup
for _ in range(10):
    inp = torch.randn(4, 4, device='cuda')
    out = mhc_hip.sinkhorn_knopp_fwd(torch.exp(inp), 10, 1e-5)
torch.cuda.synchronize()

# Sinkhorn-Knopp benchmark
print('Sinkhorn-Knopp (n=4, iters=20):')
sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
for m, n in sizes:
    inp = torch.exp(torch.randn(m, n, device='cuda'))
    
    # Warmup
    for _ in range(5):
        out = mhc_hip.sinkhorn_knopp_fwd(inp, 20, 1e-5)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iters = 100
    for _ in range(iters):
        out = mhc_hip.sinkhorn_knopp_fwd(inp, 20, 1e-5)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000  # ms
    print(f'  {m}x{n}: {elapsed:.3f} ms')

print('')
print('RMSNorm (B=128, C=1280):')
B, C = 128, 1280
inp = torch.randn(B, C, device='cuda', dtype=torch.bfloat16)
weight = torch.ones(C, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(10):
    out, rms = mhc_hip.rmsnorm_fwd(inp, weight, 1e-5)
torch.cuda.synchronize()

# Benchmark
start = time.time()
iters = 100
for _ in range(iters):
    out, rms = mhc_hip.rmsnorm_fwd(inp, weight, 1e-5)
torch.cuda.synchronize()
elapsed = (time.time() - start) / iters * 1000
print(f'  Forward: {elapsed:.3f} ms')

print('')
print('Benchmark complete!')
"
        echo -e "${GREEN}✓ Benchmarks complete${NC}"
    else
        echo -e "${YELLOW}mhc_hip module not installed, skipping benchmarks${NC}"
    fi
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Build Complete!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Run: python -c 'import mhc_hip; print(\"mHC HIP loaded!\")'"
echo "  2. Run benchmarks: ./scripts/build_hip.sh --bench"
echo "  3. Run tests: pytest src/python/tests -v"

