# Makefile for mHC.cu HIP/ROCm build (AMD MI300X)

.PHONY: all build test bench clean format install install-dev

# ROCm configuration
ROCM_PATH ?= /opt/rocm
HIP_ARCH ?= gfx942  # MI300X
HIPCC := $(ROCM_PATH)/bin/hipcc

BUILD_DIR = build_hip
STAMP_DIR = .stamps_hip

# Compiler flags
HIPCC_FLAGS := -std=c++17 -O3 -ffast-math
HIPCC_FLAGS += --offload-arch=$(HIP_ARCH)
HIPCC_FLAGS += -D__HIP_PLATFORM_AMD__
HIPCC_FLAGS += -I$(ROCM_PATH)/include
HIPCC_FLAGS += -Isrc/csrc/include
HIPCC_FLAGS += -Isrc/csrc/kernels

# Linker flags
LDFLAGS := -L$(ROCM_PATH)/lib
LDFLAGS += -lamdhip64 -lhipblas -lhipblaslt -lrocblas

# Source files
HIP_HEADERS := $(wildcard src/csrc/include/*_hip.h)
HIP_HEADERS += $(wildcard src/csrc/kernels/*_hip.h)

all: build

$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build native C++/HIP tests and benchmarks
build: $(BUILD_DIR) $(STAMP_DIR)/build

$(STAMP_DIR)/build: $(HIP_HEADERS) | $(STAMP_DIR) $(BUILD_DIR)
	@echo "Building mHC HIP kernels for MI300X ($(HIP_ARCH))..."
	@touch $@

# Install Python extension
install:
	@echo "Installing mHC HIP Python extension..."
	HIP_ARCH=$(HIP_ARCH) pip install -e . --config-settings="--build-option=--plat-name=linux_x86_64"

install-dev:
	@echo "Installing mHC HIP with dev dependencies..."
	HIP_ARCH=$(HIP_ARCH) pip install -e ".[dev]"

# Run tests
test: build
	@echo "Running HIP tests..."
	@for t in $(BUILD_DIR)/test_*; do \
		if [ -x "$$t" ]; then \
			echo "Running $$t..."; \
			$$t || exit 1; \
		fi; \
	done
	@echo "All HIP tests passed."

test-python: install
	@echo "Running Python tests..."
	pytest src/python/tests -v -x

# Run benchmarks
bench: build
	@echo "Running HIP benchmarks..."
	@for b in $(BUILD_DIR)/bench_*; do \
		if [ -x "$$b" ]; then \
			echo "Running $$b..."; \
			$$b; \
		fi; \
	done

bench-python: install
	@echo "Running Python benchmarks..."
	python src/python/benchmarks/bench_layer.py --all-configs --backward

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(STAMP_DIR)
	rm -rf *.egg-info dist build
	find . -name "*.so" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Format code
format:
	@echo "Formatting C++/HIP code..."
	find src/csrc -name "*.h" -o -name "*.cpp" | xargs clang-format -i
	@echo "Formatting Python code..."
	black src/python

# Check GPU info
gpu-info:
	@echo "=== ROCm Information ==="
	rocm-smi --showproductname
	@echo ""
	@echo "=== HIP Configuration ==="
	@echo "ROCM_PATH: $(ROCM_PATH)"
	@echo "HIP_ARCH: $(HIP_ARCH)"
	@echo "HIPCC: $(HIPCC)"
	@$(HIPCC) --version

# Help
help:
	@echo "mHC.cu HIP/ROCm Makefile for AMD MI300X"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build everything (default)"
	@echo "  build        - Build native C++/HIP code"
	@echo "  install      - Install Python extension"
	@echo "  install-dev  - Install with dev dependencies"
	@echo "  test         - Run C++/HIP tests"
	@echo "  test-python  - Run Python tests"
	@echo "  bench        - Run C++/HIP benchmarks"
	@echo "  bench-python - Run Python benchmarks"
	@echo "  clean        - Clean build artifacts"
	@echo "  format       - Format source code"
	@echo "  gpu-info     - Show GPU and ROCm info"
	@echo ""
	@echo "Variables:"
	@echo "  ROCM_PATH    - ROCm installation path (default: /opt/rocm)"
	@echo "  HIP_ARCH     - Target GPU architecture (default: gfx942 for MI300X)"
	@echo ""
	@echo "Example:"
	@echo "  make HIP_ARCH=gfx942 install  # Build for MI300X"
	@echo "  make HIP_ARCH=gfx90a install  # Build for MI250X"

