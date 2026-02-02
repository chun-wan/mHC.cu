"""
Setup script for mHC.cu HIP/ROCm version (AMD MI300X)

This builds the mHC CUDA kernels for AMD GPUs using HIP/ROCm.
Requires ROCm 6.0+ with PyTorch ROCm support.

Features:
- Custom HIP kernels optimized for MI300X (gfx942)
- Optional AITER integration for additional performance
- hipBLASLt for efficient GEMM operations

AITER (AI Tensor Engine for ROCm): https://github.com/ROCm/aiter
"""
import os
import subprocess
import sys
from setuptools import setup, find_packages

# Set HIP environment before importing torch
os.environ['HIP_PLATFORM'] = 'amd'
os.environ.setdefault('ROCM_PATH', '/opt/rocm')

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check for AITER availability
def check_aiter_available():
    """Check if AITER is installed."""
    try:
        import aiter
        return True
    except ImportError:
        return False

def get_rocm_version():
    """Get ROCm version from rocm-smi or environment."""
    try:
        result = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'ROCm' in line:
                    parts = line.split()
                    for p in parts:
                        if p[0].isdigit():
                            return p
    except:
        pass
    return os.environ.get('ROCM_VERSION', '6.0')

def get_hip_include_dirs():
    """Get HIP include directories."""
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    dirs = [
        os.path.join(rocm_path, 'include'),
        os.path.join(rocm_path, 'include', 'hip'),
        os.path.join(rocm_path, 'include', 'hipblas'),
        os.path.join(rocm_path, 'include', 'rocblas'),
        os.path.join(os.path.dirname(__file__), 'src/csrc/include'),
        os.path.join(os.path.dirname(__file__), 'src/csrc/kernels'),
    ]
    
    # Add AITER include dir if using AITER
    if USE_AITER:
        try:
            import aiter
            aiter_path = os.path.dirname(aiter.__file__)
            dirs.append(os.path.join(aiter_path, 'include'))
        except:
            pass
    
    return dirs

def get_hip_library_dirs():
    """Get HIP library directories."""
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    dirs = [
        os.path.join(rocm_path, 'lib'),
        os.path.join(rocm_path, 'lib64'),
    ]
    
    # Add AITER library dir if using AITER
    if USE_AITER:
        try:
            import aiter
            aiter_path = os.path.dirname(aiter.__file__)
            dirs.append(os.path.join(aiter_path, 'lib'))
        except:
            pass
    
    return dirs

# Parse command line for AITER option
USE_AITER = os.environ.get('MHC_USE_AITER', '0') == '1'
if '--use-aiter' in sys.argv:
    USE_AITER = True
    sys.argv.remove('--use-aiter')
elif '--no-aiter' in sys.argv:
    USE_AITER = False
    sys.argv.remove('--no-aiter')

# Auto-detect AITER if not explicitly set
if not USE_AITER and check_aiter_available():
    print("[mHC Setup] AITER detected, enabling AITER support")
    print("[mHC Setup] Use --no-aiter to disable")
    USE_AITER = True

# Get target GPU architecture
HIP_ARCH = os.environ.get('HIP_ARCH', 'gfx942')  # MI300X default

# Extra compile args for CXX
cxx_compile_args = [
    '-std=c++17',
    '-O3',
    '-ffast-math',
    '-D__HIP_PLATFORM_AMD__',
    '-DUSE_ROCM',
]

# HIP/NVCC compile args (these go to hipcc)
hip_compile_args = [
    '-std=c++17',
    '-O3',
    f'--offload-arch={HIP_ARCH}',
    '-D__HIP_PLATFORM_AMD__',
    '-DUSE_ROCM',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
]

# Add AITER flag if enabled
if USE_AITER:
    cxx_compile_args.append('-DMHC_USE_AITER')
    hip_compile_args.append('-DMHC_USE_AITER')
    print("[mHC Setup] Building with AITER support")
else:
    print("[mHC Setup] Building without AITER support")

# Libraries to link
hip_libraries = [
    'amdhip64',
    'hipblas',
    'hipblaslt',
    'rocblas',
]

# Note: AITER is a Python package, not a C++ library, so we don't link against it
# AITER operators are used through Python bindings, not C++ linking

# Define macros
define_macros = [
    ('USE_ROCM', None),
    ('__HIP_PLATFORM_AMD__', None),
]

if USE_AITER:
    define_macros.append(('MHC_USE_AITER', None))

# Python modules to install
py_modules = [
    'aiter_ops',
    'mhc_aiter',
]

setup(
    name='mhc-rocm',
    version='0.1.0',
    description='mHC kernels for AMD MI300X (HIP/ROCm) with optional AITER acceleration',
    long_description=open('README_HIP.md').read() if os.path.exists('README_HIP.md') else '',
    long_description_content_type='text/markdown',
    author='DeepSeek-AI (paper), Andre Slavescu (CUDA impl), AMD port',
    url='https://github.com/chun-wan/mHC.cu',
    packages=find_packages(where='src/python'),
    package_dir={'': 'src/python'},
    py_modules=py_modules,
    ext_modules=[
        CUDAExtension(
            name='mhc_hip',
            sources=['src/python/bindings_hip.cu'],
            include_dirs=get_hip_include_dirs(),
            library_dirs=get_hip_library_dirs(),
            libraries=hip_libraries,
            extra_compile_args={
                'cxx': cxx_compile_args,
                'nvcc': hip_compile_args,  # For ROCm, 'nvcc' maps to hipcc
            },
            define_macros=define_macros,
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
    ],
    extras_require={
        'dev': ['black', 'pytest', 'ruff'],
        'aiter': ['aiter>=0.1.9'],  # AITER for additional acceleration
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

print(f"""
========================================
mHC.cu HIP/ROCm Build Configuration
========================================
ROCm Version: {get_rocm_version()}
AITER Support: {'Enabled' if USE_AITER else 'Disabled'}
========================================
""")
