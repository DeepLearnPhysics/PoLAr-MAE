# -*- coding: utf-8 -*-
# @Author: Sam Young
# @Date:   2024-12-31 10:34:00
# @Email: youngsam@stanford.edu

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import find_packages
import os

CUDA_FLAGS = []
if os.system('nvidia-smi -q') == 0:
    # auto-detect CUDA arch
    import torch
    CUDA_ARCH = []
    for i in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(i)
        CUDA_ARCH.append(f"{capability[0]}.{capability[1]}")
    
    for arch in CUDA_ARCH:
        CUDA_FLAGS.extend([f"-gencode=arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}", f"-gencode=arch=compute_{arch.replace('.', '')},code=compute_{arch.replace('.', '')}"])

CXX_FLAGS = ["-O3", "-march=native", "-ffast-math", "-ftree-vectorize", "-fomit-frame-pointer"]
NVCC_FLAGS = ["-O3", "--use_fast_math"]

setup(
    name="cnms",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cnms._ext",
            sources=[
                "csrc/greedy_reduction.cpp",
                "csrc/greedy_reduction_cuda.cu",
                "csrc/greedy_reduction_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": CUDA_FLAGS + NVCC_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pytorch3d",
    ],
)
