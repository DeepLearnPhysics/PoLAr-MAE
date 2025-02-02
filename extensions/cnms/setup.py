# -*- coding: utf-8 -*-
# @Author: Sam Young
# @Date:   2024-12-31 10:34:00
# @Email: youngsam@stanford.edu

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import find_packages

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
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pytorch3d",
    ],
)
