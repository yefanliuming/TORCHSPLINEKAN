import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def build_ext():
    import torch
    cuda_available = torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1"

    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list)

    sources_cpp = [
        "csrc/kan_ext.cpp",
        "csrc/kan_cpu.cpp",
        "csrc/kan_autograd.cpp",
    ]

    # CUDA 11.8 requires GCC <= 11. Use gcc-11 explicitly when available.
    import shutil
    gcc11 = shutil.which("gcc-11")
    gxx11 = shutil.which("g++-11")

    cxx_flags = ["-O3", "-std=c++17"]
    if gxx11:
        os.environ["CC"]  = gcc11
        os.environ["CXX"] = gxx11

    extra_compile_args = {
        "cxx": cxx_flags,
    }

    if cuda_available:
        sources_cuda = sources_cpp + ["csrc/kan_cuda.cu", "csrc/kan_reg_cuda.cu"]
        extra_compile_args["cxx"] = cxx_flags + ["-DKAN_WITH_CUDA"]
        nvcc_flags = [
            "-O3",
            "-DKAN_WITH_CUDA",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ]
        if gxx11:
            nvcc_flags += ["-ccbin", gxx11]
        extra_compile_args["nvcc"] = nvcc_flags
        ext = CUDAExtension(
            name="spline_kan._C",
            sources=sources_cuda,
            extra_compile_args=extra_compile_args,
            include_dirs=["csrc"],
        )
    else:
        print("WARNING: CUDA not available, building CPU-only extension.")
        ext = CppExtension(
            name="spline_kan._C",
            sources=sources_cpp,
            extra_compile_args=extra_compile_args,
            include_dirs=["csrc"],
        )

    return [ext]


setup(
    name="torch-spline-kan",
    version="0.1.0",
    ext_modules=build_ext(),
    cmdclass={"build_ext": BuildExtension},
    package_data={"spline_kan": ["*.pyi"]},
)
