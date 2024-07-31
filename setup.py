# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package metadata
NAME = "SAM 2"
VERSION = "1.0"
DESCRIPTION = "SAM 2: Segment Anything in Images and Videos"
URL = "https://github.com/facebookresearch/segment-anything-2"
AUTHOR = "Meta AI"
AUTHOR_EMAIL = "segment-anything@meta.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
]

EXTRA_PACKAGES = {
    "demo": ["matplotlib>=3.9.1", "jupyter>=1.0.0", "opencv-python>=4.7.0"],
    "dev": ["black==24.2.0", "usort==1.0.2", "ufmt==2.0.0b2"],
}

# Multiple CUDA Compute Capabilities Support
# Taken from here: https://github.com/NVlabs/tiny-cuda-nn/blob/master/bindings/torch/setup.py
if "SAM2_CUDA_ARCHITECTURES" in os.environ and os.environ["SAM2_CUDA_ARCHITECTURES"]:
    arcs = os.environ["SAM2_CUDA_ARCHITECTURES"].replace(";", ",").split(",")
    compute_capabilities = [int(x) for x in arcs]
    print(f"Obtained compute capabilities {compute_capabilities} from environment variable SAM2_CUDA_ARCHITECTURES")
elif torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    compute_capabilities = [major * 10 + minor]
    print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
else:
    raise EnvironmentError(
        "Unknown compute capability. "
        "Specify the target compute capabilities in the SAM2_CUDA_ARCHITECTURES environment variable or "
        "install PyTorch with the CUDA backend to detect it automatically."
    )

if os.name == "nt":

    def find_cl_path():
        import glob

        for executable in ["Program Files (x86)", "Program Files"]:
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path
    else:
        # cl.exe was found in PATH, so we can assume that the user is already in a developer command prompt
        # In this case, BuildExtensions requires the following environment variable to be set such that it
        # won't try to activate a developer command prompt a second time.
        os.environ["DISTUTILS_USE_SDK"] = "1"

source_files = [
    "sam2/csrc/connected_components.cu",
]

base_nvcc_flags = [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]


def make_extension(compute_capability):
    nvcc_flags = base_nvcc_flags + [
        f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]
    ]

    ext = CUDAExtension(
        name=f"sam2_bindings._{compute_capability}_C",
        sources=source_files,
        extra_compile_args={"nvcc": nvcc_flags},
        libraries=["cuda"],
    )
    return ext


ext_modules = [make_extension(comp) for comp in compute_capabilities]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(exclude="notebooks"),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
