# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from setuptools import find_packages, setup

# Package metadata
NAME = "SAM-2"
VERSION = "1.0"
DESCRIPTION = "SAM 2: Segment Anything in Images and Videos"
URL = "https://github.com/facebookresearch/sam2"
AUTHOR = "Meta AI"
AUTHOR_EMAIL = "segment-anything@meta.com"
LICENSE = "Apache 2.0"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "pillow>=9.4.0",
]

EXTRA_PACKAGES = {
    "notebooks": [
        "matplotlib>=3.9.1",
        "jupyter>=1.0.0",
        "opencv-python>=4.7.0",
        "eva-decord>=0.6.1",
    ],
    "interactive-demo": [
        "Flask>=3.0.3",
        "Flask-Cors>=5.0.0",
        "av>=13.0.0",
        "dataclasses-json>=0.6.7",
        "eva-decord>=0.6.1",
        "gunicorn>=23.0.0",
        "imagesize>=1.4.1",
        "pycocotools>=2.0.8",
        "strawberry-graphql>=0.243.0",
    ],
    "dev": [
        "black==24.2.0",
        "usort==1.0.2",
        "ufmt==2.0.0b2",
        "fvcore>=0.1.5.post20221221",
        "pandas>=2.2.2",
        "scikit-image>=0.24.0",
        "tensorboard>=2.17.0",
        "pycocotools>=2.0.8",
        "tensordict>=0.6.0",
        "opencv-python>=4.7.0",
        "submitit>=1.5.1",
    ],
}

# By default, we also build the SAM 2 CUDA extension.
# You may turn off CUDA build with `export SAM2_BUILD_CUDA=0`.
BUILD_CUDA = os.getenv("SAM2_BUILD_CUDA", "1") == "1"
# By default, we allow SAM 2 installation to proceed even with build errors.
# You may force stopping on errors with `export SAM2_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = os.getenv("SAM2_BUILD_ALLOW_ERRORS", "1") == "1"

# Catch and skip errors during extension building and print a warning message
# (note that this message only shows up under verbose build mode
# "pip install -v -e ." or "python setup.py build_ext -v")
CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the SAM 2 CUDA extension due to the error above. "
    "You can still use SAM 2 and it's OK to ignore the error above, although some "
    "post-processing functionality may be limited (which doesn't affect the results in most cases; "
    "(see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension

        srcs = ["sam2/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            ext_modules = []
        else:
            raise e

    return ext_modules


try:
    from torch.utils.cpp_extension import BuildExtension

    class BuildExtensionIgnoreErrors(BuildExtension):

        def finalize_options(self):
            try:
                super().finalize_options()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def get_ext_filename(self, ext_name):
            try:
                return super().get_ext_filename(ext_name)
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []
                return "_C.so"

    cmdclass = {
        "build_ext": (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        )
    }
except Exception as e:
    cmdclass = {}
    if BUILD_ALLOW_ERRORS:
        print(CUDA_ERROR_MSG.format(e))
    else:
        raise e


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
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
