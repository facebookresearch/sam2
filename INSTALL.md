## Installation

### Requirements

- Linux with Python ≥ 3.10, PyTorch ≥ 2.3.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. Install them together at https://pytorch.org to ensure this.
  * Note older versions of Python or PyTorch may also work. However, the versions above are strongly recommended to provide all features such as `torch.compile`.
- [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. This should typically be CUDA 12.1 if you follow the default installation command.
- If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

Then, install SAM 2 from the root of this repository via
```bash
pip install -e ".[notebooks]"
```

Note that you may skip building the SAM 2 CUDA extension during installation via environment variable `SAM2_BUILD_CUDA=0`, as follows:
```bash
# skip the SAM 2 CUDA extension
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```
This would also skip the post-processing step at runtime (removing small holes and sprinkles in the output masks, which requires the CUDA extension), but shouldn't affect the results in most cases.

### Building the SAM 2 CUDA extension

By default, we allow the installation to proceed even if the SAM 2 CUDA extension fails to build. (In this case, the build errors are hidden unless using `-v` for verbose output in `pip install`.)

If you see a message like `Skipping the post-processing step due to the error above` at runtime or `Failed to build the SAM 2 CUDA extension due to the error above` during installation, it indicates that the SAM 2 CUDA extension failed to build in your environment. In this case, **you can still use SAM 2 for both image and video applications**. The post-processing step (removing small holes and sprinkles in the output masks) will be skipped, but this shouldn't affect the results in most cases.

If you would like to enable this post-processing step, you can reinstall SAM 2 on a GPU machine with environment variable `SAM2_BUILD_ALLOW_ERRORS=0` to force building the CUDA extension (and raise errors if it fails to build), as follows
```bash
pip uninstall -y SAM-2 && \
rm -f ./sam2/*.so && \
SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[notebooks]"
```

Note that PyTorch needs to be installed first before building the SAM 2 CUDA extension. It's also necessary to install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. (This should typically be CUDA 12.1 if you follow the default installation command.) After installing the CUDA toolkits, you can check its version via `nvcc --version`.

Please check the section below on common installation issues if the CUDA extension fails to build during installation or load at runtime.

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
I got `ImportError: cannot import name '_C' from 'sam2'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e ".[notebooks]"` step above or the installation failed. Please install SAM 2 first, and see the other issues if your installation fails.

In some systems, you may need to run `python setup.py build_ext --inplace` in the SAM 2 repo root as suggested in https://github.com/facebookresearch/sam2/issues/77.
</details>

<details>
<summary>
I got `MissingConfigException: Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e .` step above, so `sam2` isn't in your Python's `sys.path`. Please run this installation step. In case it still fails after the installation step, you may try manually adding the root of this repo to `PYTHONPATH` via
```bash
export SAM2_REPO_ROOT=/path/to/sam2  # path to this repo
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```
to manually add `sam2_configs` into your Python's `sys.path`.

</details>

<details>
<summary>
I got `RuntimeError: Error(s) in loading state_dict for SAM2Base` when loading the new SAM 2.1 checkpoints
</summary>
<br/>

This is likely because you have installed a previous version of this repo, which doesn't have the new modules to support the SAM 2.1 checkpoints yet. Please try the following steps:

1. pull the latest code from the `main` branch of this repo
2. run `pip uninstall -y SAM-2` to uninstall any previous installations
3. then install the latest repo again using `pip install -e ".[notebooks]"`

In case the steps above still don't resolve the error, please try running in your Python environment the following
```python
from sam2.modeling import sam2_base

print(sam2_base.__file__)
```
and check whether the content in the printed local path of `sam2/modeling/sam2_base.py` matches the latest one in https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py (e.g. whether your local file has `no_obj_embed_spatial`) to indentify if you're still using a previous installation.

</details>

<details>
<summary>
My installation failed with `CUDA_HOME environment variable is not set`
</summary>
<br/>

This usually happens because the installation step cannot find the CUDA toolkits (that contain the NVCC compiler) to build a custom CUDA kernel in SAM 2. Please install [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) or the version that matches the CUDA version for your PyTorch installation. If the error persists after installing CUDA toolkits, you may explicitly specify `CUDA_HOME` via
```
export CUDA_HOME=/usr/local/cuda  # change to your CUDA toolkit path
```
and rerun the installation.

Also, you should make sure
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```
print `(True, a directory with cuda)` to verify that the CUDA toolkits are correctly set up.

If you are still having problems after verifying that the CUDA toolkit is installed and the `CUDA_HOME` environment variable is set properly, you may have to add the `--no-build-isolation` flag to the pip command:
```
pip install --no-build-isolation -e .
```

</details>

<details>
<summary>
I got `undefined symbol: _ZN3c1015SmallVectorBaseIjE8grow_podEPKvmm` (or similar errors)
</summary>
<br/>

This usually happens because you have multiple versions of dependencies (PyTorch or CUDA) in your environment. During installation, the SAM 2 library is compiled against one version library while at run time it links against another version. This might be due to that you have different versions of PyTorch or CUDA installed separately via `pip` or `conda`. You may delete one of the duplicates to only keep a single PyTorch and CUDA version.

In particular, if you have a lower PyTorch version than 2.3.1, it's recommended to upgrade to PyTorch 2.3.1 or higher first. Otherwise, the installation script will try to upgrade to the latest PyTorch using `pip`, which could sometimes lead to duplicated PyTorch installation if you have previously installed another PyTorch version using `conda`.

We have been building SAM 2 against PyTorch 2.3.1 internally. However, a few user comments (e.g. https://github.com/facebookresearch/sam2/issues/22, https://github.com/facebookresearch/sam2/issues/14) suggested that downgrading to PyTorch 2.1.0 might resolve this problem. In case the error persists, you may try changing the restriction from `torch>=2.3.1` to `torch>=2.1.0` in both [`pyproject.toml`](pyproject.toml) and [`setup.py`](setup.py) to allow PyTorch 2.1.0.
</details>

<details>
<summary>
I got `CUDA error: no kernel image is available for execution on the device`
</summary>
<br/>

A possible cause could be that the CUDA kernel is somehow not compiled towards your GPU's CUDA [capability](https://developer.nvidia.com/cuda-gpus). This could happen if the installation is done in an environment different from the runtime (e.g. in a slurm system).

You can try pulling the latest code from the SAM 2 repo and running the following
```
export TORCH_CUDA_ARCH_LIST=9.0 8.0 8.6 8.9 7.0 7.2 7.5 6.0`
```
to manually specify the CUDA capability in the compilation target that matches your GPU.
</details>

<details>
<summary>
I got `RuntimeError: No available kernel. Aborting execution.` (or similar errors)
</summary>
<br/>

This is probably because your machine doesn't have a GPU or a compatible PyTorch version for Flash Attention (see also https://discuss.pytorch.org/t/using-f-scaled-dot-product-attention-gives-the-error-runtimeerror-no-available-kernel-aborting-execution/180900 for a discussion in PyTorch forum). You may be able to resolve this error by replacing the line
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
```
in [`sam2/modeling/sam/transformer.py`](sam2/modeling/sam/transformer.py) with
```python
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
```
to relax the attention kernel setting and use other kernels than Flash Attention.
</details>

<details>
<summary>
I got `Error compiling objects for extension`
</summary>
<br/>

You may see error log of:
> unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

This is probably because your versions of CUDA and Visual Studio are incompatible. (see also https://stackoverflow.com/questions/78515942/cuda-compatibility-with-visual-studio-2022-version-17-10 for a discussion in stackoverflow).<br> 
You may be able to fix this by adding the `-allow-unsupported-compiler` argument to `nvcc` after L48 in the [setup.py](https://github.com/facebookresearch/sam2/blob/main/setup.py). <br>
After adding the argument, `get_extension()` will look like this:
```python
def get_extensions():
    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-allow-unsupported-compiler"  # Add this argument
        ],
    }
    ext_modules = [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]
    return ext_modules
```
</details>
