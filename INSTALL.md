## Installation

### Requirements

- Linux with Python ≥ 3.10, PyTorch ≥ 2.3.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. Install them together at https://pytorch.org to ensure this.
  * Note older versions of Python or PyTorch may also work. However, the versions above are strongly recommended to provide all features such as `torch.compile`.
- [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. This should typically be CUDA 12.1 if you follow the default installation command.

Then, install SAM 2 from the root of this repository via
```bash
pip install -e ".[demo]"
```

### Common Installation Issues

Click each issue for its solutions:

<details>
<summary>
I got `ImportError: cannot import name '_C' from 'sam2'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e ".[demo]"` step above or the installation failed. Please install SAM 2 first, and see the other issues if your installation fails.
</details>

<details>
<summary>
I got `MissingConfigException: Cannot find primary config 'sam2_hiera_l.yaml'`
</summary>
<br/>

This is usually because you haven't run the `pip install -e .` step above, so `sam2_configs` isn't in your Python's `sys.path`. Please run this installation step. In case it still fails after the installation step, you may try manually adding the root of this repo to `PYTHONPATH` via
```bash
export SAM2_REPO_ROOT=/path/to/segment-anything  # path to this repo
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```
to manually add `sam2_configs` into your Python's `sys.path`.

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
</details>

<details>
<summary>
I got `undefined symbol: _ZN3c1015SmallVectorBaseIjE8grow_podEPKvmm` (or similar errors)
</summary>
<br/>

This usually happens because you have multiple versions of dependencies (PyTorch or CUDA) in your environment. During installation, the SAM 2 library is compiled against one version library while at run time it links against another version. This might be due to that you have different versions of PyTorch or CUDA installed separately via `pip` or `conda`. You may delete one of the duplicates to only keep a single PyTorch and CUDA version.

In particular, if you have a lower PyTorch version than 2.3.1, it's recommended to upgrade to PyTorch 2.3.1 or higher first. Otherwise, the installation script will try to upgrade to the latest PyTorch using `pip`, which could sometimes lead to duplicated PyTorch installation if you have previously installed another PyTorch version using `conda`.

We have been building SAM 2 against PyTorch 2.3.1 internally. However, a few user comments (e.g. https://github.com/facebookresearch/segment-anything-2/issues/22, https://github.com/facebookresearch/segment-anything-2/issues/14) suggested that downgrading to PyTorch 2.1.0 might resolve this problem. In case the error persists, you may try changing the restriction from `torch>=2.3.1` to `torch>=2.1.0` in both [`pyproject.toml`](pyproject.toml) and [`setup.py`](setup.py) to allow PyTorch 2.1.0.
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
