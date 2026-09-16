# SAM 2: 图像与视频通用分割模型 (Segment Anything in Images and Videos)

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`论文`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`项目主页`](https://ai.meta.com/sam2)] [[`在线演示`](https://sam2.metademolab.com/)] [[`数据集`](https://ai.meta.com/datasets/segment-anything-video)] [[`官方博客`](https://ai.meta.com/blog/segment-anything-2)] [[`BibTeX 引用`](#citing-sam-2)]

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

![SAM 2 architecture](assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** 是一个致力于解决图像与视频中基于提示（Promptable）视觉分割的基础模型（Foundation Model）。我们将 SAM 拓展到了视频领域——将单张图像视作仅含单帧的视频。该模型设计采用了带有流式内存（Streaming Memory）的极简 Transformer 架构，可实现实时视频处理。我们构建了一个人在回路（Model-in-the-loop）的数据引擎，通过人机交互协同迭代模型与数据，构建了迄今为止规模最大的视频分割数据集 [**SA-V 数据集**](https://ai.meta.com/datasets/segment-anything-video)。在我们的数据集上训练的 SAM 2 在广泛的任务和视觉领域中均展现出卓越的性能表现。

![SA-V dataset](assets/sa_v_dataset.jpg?raw=true)

## 最新动态 (Latest updates)

**2024年12月11日 -- 全模型编译大幅提升 VOS 推理速度，并推出全新 `SAM2VideoPredictor` 优化多目标跟踪**

- 现已支持在视频上对整个 SAM 2 模型进行 `torch.compile` 全模型编译，可在 `build_sam2_video_predictor` 中通过设置 `vos_optimized=True` 启用，大幅提升视频对象分割（VOS）的推理速度。
- 更新了 `SAM2VideoPredictor` 的实现以支持独立针对单个对象的推理，消除了此前多目标跟踪中的先验提示限制，支持在跟踪开始后动态新增目标对象。
- 详见 [`RELEASE_NOTES.md`](RELEASE_NOTES.md) 了解完整更新细节。

**2024年09月30日 -- SAM 2.1 开发者套件正式发布（全新权重、训练代码及 Web Demo）**

- 发布了一组性能更强的新模型权重（命名为 **SAM 2.1**），详见[模型说明](#model-description)。
  * 若要使用全新的 SAM 2.1 权重，需要使用本仓库的最新代码。如果您之前安装过本仓库的旧版本，请先执行 `pip uninstall SAM-2` 卸载旧版本，使用 `git pull` 拉取最新代码，然后参照下方的[安装指南](#installation)重新安装。
- 训练与微调（Fine-tuning）代码已正式开源，请参阅 [`training/README.md`](training/README.md) 了解上手指引。
- SAM 2 Web Demo 的前端与后端代码已发布，详情请见 [`demo/README.md`](demo/README.md)。

## 安装指南 (Installation)

使用 SAM 2 前需先完成安装。运行环境要求 `python>=3.10`，以及 `torch>=2.5.1` 与 `torchvision>=0.20.1`。请按照 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 安装 PyTorch 和 TorchVision 依赖。在带有 GPU 的机器上，可以通过以下命令安装 SAM 2：

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

若在 Windows 系统上安装，强烈建议使用带有 Ubuntu 的 [适用于 Linux 的 Windows 子系统 (WSL)](https://learn.microsoft.com/zh-cn/windows/wsl/install)。

若要使用 SAM 2 预测器并运行示例 Notebook，需要安装 `jupyter` 和 `matplotlib`，可通过以下命令安装：

```bash
pip install -e ".[notebooks]"
```

注意：
1. 建议使用 [Anaconda](https://www.anaconda.com/) 创建全新的 Python 虚拟环境，并参照 https://pytorch.org/ 使用 `pip` 安装 PyTorch 2.5.1（或更高版本）。如果您当前环境中的 PyTorch 版本低于 2.5.1，上述安装命令将尝试通过 `pip` 将其升级到最新的 PyTorch 版本。
2. 上述安装步骤需要使用 `nvcc` 编译器编译自定义 CUDA Kernel。如果您的机器上尚未配置 `nvcc`，请安装与您的 PyTorch CUDA 版本匹配的 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)。
3. 如果在安装过程中看到类似于 `Failed to build the SAM 2 CUDA extension` 的提示信息，可以忽略并继续使用 SAM 2（仅部分后处理功能可能会受到限制，但在绝大多数场景下不会影响分割结果）。

常见问题与解决方案请参阅 [`INSTALL.md`](./INSTALL.md)。

## 快速上手 (Getting Started)

### 下载模型权重 (Download Checkpoints)

首先需要下载模型权重。可以通过运行以下脚本下载所有模型权重：

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

或者单独下载指定权重：

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

（注：以上为优化后的 SAM 2.1 权重版本；详情请见[模型说明](#model-description)。）

只需数行代码即可使用 SAM 2 进行图像和视频预测：

### 图像预测 (Image prediction)

SAM 2 完全继承了 [SAM](https://github.com/facebookresearch/segment-anything) 在静态图像上的全部能力，并提供了与 SAM 高度相似的图像预测 API。`SAM2ImagePredictor` 类为图像提示提供了简洁易用的接口：

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

静态图像使用场景的完整示例请参考 [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb)（亦可在 Colab 中[在线体验](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)）。

SAM 2 同样支持对图像进行全自动掩膜生成（Automatic Mask Generation）。请参阅 [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb)（亦可在 Colab 中[在线体验](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb)）。

### 视频预测 (Video prediction)

针对视频中的交互式提示分割与目标跟踪，我们提供了视频预测器（Video Predictor），支持添加提示并在整个视频中传播生成掩膜片段（Masklets）。SAM 2 支持多目标视频推理，并利用推理状态（Inference State）追踪记录每个视频中的交互历史：

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # 添加新提示并即时获取当前帧的分割结果
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # 在整个视频序列中传播提示以获得连续的掩膜片段
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

关于如何在视频中添加点击或边界框提示、细化掩膜以及跟踪多个目标的详细说明，请参考 [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb)（亦可在 Colab 中[在线体验](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb)）。

## 通过 🤗 Hugging Face 加载 (Load from 🤗 Hugging Face)

此外，也可以直接从 [Hugging Face](https://huggingface.co/models?search=facebook/sam2) 加载模型（需要安装 `pip install huggingface_hub`）。

图像预测：

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

视频预测：

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # 添加新提示并即时获取当前帧的分割结果
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # 在整个视频序列中传播提示以获得连续的掩膜片段
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

## 模型说明 (Model Description)

### SAM 2.1 权重 (SAM 2.1 checkpoints)

下表展示了于 2024 年 9 月 29 日发布的升级版 SAM 2.1 模型权重：
|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2.1_hiera_tiny <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt))    |     38.9     |          91.2          |        76.5         |        71.8        |       77.3        |
|   sam2.1_hiera_small <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt))   |      46      |          84.8          |        76.6         |        73.5        |       78.3        |
| sam2.1_hiera_base_plus <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)) |     80.8     |        64.1          |        78.2         |        73.7        |       78.2        |
|   sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |          39.5          |        79.5         |        74.6        |       80.6        |

### SAM 2 权重 (SAM 2 checkpoints)

2024 年 7 月 29 日发布的初代 SAM 2 模型权重如下：

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny <br /> ([config](sam2/configs/sam2/sam2_hiera_t.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt))   |     38.9     |          91.5          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small <br /> ([config](sam2/configs/sam2/sam2_hiera_s.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt))   |      46      |          85.6          |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus <br /> ([config](sam2/configs/sam2/sam2_hiera_b+.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)) |     80.8     |     64.8    |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large <br /> ([config](sam2/configs/sam2/sam2_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt))   |    224.4     | 39.7 |        76.0         |        74.6        |       79.8        |

运行速度在单张 A100 GPU、运行环境为 `torch 2.5.1, cuda 12.4` 条件下测得。基准评测示例（编译所有模型组件）请参阅 `benchmark.py`。若仅编译图像编码器（Image Encoder），使用方式会更为灵活，也能带来一定的加速效果（可在配置文件中设置 `compile_image_encoder: True`）。

## Segment Anything 视频数据集 (Segment Anything Video Dataset)

详情请参阅 [sav_dataset/README.md](sav_dataset/README.md)。

## 训练 SAM 2 (Training SAM 2)

您可以在自定义的图像、视频或混合数据集上训练或微调 SAM 2。请参阅训练指南 [training/README.md](training/README.md) 了解如何快速启动。

## SAM 2 Web 在线演示 (Web demo for SAM 2)

我们开源了 SAM 2 Web Demo 的前端与后端完整代码（可在本地独立部署，功能类似于 https://sam2.metademolab.com/demo）。详情请参阅 Web 演示说明 [demo/README.md](demo/README.md)。

## 开源协议 (License)

SAM 2 模型权重、SAM 2 Demo 代码（前端与后端）以及 SAM 2 训练代码均基于 [Apache 2.0 协议](./LICENSE) 开源；不过在 SAM 2 Demo 代码中使用的 [Inter 字体](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file) 与 [Noto Color Emoji 彩色表情符号](https://github.com/googlefonts/noto-emoji) 遵循 [SIL 开源字体许可证 1.1 版 (SIL Open Font License, version 1.1)](https://openfontlicense.org/open-font-license-official-text/)。

## 参与贡献 (Contributing)

请参阅[贡献指南](CONTRIBUTING.md)与[行为准则](CODE_OF_CONDUCT.md)。

## 贡献者名单 (Contributors)

SAM 2 项目的顺利完成离不开众多贡献者的鼎力协助（按首字母排序）：

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

第三方代码：我们使用基于 GPU 的连通分量算法（改编自 [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch)，协议位于 [`LICENSE_cctorch`](./LICENSE_cctorch)），作为掩膜预测的可选后处理步骤。

## 引用 SAM 2 (Citing SAM 2)

如果您在学术研究中使用了 SAM 2 或 SA-V 数据集，请使用以下 BibTeX 条目进行引用：

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月16日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
