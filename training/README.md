# Training Code for SAM 2

This folder contains the training code for SAM 2, a foundation model for promptable visual segmentation in images and videos. 
The code allows users to train and fine-tune SAM 2 on their own datasets (image, video, or both).

## Structure

The training code is organized into the following subfolders:

* `dataset`: This folder contains image and video dataset and dataloader classes as well as their transforms.
* `model`: This folder contains the main model class (`SAM2Train`) for training/fine-tuning. `SAM2Train` inherits from `SAM2Base` model and provides functions to enable training or fine-tuning SAM 2. It also accepts all training-time parameters used for simulating user prompts (e.g. iterative point sampling).
* `utils`: This folder contains training utils such as loggers and distributed training utils.
* `scripts`: This folder contains the script to extract the frames of SA-V dataset to be used in training.
* `loss_fns.py`: This file has the main loss class (`MultiStepMultiMasksAndIous`) used for training.
* `optimizer.py`:  This file contains all optimizer utils that support arbitrary schedulers.
* `trainer.py`: This file contains the `Trainer` class that accepts all the `Hydra` configurable modules (model, optimizer, datasets, etc..) and implements the main train/eval loop.
* `train.py`: This script is used to launch training jobs. It supports single and multi-node jobs. For usage, please check the [Getting Started](README.md#getting-started) section or run `python training/train.py -h`

## Getting Started

To get started with the training code, we provide a simple example to fine-tune our checkpoints on [MOSE](https://henghuiding.github.io/MOSE/) dataset, which can be extended to your custom datasets.

#### Requirements:
- We assume training on A100 GPUs with **80 GB** of memory.
- Download the MOSE dataset using one of the provided links from [here](https://github.com/henghuiding/MOSE-api?tab=readme-ov-file#download).

#### Steps to fine-tune on MOSE:
- Install the packages required for training by running `pip install -e ".[dev]"`.
- Set the paths for MOSE dataset in `configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`.
    ```yaml
    dataset:
        # PATHS to Dataset
        img_folder: null # PATH to MOSE JPEGImages folder
        gt_folder: null # PATH to MOSE Annotations folder
        file_list_txt: null # Optional PATH to filelist containing a subset of videos to be used for training
    ```
- To fine-tune the base model on MOSE using 8 GPUs, run 

    ```python
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 8
    ```

    We also support multi-node training on a cluster using [SLURM](https://slurm.schedmd.com/documentation.html), for example, you can train on 2 nodes by running

    ```python
    python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 1 \
        --num-gpus 8 \
        --num-nodes 2
        --partition $PARTITION \
        --qos $QOS \
        --account $ACCOUNT
    ```
    where partition, qos, and account are optional and depend on your SLURM configuration.
    By default, the checkpoint and logs will be saved under `sam2_logs` directory in the root of the repo. Alternatively, you can set the experiment log directory in the config file as follows:
  
    ```yaml
      experiment_log_dir: null # Path to log directory, defaults to ./sam2_logs/${config_name}
    ```
    The training losses can be monitored using `tensorboard` logs stored under `tensorboard/` in the experiment log directory. We also provide a sample validation [split]( ../training/assets/MOSE_sample_val_list.txt) for evaluation purposes. To generate predictions, follow this [guide](../tools/README.md) on how to use our `vos_inference.py` script. After generating the predictions, you can run the `sav_evaluator.py` as detailed [here](../sav_dataset/README.md#sa-v-val-and-test-evaluation). The expected MOSE J&F after fine-tuning the Base plus model is 79.4.
    
    
    After training/fine-tuning, you can then use the new checkpoint (saved in `checkpoints/` in the experiment log directory) similar to SAM 2 released checkpoints (as illustrated [here](../README.md#image-prediction)).
## Training on images and videos
The code supports training on images and videos (similar to how SAM 2 is trained). We provide classes for loading SA-1B as a sample image dataset, SA-V as a sample video dataset, as well as any DAVIS-style video dataset (e.g. MOSE). Note that to train on SA-V, you must first extract all videos to JPEG frames using the provided extraction [script](./scripts/sav_frame_extraction_submitit.py). Below is an example of how to setup the datasets in your config to train on a mix of image and video datasets:

```yaml
data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset 
    phases_per_epoch: ${phases_per_epoch} # Chunks a single epoch into smaller phases
    batch_sizes: # List of batch sizes corresponding to each dataset
    - ${bs1} # Batch size of dataset 1
    - ${bs2} # Batch size of dataset 2
    datasets:
    # SA1B as an example of an image dataset
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.SA1BRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist} # Optional
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 1
        max_num_objects: ${max_num_objects_per_image}
      transforms: ${image_transforms}
    # SA-V as an example of a video dataset
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.JSONRawDataset
        img_folder: ${path_to_img_folder}
        gt_folder: ${path_to_gt_folder}
        file_list_txt: ${path_to_train_filelist} # Optional
        ann_every: 4
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: 8 # Number of frames per video
        max_num_objects: ${max_num_objects_per_video}
        reverse_time_prob: ${reverse_time_prob} # probability to reverse video
      transforms: ${video_transforms}
    shuffle: True
    num_workers: ${num_train_workers}
    pin_memory: True
    drop_last: True
    collate_fn:
    _target_: training.utils.data_utils.collate_fn
    _partial_: true
    dict_key: all
```
