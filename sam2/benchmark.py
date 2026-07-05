# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor

# Only cuda supported
assert torch.cuda.is_available()
device = torch.device("cuda")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Config and checkpoint
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Build video predictor with vos_optimized=True setting
predictor = build_sam2_video_predictor(
    model_cfg, sam2_checkpoint, device=device, vos_optimized=True
)


# Initialize with video
video_dir = "notebooks/videos/bedroom"
# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)


# Number of runs, warmup etc
warm_up, runs = 5, 25
verbose = True
num_frames = len(frame_names)
total, count = 0, 0
torch.cuda.empty_cache()

# We will select an object with a click.
# See video_predictor_example.ipynb for more detailed explanation
ann_frame_idx, ann_obj_id = 0, 1
# Add a positive click at (x, y) = (210, 350)
# For labels, `1` means positive click
points = np.array([[210, 350]], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Warmup and then average FPS over several runs
with torch.autocast("cuda", torch.bfloat16):
    with torch.inference_mode():
        for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
            start = time.time()
            # Start tracking
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(inference_state):
                pass

            end = time.time()
            total += end - start
            count += 1
            if i == warm_up - 1:
                print("Warmup FPS: ", count * num_frames / total)
                total = 0
                count = 0

print("FPS: ", count * num_frames / total)
