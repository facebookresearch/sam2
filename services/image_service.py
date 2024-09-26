import os
from tkinter import Image

import numpy as np
import torch
from PIL import Image

from config import app_config
from models.requests import SAMRequest
from providers import storage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
async def segment_image(payload: SAMRequest):
    try:
        file_path = await storage.download_file(payload.media_url, f"{app_config.paths.tmp_file_dir}/inputs/")
        model = build_model(payload.model.get_config(), payload.model.get_checkpoint())
        print(f"Built SAM model: {payload.model.get_config()} config and {payload.model.get_checkpoint()} checkpoint")
        predictor = SAM2ImagePredictor(model)
        predictor.set_image(get_image(file_path))

        input_point = np.array([[[pointer.x, pointer.y]] for pointer in payload.pointers])
        input_label = np.array([[pointer.label] for pointer in payload.pointers])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        print(f"Image Segmentation successful.")
        mask_paths = save_masks(os.path.basename(payload.media_url), masks)
        mask_urls = []
        for path in mask_paths:
            url = await storage.upload_file(
                path,
                app_config.storage.s3.bucket,
                f"stg/SAM/image_masks/{os.path.basename(path)}")
            mask_urls.append(url)
        print(f"Masked upload successful")
        await storage.delete_file(file_path)
        return {
            "media_url": payload.media_url,
            "masks": mask_urls,
        }
    except Exception as e:
        raise e


def build_model(config, checkpoint):
    return build_sam2(config, ckpt_path=checkpoint, device="cuda")


def get_image(path):
    image = Image.open(path)
    return np.array(image.convert("RGB"))


def save_masks(name, masks):
    n = name.split(".")[0]
    paths = []
    for i in range(masks.shape[0]):
        m = masks[i]
        image = Image.fromarray((m * 255).astype(np.uint8))
        paths.append(f"{app_config.paths.tmp_file_dir}/outputs/{n}_{i}.png")
        image.save(f"{app_config.paths.tmp_file_dir}/outputs/{n}_{i}.png")
    return paths
