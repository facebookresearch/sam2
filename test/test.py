# load small video
# use sam2 to process this video with high limit on number of frame 
# compare mask with expected mask through IoU > 0.9

import os, sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import json
import unittest
import pathlib


from sam2.utils.display import show_box, show_mask, show_points
from sam2.utils.amg import mask_to_rle_pytorch, rle_to_mask

class TestSAM2_LV(unittest.TestCase):
    def test_one_mask_bedroom(self):
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        from sam2.build_sam import build_sam2_video_predictor

        this_file_path = pathlib.Path(__file__).parent.resolve()
        sam2_checkpoint = os.path.join(this_file_path,"../checkpoints/sam2.1_hiera_tiny.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
        video = os.path.join(this_file_path,"./assets/bedroom.mp4")

        inference_state = predictor.init_state(video_path=video)
        predictor.reset_state(inference_state)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a 2 positive click at (x, y) = (250, 220) to refine the mask
        # sending all clicks (and their labels) to `add_new_points_or_box`
        points = np.array([[210, 350], [250, 220]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, nbr_frame_to_keep_in_memory=60):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        last_id = np.max(list(video_segments.keys()))
        out_obj_id = video_segments[last_id].keys()
        out_mask = video_segments[last_id].values()
        # print(out_mask)

        self.assertEqual(len(out_obj_id),len(out_mask))
        self.assertEqual(len(out_mask),1)
            
        mask = torch.tensor(list(out_mask)[0], dtype=torch.uint8)
        # print(mask)

        # rle_mask = mask_to_rle_pytorch(mask)
        # print(rle_mask)

        # assert len(rle_mask) == 1

        # rle_mask = rle_mask[0]

        with open(os.path.join(this_file_path,f"assets/bedroom_frame{last_id}_mask.json"), "r") as fd:
            ground_truth_rle = json.load(fd)
            
        ground_truth_mask = torch.tensor(rle_to_mask(ground_truth_rle), dtype=torch.uint8)

        inter = torch.logical_and(ground_truth_mask, mask).sum()
        union = torch.logical_or(ground_truth_mask, mask).sum()
        iou = inter / union

        self.assertGreaterEqual(iou,0.98)

        print(f"Test done with IoU={iou}")
        # to display last frame :
        # import decord
        # import matplotlib.pyplot as plt

        # vr = decord.VideoReader(video)
        # frame = vr[last_id]
        # plt.figure(figsize=(6, 4))
        # plt.title(f"frame {last_id}")
        # plt.imshow(frame)
        # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #     show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

if __name__ == "__main__":
    unittest.main()
