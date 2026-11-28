# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor


class VideoPredictorOffloadTest(unittest.TestCase):
    def test_mask_offload_waits_for_cpu_copy(self):
        for device_type, non_blocking in (("cpu", False), ("cuda", True)):
            with self.subTest(storage_device=device_type):
                prediction = Mock(spec=torch.Tensor)
                predictor = SimpleNamespace(
                    fill_hole_area=0,
                    _get_image_feature=Mock(return_value=(None,) * 5),
                    _get_maskmem_pos_enc=Mock(return_value=None),
                    track_step=Mock(
                        return_value={
                            "pred_masks": prediction,
                            "maskmem_features": None,
                            "obj_ptr": None,
                            "object_score_logits": None,
                        }
                    ),
                )
                device = torch.device(device_type)

                stored, _ = SAM2VideoPredictor._run_single_frame_inference(
                    predictor,
                    {"num_frames": 1, "storage_device": device},
                    output_dict={},
                    frame_idx=0,
                    batch_size=1,
                    is_init_cond_frame=True,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=False,
                )

                prediction.to.assert_called_once_with(device, non_blocking=non_blocking)
                self.assertIs(stored["pred_masks"], prediction.to.return_value)
