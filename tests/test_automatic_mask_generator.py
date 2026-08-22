# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types

import numpy as np
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import MaskData


class _DummyModel:
    """Stand-in for a SAM2Base model.

    `SAM2ImagePredictor.__init__` only reads `image_size` off the model at
    construction time; no forward pass is required for this test since both
    `predictor.set_image` and `_process_batch` are stubbed out below.
    """

    image_size = 1024


def _empty_process_batch(self, points, im_size, crop_box, orig_size, normalize=False):
    """Stand-in for `_process_batch` simulating zero masks surviving the
    predicted-IoU / stability-score filters for this batch of points (e.g. a
    blank crop, or strict thresholds) -- without requiring a real checkpoint.
    """
    return MaskData(
        rles=[],
        boxes=torch.zeros((0, 4)),
        iou_preds=torch.zeros((0,)),
        points=torch.zeros((0, 2)),
    )


def _make_generator_with_no_detections(crop_n_layers: int) -> SAM2AutomaticMaskGenerator:
    generator = SAM2AutomaticMaskGenerator(
        model=_DummyModel(),
        points_per_side=2,
        crop_n_layers=crop_n_layers,
    )
    # Avoid running a real forward pass through the (dummy) model.
    generator.predictor.set_image = lambda image: None
    generator._process_batch = types.MethodType(_empty_process_batch, generator)
    return generator


def test_generate_masks_with_zero_detections_across_multiple_crops():
    """Regression test for GitHub issue #681.

    When `crop_n_layers > 0` splits the image into more than one crop and
    every crop is filtered down to zero surviving masks, `_generate_masks`
    must not crash while deduplicating masks across crops (it previously
    raised `IndexError` from `box_area(data["crop_boxes"])` because
    `data["crop_boxes"]` collapsed to a 1-D empty tensor instead of shape
    `(0, 4)`).
    """
    generator = _make_generator_with_no_detections(crop_n_layers=1)
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    mask_data = generator._generate_masks(image)

    assert len(mask_data["rles"]) == 0


def test_generate_masks_with_single_crop_and_zero_detections():
    """Sanity check: the default `crop_n_layers=0` path (single crop, so the
    cross-crop dedup branch is skipped entirely) must keep working too.
    """
    generator = _make_generator_with_no_detections(crop_n_layers=0)
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    mask_data = generator._generate_masks(image)

    assert len(mask_data["rles"]) == 0
