# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import sam2.automatic_mask_generator as amg
import torch
import torch.nn.functional as F
from sam2.utils.amg import MaskData, rle_to_mask


def make_generator(refine=False, threshold=0.5, cleanup=0):
    """Synthetic decoder outputs with interior masks and one rejected score."""
    raw = torch.full((2, 1, 4, 4), -50.0)
    raw[0, 0, 1:3, 1:3] = 50.0
    raw[1, 0, 1:2, 1:3] = 50.0
    scores = torch.tensor([[0.9], [0.1]])
    expanded = []

    def postprocess(masks, size):
        expanded.append(masks.shape[0] * masks.shape[1])
        return F.interpolate(masks, size, mode="bilinear", align_corners=False)

    def predict(*args, postprocess=True, **kwargs):
        masks = transforms.postprocess_masks(raw, (16, 16)) if postprocess else raw
        return masks.clone(), scores.clone(), raw.clamp(-32, 32)

    transforms = SimpleNamespace(
        transform_coords=lambda points, **kwargs: points,
        postprocess_masks=postprocess,
    )
    generator = object.__new__(amg.SAM2AutomaticMaskGenerator)
    generator.predictor = SimpleNamespace(
        device=torch.device("cpu"), _transforms=transforms, _predict=predict
    )
    generator.use_m2m = refine
    generator.pred_iou_thresh = threshold
    generator.min_mask_region_area = cleanup
    generator.stability_score_thresh = 0.0
    generator.stability_score_offset = 1.0
    generator.mask_threshold = 0.0
    generator.multimask_output = True
    generator.points_per_batch = 2

    def refine_masks(points, labels, low_res, batch):
        torch.testing.assert_close(
            low_res, raw.flatten(0, 1).clamp(-32, 32), rtol=0, atol=0
        )
        return postprocess(raw, (16, 16)), scores.clone()

    generator.refine_with_m2m = refine_masks
    return generator, expanded


def run_batch(generator):
    return generator._process_batch(
        np.array([[4, 4], [8, 8]], dtype=np.float32),
        (16, 16),
        [0, 0, 16, 16],
        (16, 16),
        normalize=False,
    )


class MaskLifetimeTests(unittest.TestCase):
    def test_downstream_filters_do_not_retain_refinement_logits(self):
        class CheckedData(MaskData):
            def filter(self, keep):
                if "low_res_masks" in self._stats:
                    raise AssertionError(
                        "refinement logits reached a downstream filter"
                    )
                return super().filter(keep)

        for refine in (False, True):
            with self.subTest(refine=refine):
                generator, _ = make_generator(refine=refine)
                with patch.object(amg, "MaskData", CheckedData):
                    result = run_batch(generator)
                self.assertNotIn("low_res_masks", result._stats)
                self.assertEqual(len(result["rles"]), 1)
                expected = np.zeros((16, 16), dtype=bool)
                expected[4:12, 4:12] = True
                # Bilinear interpolation rounds the four corners inward.
                expected[4, 4] = expected[4, 11] = False
                expected[11, 4] = expected[11, 11] = False
                np.testing.assert_array_equal(rle_to_mask(result["rles"][0]), expected)


if __name__ == "__main__":
    unittest.main()
