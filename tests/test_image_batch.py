# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


class ImageBatchTests(unittest.TestCase):
    def test_positional_reuse_is_limited_to_one_call(self):
        class Prompt:
            def __init__(self):
                self.value = torch.ones(1, 1, 2, 2)
                self.calls = 0

            def get_dense_pe(self):
                self.calls += 1
                return self.value.clone()

            def __call__(self, points, boxes, masks):
                value = points[0].sum().reshape(1, 1, 1)
                return value, torch.zeros(1, 1, 2, 2)

        prompt = Prompt()
        seen = []

        def decoder(image_embeddings, image_pe, sparse_prompt_embeddings, **kwargs):
            self.assertEqual(image_pe.device, image_embeddings.device)
            self.assertEqual(image_pe.dtype, image_embeddings.dtype)
            self.assertEqual(image_pe.shape, (1, 1, 2, 2))
            seen.append(image_pe.clone())
            masks = (
                image_embeddings
                + image_pe
                + sparse_prompt_embeddings.reshape(1, 1, 1, 1)
            )
            return masks, torch.ones(1, 1), None, None

        predictor = object.__new__(SAM2ImagePredictor)
        predictor._is_batch = predictor._is_image_set = True
        predictor._features = {
            "image_embed": torch.arange(8.0).reshape(2, 1, 2, 2),
            "high_res_feats": [],
        }
        predictor._orig_hw = [(2, 2), (2, 2)]
        predictor.mask_threshold = 0.0
        predictor.model = SimpleNamespace(
            sam_prompt_encoder=prompt, sam_mask_decoder=decoder
        )
        predictor._transforms = SimpleNamespace(
            postprocess_masks=lambda masks, size: masks
        )
        predictor._prep_prompts = lambda points, labels, *args, **kwargs: (
            None,
            torch.tensor(points).unsqueeze(0),
            torch.tensor(labels).unsqueeze(0),
            None,
        )
        points = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]])]
        labels = [np.array([1]), np.array([1, 0])]
        for value in (1.0, 7.0):
            prompt.value.fill_(value)
            prompt.calls = 0
            actual = predictor.predict_batch(points, labels, return_logits=True)
            self.assertEqual(prompt.calls, 1)
            self.assertTrue(all(torch.equal(pe, prompt.value) for pe in seen[-2:]))
            # Independently compute each image with a fresh positional encoding.
            expected = [[], [], []]
            for index in range(2):
                output = predictor._predict(
                    torch.tensor(points[index]).unsqueeze(0),
                    torch.tensor(labels[index]).unsqueeze(0),
                    img_idx=index,
                    return_logits=True,
                )
                for group, item in zip(expected, output):
                    group.append(item.squeeze(0).float().numpy())
            for group, reference in zip(actual, expected):
                for result, wanted in zip(group, reference):
                    np.testing.assert_array_equal(result, wanted)


if __name__ == "__main__":
    unittest.main()
