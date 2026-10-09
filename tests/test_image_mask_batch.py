# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class TestImageMaskBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20)
        cls.model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            device="cpu",
            apply_postprocessing=False,
        )
        cls.image = np.arange(16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
        cls.predictor = SAM2ImagePredictor(cls.model)
        cls.predictor.set_image(cls.image)

    @staticmethod
    def _mask_inputs(batch_size):
        grid = np.linspace(-2.0, 2.0, 256 * 256, dtype=np.float32).reshape(
            1, 1, 256, 256
        )
        return np.concatenate([grid + 0.25 * index for index in range(batch_size)])

    def _assert_outputs_close(self, actual, expected):
        for output, reference in zip(actual, expected):
            self.assertEqual(output.shape, reference.shape)
            if output.dtype == np.bool_:
                np.testing.assert_array_equal(output, reference)
            else:
                np.testing.assert_allclose(output, reference, rtol=1e-4, atol=2e-6)

    def test_mask_only_prompt_batches_match_individual_predictions(self):
        for batch_size in (1, 2, 3):
            mask_inputs = self._mask_inputs(batch_size)
            for multimask in (False, True):
                for return_logits in (False, True):
                    with self.subTest(
                        batch_size=batch_size,
                        multimask=multimask,
                        return_logits=return_logits,
                    ):
                        individual = [
                            self.predictor.predict(
                                mask_input=mask_inputs[index],
                                multimask_output=multimask,
                                return_logits=return_logits,
                            )
                            for index in range(batch_size)
                        ]
                        expected = tuple(
                            np.stack([result[field] for result in individual])
                            for field in range(3)
                        )
                        if batch_size == 1:
                            expected = tuple(output[0] for output in expected)
                        actual = self.predictor.predict(
                            mask_input=mask_inputs,
                            multimask_output=multimask,
                            return_logits=return_logits,
                        )
                        self._assert_outputs_close(actual, expected)

    def test_batched_box_outputs_can_be_refined_without_more_points(self):
        boxes = np.array([[1, 2, 8, 12], [10, 1, 18, 14]], dtype=np.float32)
        _, _, logits = self.predictor.predict(box=boxes, multimask_output=False)
        self.assertEqual(logits.shape, (2, 1, 256, 256))
        individual = [
            self.predictor.predict(
                mask_input=logits[index], multimask_output=False, return_logits=True
            )
            for index in range(2)
        ]
        expected = tuple(
            np.stack([result[field] for result in individual]) for field in range(3)
        )
        actual = self.predictor.predict(
            mask_input=logits, multimask_output=False, return_logits=True
        )
        self._assert_outputs_close(actual, expected)

    def test_multiple_images_with_different_mask_prompt_batch_sizes(self):
        predictor = SAM2ImagePredictor(self.model)
        predictor.set_image_batch([self.image, self.image[:, ::-1].copy()])
        mask_inputs = [self._mask_inputs(2), self._mask_inputs(3)]
        individual = [
            predictor.predict_batch(
                mask_input_batch=[
                    masks[min(index, len(masks) - 1)] for masks in mask_inputs
                ],
                multimask_output=False,
                return_logits=True,
            )
            for index in range(3)
        ]
        actual = predictor.predict_batch(
            mask_input_batch=mask_inputs,
            multimask_output=False,
            return_logits=True,
        )
        for image_index, batch_size in enumerate((2, 3)):
            expected = tuple(
                np.stack(
                    [
                        individual[index][field][image_index]
                        for index in range(batch_size)
                    ]
                )
                for field in range(3)
            )
            outputs = tuple(actual[field][image_index] for field in range(3))
            self._assert_outputs_close(outputs, expected)


if __name__ == "__main__":
    unittest.main()
