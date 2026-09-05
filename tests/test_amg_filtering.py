# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sam2.sam2_image_predictor import SAM2ImagePredictor
from test_amg_lifetime import make_generator, run_batch


class EarlyFilteringTests(unittest.TestCase):
    def test_existing_iou_selection_preserves_results_and_skips_expansion(self):
        for threshold, expected_count in [(0.5, 1), (1.0, 0), (0.0, 2)]:
            with self.subTest(threshold=threshold):
                # The synthetic cleanup implementation is identical; its flag
                # selects the original expansion-before-filtering ordering.
                original, full_work = make_generator(threshold=threshold, cleanup=1)
                filtered, kept_work = make_generator(threshold=threshold)
                expected = run_batch(original)
                actual = run_batch(filtered)
                self.assertEqual(actual._stats.keys(), expected._stats.keys())
                for key in actual._stats:
                    if isinstance(actual[key], torch.Tensor):
                        torch.testing.assert_close(
                            actual[key], expected[key], rtol=0, atol=0
                        )
                    else:
                        self.assertEqual(actual[key], expected[key])
                self.assertEqual(sum(full_work), 2)
                self.assertEqual(sum(kept_work), expected_count)

    def test_refinement_and_cleanup_keep_original_expansion(self):
        for refine, cleanup in [(True, 0), (False, 1)]:
            generator, work = make_generator(refine=refine, cleanup=cleanup)
            run_batch(generator)
            self.assertEqual(work[0], 2)

    def test_deferred_logits_are_not_clamped_before_interpolation(self):
        raw = torch.tensor([[[[-50.0, 50.0], [-1.0, 1.0]]]])
        prompt = SimpleNamespace(get_dense_pe=lambda: torch.zeros(1, 1, 1))

        class Prompt:
            def __call__(self, **kwargs):
                return torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)

            get_dense_pe = staticmethod(prompt.get_dense_pe)

        calls = []

        def post(masks, size):
            calls.append(masks.shape)
            return F.interpolate(
                masks.float(), size, mode="bilinear", align_corners=False
            )

        predictor = SimpleNamespace(
            _is_image_set=True,
            _features={"high_res_feats": [], "image_embed": torch.zeros(1, 1, 1, 1)},
            _orig_hw=[(4, 4)],
            mask_threshold=0,
            _transforms=SimpleNamespace(postprocess_masks=post),
            model=SimpleNamespace(
                sam_prompt_encoder=Prompt(),
                sam_mask_decoder=lambda **kwargs: (raw, torch.ones(1, 1), None, None),
            ),
        )
        full, _, refine = SAM2ImagePredictor._predict(
            predictor, None, None, return_logits=True
        )
        deferred, _, refine2 = SAM2ImagePredictor._predict(
            predictor, None, None, return_logits=True, postprocess=False
        )
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(deferred, raw, rtol=0, atol=0)
        torch.testing.assert_close(refine, raw.clamp(-32, 32), rtol=0, atol=0)
        torch.testing.assert_close(refine, refine2, rtol=0, atol=0)
        torch.testing.assert_close(full, post(deferred, (4, 4)), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
