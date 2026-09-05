# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest

import numpy as np
import torch

from sam2.utils.amg import mask_to_rle_pytorch, rle_to_mask


def reference(mask):
    flat = mask.T.flatten().tolist()
    counts = [len(list(group)) for _, group in itertools.groupby(flat)]
    if flat and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


class RLETests(unittest.TestCase):
    def check_layouts(self, device):
        rng = np.random.default_rng(123)
        for batch, height, width, probability in [
            (0, 8, 8, 0),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
            (8, 7, 11, 0),
            (8, 7, 11, 1),
            (8, 17, 31, 0.1),
            (2, 256, 257, 0.5),
        ]:
            with self.subTest(
                device=device, batch=batch, height=height, probability=probability
            ):
                masks = rng.random((batch, height, width)) < probability
                tensor = torch.as_tensor(masks, device=device).transpose(1, 2)
                expected = [reference(mask.T) for mask in masks]
                self.assertEqual(mask_to_rle_pytorch(tensor), expected)

    def test_cpu_layouts(self):
        self.check_layouts("cpu")

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for the sparse transfer path"
    )
    def test_cuda_layouts(self):
        self.check_layouts("cuda")

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for the crossover paths"
    )
    def test_both_sides_of_sparse_boundary_guard(self):
        for changes in (16383, 16384, 16385, 65535):
            with self.subTest(changes=changes):
                flat = torch.full((2, 256 * 256), bool(changes % 2), dtype=torch.bool)
                flat[:, : changes + 1] = torch.arange(changes + 1) % 2 == 1
                masks = flat.reshape(2, 256, 256).transpose(1, 2)
                result = mask_to_rle_pytorch(masks.cuda())
                expected = [reference(mask.numpy()) for mask in masks]
                self.assertEqual(result, expected)
                for encoded, mask in zip(result, masks):
                    np.testing.assert_array_equal(rle_to_mask(encoded), mask.numpy())


if __name__ == "__main__":
    unittest.main()
