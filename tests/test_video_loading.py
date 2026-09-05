# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.utils.misc import _load_img_as_tensor, load_video_frames_from_jpg_images


class EagerFrameTests(unittest.TestCase):
    def test_all_byte_values_round_identically(self):
        values = np.arange(256, dtype=np.uint8)
        np.testing.assert_array_equal(
            (values / 255.0).astype(np.float32), values.astype(np.float32) / 255.0
        )

    def test_jpeg_values_layout_and_default_helper_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            gray = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
            rgb = np.stack([gray, np.flip(gray, 1), gray // 2], axis=-1)
            for name, array in [("gray", gray), ("rgb", rgb)]:
                with self.subTest(mode=name):
                    path = Path(tmp) / (name + ".jpg")
                    Image.fromarray(array).save(path)
                    image = Image.open(path)
                    pixels = np.array(image.convert("RGB").resize((64, 64)))
                    expected = torch.from_numpy(pixels / 255.0).permute(2, 0, 1).float()
                    result, height, width = _load_img_as_tensor(
                        str(path), 64, use_float32=True
                    )
                    default, _, _ = _load_img_as_tensor(str(path), 64)
                    self.assertEqual(default.dtype, torch.float64)
                    self.assertEqual(result.dtype, torch.float32)
                    self.assertEqual((height, width), (32, 256))
                    self.assertEqual(result.shape, (3, 64, 64))
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_eager_normalization_matches_previous_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            pixels = np.arange(3 * 17 * 19, dtype=np.uint8).reshape(17, 19, 3)
            path = Path(tmp) / "00000.jpg"
            Image.fromarray(pixels).save(path)
            expected, _, _ = _load_img_as_tensor(str(path), 32)
            expected = expected.float().unsqueeze(0)
            expected -= torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            expected /= torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            result, height, width = load_video_frames_from_jpg_images(
                tmp,
                32,
                offload_video_to_cpu=True,
                async_loading_frames=False,
                compute_device=torch.device("cpu"),
            )
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual((height, width), (17, 19))


if __name__ == "__main__":
    unittest.main()
