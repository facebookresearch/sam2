# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import tempfile
import unittest

import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


class TestVideoMemoryClearing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the native predictor without downloading pretrained weights. The
        # small image size keeps its real init_state backbone pass CPU-friendly.
        cls.predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=None,
            device="cpu",
            apply_postprocessing=False,
            hydra_overrides_extra=["++model.image_size=64"],
        )
        frames = tempfile.TemporaryDirectory()
        cls.addClassCleanup(frames.cleanup)
        for frame_idx in range(40):
            Image.new("RGB", (8, 6), color=(20, 40, 60)).save(
                f"{frames.name}/{frame_idx:05d}.jpg"
            )
        cls.initial_state = cls.predictor.init_state(
            frames.name, offload_video_to_cpu=True, offload_state_to_cpu=True
        )

    def setUp(self):
        self.predictor.clear_non_cond_mem_around_input = True
        self.predictor.memory_temporal_stride_for_eval = 1

    @staticmethod
    def frame_output(value=1.0):
        # Completed frame outputs isolate native state transitions from learned
        # mask inference; no predictor method or model component is replaced.
        return {
            "maskmem_features": torch.full((1, 64, 4, 4), value),
            "maskmem_pos_enc": [torch.zeros(1, 64, 4, 4)],
            "pred_masks": torch.full((1, 1, 16, 16), value),
            "obj_ptr": torch.zeros(1, 256),
            "object_score_logits": torch.ones(1, 1),
        }

    def make_state(self, obj_ids=(101, 205)):
        state = copy.deepcopy(self.initial_state)
        for obj_id in obj_ids:
            obj_idx = self.predictor._obj_id_to_idx(state, obj_id)
            output = state["output_dict_per_obj"][obj_idx]
            output["cond_frame_outputs"][0] = self.frame_output()
            output["non_cond_frame_outputs"] = {
                t: self.frame_output() for t in range(1, 40)
            }
        return state

    def add_pending_output(self, state, obj_idx, frame_idx, storage_key):
        out = self.frame_output(2.0)
        state["point_inputs_per_obj"][obj_idx][frame_idx] = {
            "point_coords": torch.tensor([[[2.0, 3.0]]]),
            "point_labels": torch.ones(1, 1, dtype=torch.int32),
        }
        state["temp_output_dict_per_obj"][obj_idx][storage_key][frame_idx] = out
        return out

    def test_preflight_clears_only_corrected_object(self):
        state = self.make_state()
        target = state["output_dict_per_obj"][0]
        other = state["output_dict_per_obj"][1]
        original_other = other["non_cond_frame_outputs"].copy()
        original_cond = target["cond_frame_outputs"][0]
        original_ids = state["obj_id_to_idx"].copy()
        correction = self.add_pending_output(state, 0, 16, "cond_frame_outputs")

        self.predictor.propagate_in_video_preflight(state)

        self.assertEqual(
            set(target["non_cond_frame_outputs"]),
            set(range(1, 40)) - set(range(9, 24)),
        )
        self.assertIs(target["cond_frame_outputs"][16], correction)
        self.assertIs(target["cond_frame_outputs"][0], original_cond)
        self.assertEqual(other["non_cond_frame_outputs"], original_other)
        self.assertEqual(state["obj_id_to_idx"], original_ids)
        self.assertEqual(state["obj_ids"], [101, 205])
        self.assertEqual(state["temp_output_dict_per_obj"][0]["cond_frame_outputs"], {})

    def test_preflight_retains_nearby_pending_corrections(self):
        state = self.make_state()
        target = state["output_dict_per_obj"][0]
        corrections = {
            16: self.add_pending_output(state, 0, 16, "non_cond_frame_outputs"),
            17: self.add_pending_output(state, 0, 17, "non_cond_frame_outputs"),
        }
        cond_correction = self.add_pending_output(state, 0, 18, "cond_frame_outputs")

        self.predictor.propagate_in_video_preflight(state)

        for frame_idx, out in corrections.items():
            self.assertIs(target["non_cond_frame_outputs"][frame_idx], out)
        self.assertIs(target["cond_frame_outputs"][18], cond_correction)
        self.assertEqual(
            set(target["non_cond_frame_outputs"]),
            (set(range(1, 40)) - set(range(9, 26))) | {16, 17},
        )
        self.assertEqual(
            set(state["output_dict_per_obj"][1]["non_cond_frame_outputs"]),
            set(range(1, 40)),
        )

    def test_preflight_default_retains_surrounding_memories(self):
        self.predictor.clear_non_cond_mem_around_input = False
        state = self.make_state()
        target = state["output_dict_per_obj"][0]
        correction = self.add_pending_output(state, 0, 16, "non_cond_frame_outputs")

        self.predictor.propagate_in_video_preflight(state)

        self.assertEqual(set(target["non_cond_frame_outputs"]), set(range(1, 40)))
        self.assertIs(target["non_cond_frame_outputs"][16], correction)

    def test_preflight_encodes_and_retains_new_non_conditioning_output(self):
        state = self.make_state()
        correction = self.add_pending_output(state, 0, 16, "non_cond_frame_outputs")
        correction["maskmem_features"] = None
        correction["maskmem_pos_enc"] = None

        self.predictor.propagate_in_video_preflight(state)

        stored = state["output_dict_per_obj"][0]["non_cond_frame_outputs"][16]
        self.assertIs(stored, correction)
        self.assertEqual(tuple(stored["maskmem_features"].shape), (1, 64, 4, 4))
        self.assertEqual(stored["maskmem_features"].device, torch.device("cpu"))
        self.assertIsNotNone(stored["maskmem_pos_enc"])

    def test_clearing_respects_stride_boundaries_and_other_buffers(self):
        for stride in (1, 2):
            for frame_idx in (0, 16, 39):
                with self.subTest(stride=stride, frame_idx=frame_idx):
                    self.predictor.memory_temporal_stride_for_eval = stride
                    state = self.make_state()
                    before = copy.deepcopy(state)
                    target = state["output_dict_per_obj"][1]
                    cond_before = target["cond_frame_outputs"].copy()
                    other_before = state["output_dict_per_obj"][0][
                        "non_cond_frame_outputs"
                    ].copy()
                    radius = stride * self.predictor.num_maskmem

                    self.predictor._clear_obj_non_cond_mem_around_input(
                        state, frame_idx, obj_idx=1
                    )

                    self.assertEqual(
                        set(target["non_cond_frame_outputs"]),
                        {
                            t
                            for t in range(1, 40)
                            if not frame_idx - radius <= t <= frame_idx + radius
                        },
                    )
                    self.assertEqual(target["cond_frame_outputs"], cond_before)
                    self.assertEqual(
                        state["output_dict_per_obj"][0]["non_cond_frame_outputs"],
                        other_before,
                    )
                    for key in (
                        "obj_id_to_idx",
                        "obj_idx_to_id",
                        "obj_ids",
                        "frames_tracked_per_obj",
                        "point_inputs_per_obj",
                        "mask_inputs_per_obj",
                        "temp_output_dict_per_obj",
                    ):
                        self.assertEqual(state[key], before[key])
                    # Clearing an already-empty range is harmless.
                    self.predictor._clear_obj_non_cond_mem_around_input(
                        state, frame_idx, obj_idx=1
                    )

    def test_clearing_uses_current_index_after_object_removal(self):
        state = self.make_state(obj_ids=(101, 205, 309))
        self.predictor.remove_object(state, 101, need_output=False)
        other_before = state["output_dict_per_obj"][1]["non_cond_frame_outputs"].copy()
        obj_idx = state["obj_id_to_idx"][205]
        self.assertEqual(obj_idx, 0)

        self.predictor._clear_obj_non_cond_mem_around_input(state, 16, obj_idx)

        self.assertEqual(
            set(state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]),
            set(range(1, 40)) - set(range(9, 24)),
        )
        self.assertEqual(
            state["output_dict_per_obj"][1]["non_cond_frame_outputs"], other_before
        )
        self.assertEqual(state["obj_ids"], [205, 309])

    def test_propagation_clears_conditioning_frame_memories(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                state = self.make_state(obj_ids=(101,))
                target = state["output_dict_per_obj"][0]
                target["cond_frame_outputs"][16] = self.frame_output(2.0)

                results = list(
                    self.predictor.propagate_in_video(
                        state,
                        start_frame_idx=16,
                        max_frame_num_to_track=0,
                        reverse=reverse,
                    )
                )

                self.assertEqual(len(results), 1)
                frame_idx, obj_ids, masks = results[0]
                self.assertEqual((frame_idx, obj_ids), (16, [101]))
                torch.testing.assert_close(masks, torch.full((1, 1, 6, 8), 2.0))
                self.assertEqual(
                    set(target["non_cond_frame_outputs"]),
                    set(range(1, 40)) - set(range(9, 24)),
                )
                self.assertEqual(
                    state["frames_tracked_per_obj"][0][16], {"reverse": reverse}
                )

    def test_propagation_default_retains_surrounding_memories(self):
        self.predictor.clear_non_cond_mem_around_input = False
        state = self.make_state(obj_ids=(101,))
        target = state["output_dict_per_obj"][0]
        target["cond_frame_outputs"][16] = self.frame_output(2.0)

        results = list(
            self.predictor.propagate_in_video(
                state, start_frame_idx=16, max_frame_num_to_track=0
            )
        )

        self.assertEqual(len(results), 1)
        # Preflight removes only the non-conditioning duplicate of the input frame.
        self.assertEqual(
            set(target["non_cond_frame_outputs"]), set(range(1, 40)) - {16}
        )


if __name__ == "__main__":
    unittest.main()
