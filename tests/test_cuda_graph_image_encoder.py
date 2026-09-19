import pytest
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_base import SAM2Base


class _TinyEncoder(torch.nn.Module):
    def forward(self, x):
        y = x * 2 + 1
        return {
            "backbone_fpn": [y],
            "vision_pos_enc": [y.square()],
        }


class _TinyCudaGraphModel(SAM2Base):
    def __init__(self, enabled=True, device="cpu"):
        torch.nn.Module.__init__(self)
        self.image_encoder = _TinyEncoder().to(device)
        self.use_high_res_features_in_sam = False
        self.cuda_graph_image_encoder = enabled
        self._cuda_graph_image_encoder_state = None
        self.eval()


def _assert_output_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        for x, y in zip(left[key], right[key]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_cuda_graph_image_encoder_falls_back_on_cpu():
    model = _TinyCudaGraphModel(device="cpu")
    x = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        got = model.forward_image(x)
        expected = model._forward_image_eager(x)
    _assert_output_equal(got, expected)
    assert model._cuda_graph_image_encoder_state is None


def test_builder_rejects_compile_and_cuda_graph_fast_paths():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        build_sam2_video_predictor(
            "unused.yaml",
            vos_optimized=True,
            cuda_graph_image_encoder=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_image_encoder_parity_ownership_and_recapture():
    model = _TinyCudaGraphModel(device="cuda")
    first = torch.randn(1, 3, 8, 8, device="cuda")
    second = torch.randn_like(first)

    with torch.inference_mode():
        expected_first = model._forward_image_eager(first)
        got_first = model.forward_image(first)
        first_owned_copy = {
            key: [value.clone() for value in values]
            for key, values in got_first.items()
        }
        first_graph = model._cuda_graph_image_encoder_state["graph"]

        expected_second = model._forward_image_eager(second)
        got_second = model.forward_image(second)

        _assert_output_equal(got_first, first_owned_copy)
        _assert_output_equal(got_first, expected_first)
        _assert_output_equal(got_second, expected_second)
        assert model._cuda_graph_image_encoder_state["graph"] is first_graph

        resized = torch.randn(1, 3, 10, 10, device="cuda")
        expected_resized = model._forward_image_eager(resized)
        got_resized = model.forward_image(resized)
        _assert_output_equal(got_resized, expected_resized)
        assert model._cuda_graph_image_encoder_state["graph"] is not first_graph
