"""Checkpoint-free checks for the SAM3 integration's data and API contracts."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from models.sam3 import build_sam3
from wrappers.sam3 import SAM3Learner, _points, _rgb_image


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.inst_interactive_predictor = object()

    def predict_inst(self, state, **kwargs):
        self.last_prompt = kwargs
        return np.ones((3, 4, 8)), np.array([0.2, 0.9, 0.1]), None


class Processor:
    def __init__(self, model, device, confidence_threshold):
        assert device == str(next(model.parameters()).device)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.empty = False

    def set_image(self, image):
        self.image = np.asarray(image)
        return {"original_height": image.height, "original_width": image.width}

    def reset_all_prompts(self, state):
        pass

    def add_geometric_prompt(self, state, box, label):
        self.last_box = box
        assert label is True
        if self.empty:
            return {"masks": torch.empty(0, 1, 4, 8), "scores": torch.empty(0)}
        return {"masks": torch.ones(2, 1, 4, 8, dtype=torch.bool),
                "scores": torch.tensor([0.2, 0.8])}


@pytest.fixture
def sam3_package():
    root = ModuleType("sam3")
    root.__path__ = []
    model = ModuleType("sam3.model")
    model.__path__ = []
    processor = ModuleType("sam3.model.sam3_image_processor")
    processor.Sam3Processor = Processor
    builder = ModuleType("sam3.model_builder")
    builder.build_sam3_image_model = lambda **kwargs: ImageModel()
    modules = {"sam3": root, "sam3.model": model,
               "sam3.model.sam3_image_processor": processor,
               "sam3.model_builder": builder}
    with patch.dict(sys.modules, modules):
        yield builder


def test_dataset_bgr_normalization_round_trip():
    bgr = np.array([[[10, 100, 220]]], dtype=np.float32)
    normalized = ((bgr / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
    image = torch.tensor(normalized.transpose(2, 0, 1)[None])
    np.testing.assert_array_equal(_rgb_image(image)[0, 0], [220, 100, 10])


def test_dataset_points_preserve_xy_layout():
    np.testing.assert_array_equal(_points(torch.tensor([[3, 7]])), [[3, 7]])
    np.testing.assert_array_equal(
        _points(torch.tensor([[[1, 2], [7, 8]]])), [[1, 7], [2, 8]])


@pytest.mark.parametrize("flag", ["point", "bbox"])
def test_interactive_api_and_trainer_mask_selection(sam3_package, flag):
    model = ImageModel()
    learner = SAM3Learner(model)
    learner.set_torch_image(torch.zeros(1, 3, 4, 8), (4, 8))
    masks, scores = learner.decode({
        "prompt_point": torch.tensor([[[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]]]),
        "prompt_box": torch.tensor([[2, 1, 6, 3]]),
    }, flag=flag)
    assert masks.shape == (1, 3, 4, 8)
    assert masks[0][scores.argmax()].shape == (4, 8)
    if flag == "point":
        assert model.last_prompt["point_coords"].shape == (5, 2)
        np.testing.assert_array_equal(model.last_prompt["point_labels"], np.ones(5))
    else:
        np.testing.assert_array_equal(model.last_prompt["box"], [2, 1, 6, 3])


def test_medical_box_geometry_and_empty_output(sam3_package):
    learner = SAM3Learner(ImageModel(), medical=True)
    learner.set_torch_image(torch.zeros(1, 3, 4, 8), (4, 8))
    data = {"prompt_box": torch.tensor([[2, 1, 6, 3]])}
    masks, scores = learner.decode(data, flag="bbox")
    assert learner.processor.last_box == [0.5, 0.5, 0.5, 0.5]
    assert learner.processor.confidence_threshold == 0.1
    assert masks.shape == (1, 2, 4, 8)
    assert scores.argmax() == 1
    learner.processor.empty = True
    masks, scores = learner.decode(data, flag="bbox")
    assert masks.shape == (1, 1, 4, 8)
    assert not masks.any() and scores.item() == 0
    with pytest.raises(ValueError, match="bbox"):
        learner.decode(data, flag="point")


def test_sam3_builder_enables_interactivity_and_download(sam3_package):
    args = SimpleNamespace(model="SAM3", sam_ckpt_path=None, device="cpu")
    with patch.object(sam3_package, "build_sam3_image_model", return_value=ImageModel()) as builder:
        build_sam3(args)
        assert builder.call_args.kwargs["enable_inst_interactivity"] is True
        assert builder.call_args.kwargs["load_from_HF"] is True


@pytest.mark.parametrize("prefix", ["", "detector."])
def test_medical_checkpoint_formats(sam3_package, tmp_path, prefix):
    checkpoint = tmp_path / "medical.pt"
    torch.save({"model": {prefix + "weight": torch.ones(1)}}, checkpoint)
    args = SimpleNamespace(model="MedicalSAM3", prompt="bbox",
                           sam_ckpt_path=str(checkpoint), device="cpu")
    model = build_sam3(args)
    assert model.weight.item() == 1


def test_medical_rejects_unmatched_checkpoint(sam3_package, tmp_path):
    checkpoint = tmp_path / "wrong.pt"
    torch.save({"model": {"unrelated": torch.ones(1)}}, checkpoint)
    args = SimpleNamespace(model="MedicalSAM3", prompt="bbox",
                           sam_ckpt_path=str(checkpoint), device="cpu")
    with pytest.raises(ValueError, match="matches only"):
        build_sam3(args)
