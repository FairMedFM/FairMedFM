"""Adapt SAM 3 image inference to FairMedFM's segmentation trainer."""

import numpy as np
import torch
from torch import nn

from fairmedfm.wrappers.base import BaseWrapper


def _rgb_image(image):
    """Undo Dataset2D's Albumentations ImageNet normalization and BGR input."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("SAM3 requires batch_size=1 and three image channels.")
    bgr = image[0].detach().cpu().float().permute(1, 2, 0).numpy()
    bgr = bgr * np.array([0.229, 0.224, 0.225])
    bgr = bgr + np.array([0.485, 0.456, 0.406])
    return np.rint(np.clip(bgr[..., ::-1] * 255, 0, 255)).astype(np.uint8)


def _points(prompt):
    points = prompt[0].detach().cpu().float().numpy()
    if points.ndim == 1:
        points = points[None, :]
    elif prompt.ndim == 3 and points.shape[0] == 2:
        points = points.T
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"Expected point coordinates as Nx2, got {points.shape}.")
    return points


class SAM3Learner(nn.Module):
    def __init__(self, model, medical=False, data_engine=None):
        super().__init__()
        self.model = model
        self.net = model
        self.medical = medical
        self.data_engine = data_engine
        from sam3.model.sam3_image_processor import Sam3Processor
        self.processor = Sam3Processor(
            model, device=str(self.device),
            confidence_threshold=0.1 if medical else 0.5)
        if not medical and model.inst_interactive_predictor is None:
            raise ValueError("SAM3 was built without instance interactivity.")
        self.is_image_set = False

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def set_torch_image(self, image, original_image_size):
        from PIL import Image

        rgb = _rgb_image(image)
        self.state = self.processor.set_image(Image.fromarray(rgb))
        self.image_size = rgb.shape[:2]
        self.is_image_set = True

    @torch.no_grad()
    def encode(self, data):
        if not self.is_image_set:
            image = data["img"]
            self.set_torch_image(image, image.shape[-2:])
        return True

    @torch.no_grad()
    def decode(self, data, batch_idx=None, flag="point", **kwargs):
        if not self.is_image_set:
            raise RuntimeError("Set an image before predicting a mask.")
        if self.medical:
            if flag != "bbox":
                raise ValueError("MedicalSAM3 supports bbox prompts only.")
            box = data["prompt_box"][0].detach().cpu().float().numpy()
            height, width = self.image_size
            x0, y0, x1, y1 = box
            normalized_box = [
                float((x0 + x1) / (2 * width)),
                float((y0 + y1) / (2 * height)),
                float((x1 - x0) / width),
                float((y1 - y0) / height),
            ]
            self.processor.reset_all_prompts(self.state)
            result = self.processor.add_geometric_prompt(
                state=self.state, box=normalized_box, label=True)
            masks, scores = result["masks"], result["scores"]
            if masks is None or len(masks) == 0:
                masks = torch.zeros((1, 1, height, width), device=self.device)
                scores = torch.zeros(1, device=self.device)
            else:
                masks = torch.as_tensor(masks, device=self.device)
                scores = torch.as_tensor(scores, device=self.device)
                if masks.ndim == 3:
                    masks = masks[:, None]
            return masks[:, 0].unsqueeze(0) > 0, scores

        if flag == "point":
            points = _points(data["prompt_point"])
            masks, scores, _ = self.model.predict_inst(
                self.state,
                point_coords=points,
                point_labels=np.ones(len(points), dtype=np.int32),
                multimask_output=True,
            )
        elif flag == "bbox":
            box = data["prompt_box"][0].detach().cpu().float().numpy()
            masks, scores, _ = self.model.predict_inst(
                self.state, box=box, multimask_output=True)
        else:
            raise ValueError(f"Unsupported SAM3 prompt: {flag}")
        return (torch.as_tensor(masks, device=self.device).unsqueeze(0),
                torch.as_tensor(scores, device=self.device))


class SAM3Wrapper(BaseWrapper):
    def __init__(self, model, data_engine=None, medical=False):
        super().__init__(model)
        self.model = SAM3Learner(model, medical=medical, data_engine=data_engine)
