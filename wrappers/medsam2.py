from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from wrappers.base import BaseWrapper


class MedSAM2Learner(nn.Module):
    def __init__(self, sam2_model, data_engine=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "MedSAM2 requires SAM2ImagePredictor from a SAM2-compatible "
                "installation."
            ) from exc

        self.model = sam2_model
        self.net = sam2_model
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.data_engine = data_engine
        self.features = None
        self.original_size = None
        self.input_size = None
        self.is_image_set = False

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        if transformed_image.shape[0] != 1:
            raise ValueError(
                "MedSAM2 image predictor currently supports batch_size=1 in "
                "the FairMedFM segmentation trainer."
            )

        image = transformed_image[0].detach().cpu()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected BCHW image tensor with 3 channels, got {transformed_image.shape}.")

        image = image.permute(1, 2, 0).float().numpy()
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

        self.predictor.set_image(image)
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        self.features = True
        self.is_image_set = True

    @torch.no_grad()
    def encode(self, data):
        if not self.is_image_set:
            img = data["img"]
            self.set_torch_image(img, img.shape[2:])
        return self.features

    @torch.no_grad()
    def decode(self, data, batch_idx=None, flag="point", **kwargs):
        if not self.is_image_set:
            raise RuntimeError("An image must be set before MedSAM2 mask prediction.")

        point_coords = None
        point_labels = None
        box = None

        if flag == "point":
            point_coords = self._point_prompt_to_numpy(data["prompt_point"])
            point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
        elif flag == "bbox":
            box = self._box_prompt_to_numpy(data["prompt_box"])
        else:
            raise ValueError(f"Unsupported MedSAM2 prompt flag: {flag}")

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
            return_logits=False,
        )

        masks = torch.as_tensor(masks, device=self.device)
        scores = torch.as_tensor(scores, device=self.device)
        if masks.ndim == 3:
            masks = masks.unsqueeze(0)

        return masks, scores

    def _point_prompt_to_numpy(self, prompt_point: torch.Tensor) -> np.ndarray:
        prompt_point = prompt_point.detach().cpu()
        if prompt_point.ndim == 3:
            prompt_point = prompt_point[0]
        if prompt_point.ndim == 2 and prompt_point.shape[0] == 2:
            prompt_point = prompt_point.t()
        if prompt_point.ndim == 1:
            prompt_point = prompt_point.view(1, 2)
        if prompt_point.shape[-1] != 2:
            raise ValueError(f"Expected point prompts with final dimension 2, got {prompt_point.shape}.")
        return prompt_point.float().numpy()

    def _box_prompt_to_numpy(self, prompt_box: torch.Tensor) -> Optional[np.ndarray]:
        prompt_box = prompt_box.detach().cpu()
        if prompt_box.ndim == 3:
            prompt_box = prompt_box[0]
        if prompt_box.ndim == 2:
            prompt_box = prompt_box[0]
        if prompt_box.numel() == 0:
            return None
        if prompt_box.shape[-1] != 4:
            raise ValueError(f"Expected box prompts with final dimension 4, got {prompt_box.shape}.")
        return prompt_box.float().numpy()


class MedSAM2Wrapper(BaseWrapper):
    def __init__(self, model, data_engine=None, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.model = MedSAM2Learner(model, data_engine=data_engine)

    def forward(self, x):
        return super().forward(x)

    def get_model(self):
        return self.model
