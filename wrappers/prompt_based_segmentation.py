import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper


class SAMWrapper(BaseWrapper):
    def __init__(self, model, prompt_type, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.model = model
        self.prompt_type = prompt_type

        if self.prompt_type not in ["center", "rand", "rands", "bbox"]:
            raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    def forward(self, x, prompt_point, prompt_bbox):
        f = self.model.encode({"img": x})
        output, iou = self.model.decode({
            "features": f,
            "prompt_point": prompt_point,
            "prompt_box": prompt_bbox}, flag=self.prompt_type)

        mask = output[0][iou.argmax()]

        return mask
