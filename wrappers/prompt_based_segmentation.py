import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_builder import (sam_model_registry, sam_model_registry1,
                         sam_model_registry2)

from wrappers.base import BaseWrapper


class SAM2DWrapper(BaseWrapper):
    def __init__(self, model_name, ckpt_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = self.model_regist(self.model_name, self.ckpt_path)

    def forward(self, x, prompt_point, prompt_bbox, prompt_type):
        f = self.model.encode({"img": x})
        output, iou = self.model.decode({
            "features": f,
            "prompt_point": prompt_point,
            "prompt_box": prompt_bbox}, flag=prompt_type)

        mask = output[0][iou.argmax()]

        return mask

    def model_regist(model_name, ckpt_path):
        '''
            Regist SAM model based on model name.
        '''
        if model_name in ["SAM", "MedSAM"]:
            sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
        elif model_name in ['SAMMed2D', "FT-SAM"]:
            sam = sam_model_registry1["vit_b"]()
        elif model_name in ["TinySAM", "MobileSAM"]:
            sam = sam_model_registry2["vit_t"](checkpoint=ckpt_path)
        else:
            raise ValueError("Invalid Model Name!")

        return sam

    def get_model(self):
        return self.model


class SAM3DWrapper(BaseWrapper):
    def __init__(self, model_name, ckpt_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        pass
        # TODO
