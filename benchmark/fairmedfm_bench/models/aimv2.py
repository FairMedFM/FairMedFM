import torch
import torch.nn as nn
from transformers import Aimv2VisionModel


class AIMv2(nn.Module):
    def __init__(self, backbone="apple/aimv2-large-patch14-native", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model = Aimv2VisionModel.from_pretrained(backbone)
        self.feat_dim = self.model.config.hidden_size

    def forward(self, images):
        outputs = self.model(pixel_values=images)

        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output

        if getattr(outputs, "last_hidden_state", None) is not None:
            return outputs.last_hidden_state.mean(dim=1)

        if isinstance(outputs, torch.Tensor):
            return outputs

        raise RuntimeError("AIMv2 did not return image features.")

    def from_pretrained(self, path):
        self.backbone = path
        self.model = Aimv2VisionModel.from_pretrained(path)
        self.feat_dim = self.model.config.hidden_size
