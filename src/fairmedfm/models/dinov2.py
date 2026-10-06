import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.feat_dim = 768

    def forward(self, images):
        return self.model(images)

    def from_pretrained(self, path):
        pass
