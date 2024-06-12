import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper


class CLIPWrapper(BaseWrapper):
    def __init__(self, model, base_text_features, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.model = model
        self.base_text_features = base_text_features

        self.prototypes = nn.Parameter(base_text_features.clone())

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model.forward_clip(x, self.prototypes)

    def load_prototype(self, prototype):
        device = self.prototypes.device
        self.prototypes = nn.Parameter(prototype.clone().to(device))

    def get_prototype(self):
        return self.prototypes
