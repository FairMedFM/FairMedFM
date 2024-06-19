import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_model import SamLearner

from wrappers.base import BaseWrapper


class SAMWrapper(BaseWrapper):
    def __init__(self, model, data_engine, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = SamLearner(
            sam_model=model, config=None, data_engine=data_engine)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return super().forward(x)

    def get_model(self):
        return self.model
