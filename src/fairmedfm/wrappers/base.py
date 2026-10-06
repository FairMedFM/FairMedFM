import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseWrapper(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        return self.model
