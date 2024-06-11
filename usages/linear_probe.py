import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbeWarpper(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = model

        self.head = torch.nn.Linear(self.encoder.feat_dim, 2)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.head(self.encoder(x))
