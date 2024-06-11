import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model


class LoRAWarpper(nn.Module):
    def __init__(self, model, lora_targets, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = get_peft_model(model, LoraConfig(lora_targets))

        self.head = torch.nn.Linear(self.encoder.feat_dim, 1)

        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.head(self.encoder(x))
