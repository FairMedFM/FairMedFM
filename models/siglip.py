import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel


class SigLIP(nn.Module):
    def __init__(self, backbone="google/siglip-base-patch16-224", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model = SiglipModel.from_pretrained(backbone)
        self.feat_dim = self.model.config.vision_config.hidden_size

    def forward_clip(self, images, text_features):
        image_features = F.normalize(self.forward(images), dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = image_features @ text_features.t()

        if hasattr(self.model, "logit_scale"):
            logits = self.model.logit_scale.exp() * logits
        if hasattr(self.model, "logit_bias"):
            logits = logits + self.model.logit_bias

        return logits

    def encode_text(self, text):
        device = next(self.model.parameters()).device
        if hasattr(text, "to"):
            text = text.to(device)

        if isinstance(text, dict):
            return self.model.get_text_features(**text)

        return self.model.get_text_features(input_ids=text.to(device))

    def forward(self, images):
        return self.model.get_image_features(images)

    def from_pretrained(self, path):
        self.backbone = path
        self.model = SiglipModel.from_pretrained(path)
        self.feat_dim = self.model.config.vision_config.hidden_size
