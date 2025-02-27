import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPProcessor


class PLIP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = CLIPModel.from_pretrained("vinid/plip")
        self.feat_dim = 768

    def forward_clip(self, images, text_features):
        image_features = self.model.get_image_features(images)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = 100.0 * image_features @ text_features.T

        return logits

    def encode_text(self, text):
        return self.model.get_text_features(text.to(next(self.model.parameters()).device))

    def forward(self, images):
        return self.model.vision_model(images)

    def from_pretrained(self, path):
        pass