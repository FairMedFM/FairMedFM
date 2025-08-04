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
        return self.model.forward(input_ids=text_features, pixel_values=images, return_loss=False).logits_per_image

    def encode_text(self, text):
        return self.model.get_text_features(text.to(next(self.model.parameters()).device))

    def forward(self, images):
        return self.model.vision_model(images)

    def from_pretrained(self, path):
        pass