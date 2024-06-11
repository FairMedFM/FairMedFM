import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIP(nn.Module):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _ = clip.load(backbone, device="cpu")
        self.feat_dim = 512

    def forward_clip(self, images, text_features):
        image_features = self.model.encode_image(images)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logits = 100.0 * image_features @ text_features.T

        return logits

    def encode_text(self, text):
        return self.model.encode_text(text.to(next(self.model.parameters()).device))

    def forward(self, images):
        return self.model.visual(images)

    def from_pretrained(self, path):
        pass
