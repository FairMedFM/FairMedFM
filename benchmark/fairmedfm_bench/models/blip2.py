import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess


class BLIP2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _, _ = load_model_and_preprocess("blip2_feature_extractor", "pretrain", is_eval=True)
        self.feat_dim = 768

    def forward_clip(self, images, text_features):
        sample = {"image": images, "text_input": None}
        image_features = self.model.extract_features(sample, mode="image").image_embeds_proj

        text_features = F.normalize(text_features, dim=-1)

        logits, _ = torch.max((image_features @ text_features.T) / self.model.temp, dim=1)

        return logits

    def encode_text(self, text):
        sample = {"image": None, "text_input": text}

        text_features = self.model.extract_features(sample, mode="text").text_embeds_proj[:, 0, :]
        return text_features

    def forward(self, images):
        sample = {"image": images, "text_input": None}
        return self.model.extract_features(sample, mode="image").image_embeds[:, 0, :]

    def from_pretrained(self, path):
        pass
