import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip import create_model_from_pretrained, get_tokenizer


class BiomedCLIP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        self.feat_dim = 512

    def forward_clip(self, images, text_features):
        image_features = self.model.encode_image(images, normalize=True)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()

        return logits

    def encode_text(self, text):
        return self.model.encode_text(text.to(next(self.model.parameters()).device), normalize=False)

    def forward(self, images):
        return self.model.visual(images)

    def from_pretrained(self, path):
        pass
