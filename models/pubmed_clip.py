import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import clip


class PubMedCLIP(nn.Module):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ckpt can be downloaded from https://onedrive.live.com/?authkey=%21APg2nf5%5Fs4MCi3w&id=132993BDA73EE095%21384&cid=132993BDA73EE095

        self.model, _ = clip.load(backbone, device="cpu", jit=False)
        self.feat_dim = 512

    def forward_clip(self, images, text_features):
        image_features = self.model.encode_image(images)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = np.exp(0.01) / np.exp(0.07) * self.model.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.T

        return logits

    def encode_text(self, text):
        return self.model.encode_text(text.to(next(self.model.parameters()).device))

    def forward(self, images):
        return self.model.visual(images)

    def from_pretrained(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
