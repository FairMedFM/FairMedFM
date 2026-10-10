import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText


class MedGemma(nn.Module):
    """Feature extractor built on MedGemma's SigLIP-based vision tower.

    MedGemma is a generative vision-language model, not a dual encoder, so
    only its vision tower is exposed here (pooled image features) for use
    with linear-probe / LoRA style usages, mirroring how RAD-DINO/DINOv2 are
    wrapped in this repo.
    """

    def __init__(self, backbone="google/medgemma-4b-pt", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self._load_vision_tower(backbone)

    def _load_vision_tower(self, backbone):
        full_model = AutoModelForImageTextToText.from_pretrained(backbone)

        self.vision_tower = full_model.model.vision_tower
        self.feat_dim = full_model.config.vision_config.hidden_size

        del full_model.model.language_model
        del full_model.lm_head

    def forward(self, images):
        outputs = self.vision_tower(pixel_values=images)

        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output

        return outputs.last_hidden_state.mean(dim=1)

    def from_pretrained(self, path):
        self.backbone = path
        self._load_vision_tower(path)
