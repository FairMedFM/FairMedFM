import torch
import torch.nn as nn


class ProvGigaPath(nn.Module):
    """Providence's Prov-GigaPath pathology tile encoder (gated on Hugging Face)."""

    def __init__(self, backbone="hf_hub:prov-gigapath/prov-gigapath", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model = self._build(backbone)
        self.feat_dim = 1536

    def _build(self, backbone):
        import timm

        return timm.create_model(backbone, pretrained=True)

    def forward(self, images):
        return self.model(images)

    def from_pretrained(self, path):
        self.backbone = path
        self.model = self._build(path)
