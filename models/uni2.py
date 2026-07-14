import torch
import torch.nn as nn


class UNI2(nn.Module):
    """Mahmood Lab's UNI2-h pathology foundation model (gated on Hugging Face)."""

    def __init__(self, backbone="hf-hub:MahmoodLab/UNI2-h", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model = self._build(backbone)
        self.feat_dim = 1536

    def _build(self, backbone):
        import timm

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        return timm.create_model(backbone, pretrained=True, **timm_kwargs)

    def forward(self, images):
        return self.model(images)

    def from_pretrained(self, path):
        self.backbone = path
        self.model = self._build(path)
