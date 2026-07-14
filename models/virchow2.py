import torch
import torch.nn as nn


class Virchow2(nn.Module):
    """Paige AI's Virchow2 pathology tile encoder (gated on Hugging Face)."""

    num_register_tokens = 4

    def __init__(self, backbone="hf-hub:paige-ai/Virchow2", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model = self._build(backbone)
        self.feat_dim = 2560

    def _build(self, backbone):
        import timm
        from timm.layers import SwiGLUPacked

        return timm.create_model(
            backbone, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU
        )

    def forward(self, images):
        output = self.model(images)

        class_token = output[:, 0]
        patch_tokens = output[:, self.num_register_tokens + 1:]

        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

    def from_pretrained(self, path):
        self.backbone = path
        self.model = self._build(path)
