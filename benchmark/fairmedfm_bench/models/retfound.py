import torch
import torch.nn as nn
import timm


class RETFound(nn.Module):
    """RETFound retinal foundation model (Nature, 2023/2024).

    The backbone is a standard timm ViT (ViT-L/14 DINOv2-style by default, or
    ViT-L/16 for the original MAE-pretrained variant via
    `backbone="vit_large_patch16_224"`). RETFound's own checkpoints are gated
    on Hugging Face (`YukunZhou/RETFound_mae_natureCFP`,
    `RETFound_mae_natureOCT`, `RETFound_dinov2_meh`, `RETFound_dinov2_shanghai`)
    and must be downloaded manually to a local `.pth` file, then referenced via
    `pretrained_path` in configs/models/RETFound.json — the same convention
    used for MedMAE/MoCo-CXR/C2L in this repo.
    """

    def __init__(self, backbone="vit_large_patch14_dinov2.lvd142m", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool="avg", img_size=224
        )
        self.feat_dim = self.model.num_features

    def forward(self, images):
        return self.model(images)

    def from_pretrained(self, path):
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"RETFound checkpoint loaded from {path}: {msg}")
