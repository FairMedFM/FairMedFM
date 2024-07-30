import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MedMAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = timm.models.vision_transformer.VisionTransformer()
        self.from_pretrained(path="./pretrained/medmae/vit-b_CXR_0.5M_mae.pth")
        self.model.head = torch.nn.Identity()

        self.feat_dim = 768

    def forward(self, images):
        return self.model(images)

    def from_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = self.model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        self.model.load_state_dict(checkpoint_model, strict=False)
