import torch
import torch.nn as nn
import torch.nn.functional as F


class CONCH(nn.Module):
    """Mahmood Lab's CONCH pathology vision-language model (gated on Hugging Face).

    Requires: pip install git+https://github.com/Mahmoodlab/CONCH.git
    """

    def __init__(self, backbone="hf_hub:MahmoodLab/conch", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.model, self._tokenizer = self._build(backbone)
        self.feat_dim = 512

    def _build(self, backbone):
        try:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        except ImportError as exc:
            raise ImportError(
                "CONCH requires the `conch` package: "
                "pip install git+https://github.com/Mahmoodlab/CONCH.git"
            ) from exc

        model, _ = create_model_from_pretrained("conch_ViT-B-16", backbone)
        return model, get_tokenizer()

    def forward_clip(self, images, text_features):
        image_features = F.normalize(self.forward(images), dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features @ text_features.t()

    def encode_text(self, text):
        device = next(self.model.parameters()).device

        if isinstance(text, str) or (isinstance(text, (list, tuple)) and isinstance(text[0], str)):
            from conch.open_clip_custom import tokenize

            text = tokenize(texts=text, tokenizer=self._tokenizer)

        return self.model.encode_text(text.to(device))

    def forward(self, images):
        return self.model.encode_image(images, proj_contrast=False, normalize=False)

    def from_pretrained(self, path):
        self.backbone = path
        self.model, self._tokenizer = self._build(path)
