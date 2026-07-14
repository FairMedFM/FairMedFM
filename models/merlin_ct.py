import torch
import torch.nn as nn


class Merlin(nn.Module):
    """Stanford's Merlin 3D CT + EHR foundation model.

    Requires: pip install merlin-vlm
    Only the image-embedding head is used here (`ImageEmbedding=True`), so
    this is a feature extractor for linear-probe/LoRA style usages on
    volumetric (3D) classification datasets, consistent with how other
    encoder-only models are wrapped in this repo.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = self._build()
        self.feat_dim = 2048

    def _build(self):
        try:
            from merlin import Merlin as _Merlin
        except ImportError as exc:
            raise ImportError(
                "Merlin requires the `merlin-vlm` package: pip install merlin-vlm"
            ) from exc

        return _Merlin(ImageEmbedding=True)

    def forward(self, images):
        return self.model(images)[0]

    def from_pretrained(self, path):
        pass
