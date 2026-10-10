# Third-party code in the FairMedFM benchmark runner

FairMedFM is licensed under the Apache License 2.0 (`LICENSE`). The files below contain code from other projects
and remain under those projects' licenses. The `fairmedfm` pip package (`src/fairmedfm/`) contains no third-party
code.

| File | Source | License |
| --- | --- | --- |
| `fairmedfm_bench/models/moco_cxr.py` (`MoCo`, `concat_all_gather`) | [facebookresearch/moco](https://github.com/facebookresearch/moco), via [MoCo-CXR](https://github.com/stanfordmlgroup/MoCo-CXR) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only) |
| `fairmedfm_bench/models/c2l.py` | [funnyzhou/C2L_MICCAI2020](https://github.com/funnyzhou/C2L_MICCAI2020), itself based on torchvision's ResNet | MIT; torchvision is BSD-3-Clause |
| `fairmedfm_bench/models/sam_builder/build_sammed2d.py` | [OpenGVLab/SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D), based on [Segment Anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0 |
| `fairmedfm_bench/models/sam_builder/build_tinysam.py` | [xinghaochen/TinySAM](https://github.com/xinghaochen/TinySAM), based on Segment Anything | Apache-2.0 |
| `fairmedfm_bench/models/medlvm.py` (image encoder) | Segment Anything's ViTDet-style encoder, from [detectron2](https://github.com/facebookresearch/detectron2) and [MViT](https://github.com/facebookresearch/mvit) | Apache-2.0 |

`moco_cxr.py`, used only by `--model MoCoCXR`, is the only non-commercial code. Pretrained weights downloaded by
the runner have their own terms, set by each model's authors.
