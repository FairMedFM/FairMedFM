---
title: Foundation models in the FairMedFM benchmark
description: Classification and segmentation foundation models supported by fairmedfm run - CLIP, BiomedCLIP, MedCLIP, DINOv2, DINOv3, SigLIP, RETFound, UNI2-h, SAM, MedSAM, SAM2, SAM3 and more - with install notes.
---

# Models

Pass the name in the first column to `fairmedfm run --model`. "Extra" is the pip extra that installs the
model's dependencies; "separate" means an additional install (see [Installation](installation.md#models-with-separate-installs)).

## Classification

All classification models support linear probing (`--usage lp`). Models marked CLIP-style also support
`clip-zs` and `clip-adapt`.

| `--model` | Model | CLIP-style | Install | Weights |
| --- | --- | --- | --- | --- |
| `CLIP` | OpenAI CLIP | yes | `[cls]` | downloaded |
| `BLIP` | BLIP | yes | separate (LAVIS) | downloaded |
| `BLIP2` | BLIP-2 | yes | separate (LAVIS) | downloaded |
| `BiomedCLIP` | BiomedCLIP | yes | `[cls]` | downloaded |
| `PubMedCLIP` | PubMedCLIP | yes | `[cls]` | downloaded |
| `MedCLIP` | MedCLIP | yes | `[cls]` | downloaded |
| `PLIP` | PLIP (pathology) | yes | `[cls]` | downloaded |
| `SigLIP` | SigLIP | yes | `[cls]` | downloaded |
| `SigLIP2` | SigLIP 2 | yes | `[cls]` | downloaded |
| `MedSigLIP` | MedSigLIP | yes | `[cls]` | gated |
| `CONCH` | CONCH (pathology) | yes | separate | gated |
| `DINOv2` | DINOv2 | | `[cls]` | downloaded |
| `DINOv3` | DINOv3 | | `[cls]` | gated |
| `AIMv2` | AIMv2 | | `[cls]` | downloaded |
| `RADDINO` | RAD-DINO (chest X-ray) | | `[cls]` | downloaded |
| `MedGemma` | MedGemma vision tower | | `[cls]` | gated |
| `UNI2` | UNI2-h (pathology) | | `[cls]` | gated |
| `Virchow2` | Virchow2 (pathology) | | `[cls]` | gated |
| `ProvGigaPath` | Prov-GigaPath (pathology) | | `[cls]` | gated |
| `RETFound` | RETFound (ophthalmology) | | `[cls]` | gated, local file |
| `Merlin` | Merlin (3D CT) | | separate | downloaded |
| `MedLVM` | LVM-Med | | `[cls]` | `pretrained/` |
| `C2L` | C2L | | `[cls]` | `pretrained/` |
| `MedMAE` | MedMAE | | `[cls]` | `pretrained/` |
| `MoCoCXR` | MoCo-CXR | | `[cls]` | `pretrained/` |

- **downloaded**: fetched from Hugging Face or the model's source on first use.
- **gated**: request access on the model's Hugging Face page, then `huggingface-cli login` or set `HF_TOKEN`.
- **`pretrained/`**: from `pretrained.zip` (see [Run experiments](benchmark.md#working-directory)).
- **RETFound** ships `.pth` files: download it after access is granted and set `pretrained_path` in
  `configs/models/RETFound.json` in your working directory.

## Segmentation

All segmentation models use `--task seg --usage seg2d` with a `--prompt`.

| `--model` | Model | Install | Checkpoint |
| --- | --- | --- | --- |
| `SAM` | Segment Anything (ViT-B) | `[seg]` | `--sam_ckpt_path` |
| `MedSAM` | MedSAM | `[seg]` | `--sam_ckpt_path` |
| `MobileSAM` | MobileSAM | `[seg]` | `--sam_ckpt_path` |
| `TinySAM` | TinySAM | `[seg]` | `--sam_ckpt_path` |
| `SAMMed2D` | SAM-Med2D | `[seg]` | `--sam_ckpt_path` |
| `FT-SAM` | FT-SAM | `[seg]` | `--sam_ckpt_path` |
| `SAM2` | SAM 2 | `[seg]` + SAM2 | `--sam_ckpt_path`, `--sam2_model_cfg` |
| `MedSAM2` | MedSAM2 | `[seg]` + SAM2 | `--sam_ckpt_path`, `--sam2_model_cfg` |
| `SAM3` | SAM 3 | `[seg]` + SAM3 | downloaded (gated) |
| `MedicalSAM3` | Medical SAM3 (2D, box prompts only) | `[seg]` + SAM3 | `--sam_ckpt_path` to `checkpoint_2D.pt` |

SAM3 requires Python 3.12+ and PyTorch 2.7+; use a separate environment if needed. MedicalSAM3 keeps detections
above confidence 0.1 and uses the highest-scoring mask; an empty detection produces an empty mask.

## Your own model

To evaluate a model that is not listed, you do not need to add it to FairMedFM: run it yourself and score its
predictions with [`fairmedfm score`](evaluate-your-model.md).
