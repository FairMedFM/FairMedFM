---
title: Foundation models in the FairMedFM benchmark
description: Classification and segmentation foundation models supported by fairmedfm run - CLIP, BiomedCLIP, MedCLIP, DINOv2, DINOv3, SigLIP, RETFound, UNI2-h, SAM, MedSAM, SAM2, SAM3 and more - with install notes.
---

# Models

Pass the name in the first column to `fairmedfm run --model`. "Install" says what provides the model's
dependencies: "runner" is the [benchmark runner](installation.md#benchmark-runner), `[seg]` its segmentation
extra, and "separate" an additional install (see [Installation](installation.md#models-with-separate-installs)).

## Classification

All classification models support linear probing (`--usage lp`). Models marked CLIP-style also support
`clip-zs` and `clip-adapt`.

| `--model` | Model | CLIP-style | Install | Weights |
| --- | --- | --- | --- | --- |
| `CLIP` | OpenAI CLIP | yes | runner | downloaded |
| `BLIP` | BLIP | yes | separate (LAVIS) | downloaded |
| `BLIP2` | BLIP-2 | yes | separate (LAVIS) | downloaded |
| `BiomedCLIP` | BiomedCLIP | yes | runner | downloaded |
| `PubMedCLIP` | PubMedCLIP | yes | runner | downloaded |
| `MedCLIP` | MedCLIP | yes | runner | downloaded |
| `PLIP` | PLIP (pathology) | yes | runner | downloaded |
| `SigLIP` | SigLIP | yes | runner | downloaded |
| `SigLIP2` | SigLIP 2 | yes | runner | downloaded |
| `MedSigLIP` | MedSigLIP | yes | runner | gated |
| `CONCH` | CONCH (pathology) | yes | separate | gated |
| `DINOv2` | DINOv2 | | runner | downloaded |
| `DINOv3` | DINOv3 | | runner | gated |
| `AIMv2` | AIMv2 | | runner | downloaded |
| `RADDINO` | RAD-DINO (chest X-ray) | | runner | downloaded |
| `MedGemma` | MedGemma vision tower | | runner | gated |
| `UNI2` | UNI2-h (pathology) | | runner | gated |
| `Virchow2` | Virchow2 (pathology) | | runner | gated |
| `ProvGigaPath` | Prov-GigaPath (pathology) | | runner | gated |
| `RETFound` | RETFound (ophthalmology) | | runner | gated, local file |
| `Merlin` | Merlin (3D CT) | | separate | downloaded |
| `MedLVM` | LVM-Med | | runner | `pretrained/` |
| `C2L` | C2L | | runner | `pretrained/` |
| `MedMAE` | MedMAE | | runner | `pretrained/` |
| `MoCoCXR` | MoCo-CXR | | runner | `pretrained/` |

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
