---
title: Run FairMedFM benchmark experiments
description: Run fairness experiments with medical imaging foundation models - linear probing, CLIP zero-shot, CLIP adaptation and promptable SAM segmentation - using fairmedfm run.
---

# Run experiments

`fairmedfm run` evaluates a built-in foundation model on a dataset and reports its utility and fairness for one
sensitive attribute. Install the runner first:

```bash
pip install "fairmedfm[cls]"   # or "fairmedfm[seg]" for segmentation
```

## Working directory

Run experiments from a directory that contains your data and checkpoints:

```text
my-experiments/
├── data/           # datasets, as referenced by the dataset configs (e.g. data/HAM10000/...)
├── pretrained/     # local checkpoints for C2L, MedMAE, MoCo-CXR, LVM-Med and RETFound
└── configs/        # optional: your own dataset or model configs
```

The packaged configs refer to `./data/...` and `./pretrained/...`. A file in `./configs/datasets/<name>.json` or
`./configs/models/<name>.json` replaces the packaged config of the same name, so you can point to other paths
without changing the package. For example, the HAM10000 config:

```json title="configs/datasets/HAM10000.json"
{
    "train_meta_path": "./data/HAM10000/split/train.csv",
    "test_sex_meta_path": "./data/HAM10000/split/test.csv",
    "test_age_meta_path": "./data/HAM10000/split/test_age.csv",
    "image_train_path": "./data/HAM10000/HAM10000_images",
    "image_test_path": "./data/HAM10000/HAM10000_images",
    "text_template": "a photo of a {}",
    "class_names": ["benign lesion", "malignant lesion"]
}
```

The test split is chosen by the sensitive attribute: `--sensitive_name Age` reads `test_age_meta_path`.
Download the local checkpoints into `pretrained/`:

```bash
wget https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/pretrained.zip
unzip pretrained.zip && rm -f pretrained.zip
```

See [Datasets](datasets.md) for obtaining and preparing data.

## Classification

=== "Linear probing"

    Trains a linear classifier on frozen image features. Works with every classification model.

    ```bash
    fairmedfm run --task cls --usage lp --dataset HAM10000 --model BiomedCLIP --sensitive_name Sex \
      --method erm --total_epochs 100 --warmup_epochs 5 --blr 2.5e-4 --batch_size 128 \
      --optimizer adamw --min_lr 1e-5 --weight_decay 0.05
    ```

=== "CLIP zero-shot"

    Classifies with text prompts built from the dataset's `text_template` and `class_names`; no training.
    Works with CLIP-style models (CLIP, BiomedCLIP, PubMedCLIP, MedCLIP, PLIP, BLIP, BLIP2, SigLIP, SigLIP2,
    MedSigLIP, CONCH).

    ```bash
    fairmedfm run --task cls --usage clip-zs --dataset PAPILA --model MedCLIP --sensitive_name Sex
    ```

=== "CLIP adaptation"

    Trains a lightweight adapter on top of a CLIP-style model.

    ```bash
    fairmedfm run --task cls --usage clip-adapt --dataset PAPILA --model MedCLIP --sensitive_name Sex \
      --total_epochs 100 --blr 2.5e-4 --batch_size 128
    ```

## Segmentation

Promptable 2D segmentation with SAM-family models. Box and point prompts are derived from the ground-truth mask,
so the results measure interactive segmentation. The segmentation trainer evaluates one image at a time
(`--batch_size 1`).

```bash
fairmedfm run --task seg --usage seg2d --dataset TUSC --sensitive_name Sex --method erm \
  --batch_size 1 --pos_class 255 --model SAM --sam_ckpt_path ./pretrained/SAM.pth --img_size 1024 --prompt center
```

| `--prompt` | Prompt given to the model |
| --- | --- |
| `center` | One point at the center of the object |
| `rand` | One random point inside the object |
| `rands` | Several random points inside the object |
| `bbox` | The object's bounding box |

`--pos_class` is the mask value of the target class. See [Models](models.md) for SAM2, SAM3 and MedicalSAM3.

## Outputs

Results are written to `<exp_path>/<task>/<usage>/<method>/<dataset>/<model>/<sensitive_name>/seed<random_seed>/`
(by default under `./output`):

- `history.log`: the arguments and the metrics logged during training and evaluation.
- A folder for the final evaluation (`lp_final`, `clip_zs_final`, `clip_adaptor_final`, or the segmentation
  prompt name) with `metrics.pkl` (overall and per-group metrics) and `predictions.pkl`.

The fairness metrics are those described in [Fairness metrics](metrics.md), computed with the same
`fairmedfm.metrics` code as `fairmedfm score`.

## Current limitations

- `--method erm` is the only training method in this release; the other values in `--help` stop with
  `NotImplementedError`.
- Segmentation datasets other than TUSC need your own config in `./configs/datasets/` with at least `data_path`,
  `train_meta_path` and `test_sex_meta_path`.
- The integrations are checked for imports and configuration in continuous integration, but full runs need data,
  checkpoints and a GPU and are not part of CI.
