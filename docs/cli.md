---
title: FairMedFM command line reference
description: Reference for the fairmedfm score command (fairness metrics from a CSV of predictions) and the fairmedfm run command (benchmark experiments).
---

# Command line

The `fairmedfm` command has two subcommands: `score`, which computes fairness metrics from your predictions,
and `run`, which runs a benchmark experiment with a built-in foundation model.

## `fairmedfm score`

```text
fairmedfm score --task {cls,seg} --input INPUT --sensitive COLUMN [COLUMN ...]
                [--prob-col PROB_COL] [--label-col LABEL_COL] [--dice-col DICE_COL]
                [--output OUTPUT]
```

| Option | Description |
| --- | --- |
| `--task` | `cls` for binary classification, `seg` for segmentation |
| `--input` | CSV file with one row per sample |
| `--sensitive` | One or more sensitive attribute columns; each is evaluated separately |
| `--prob-col` | Positive-class probability column (`cls`; default `prob`) |
| `--label-col` | Ground-truth 0/1 label column (`cls`; default `label`) |
| `--dice-col` | Dice score column (`seg`; default `dice`) |
| `--output` | Also write the full result, including per-group metrics, to this JSON file |

The command prints the fairness summary of each attribute as JSON. For each attribute, rows with an empty value in
the prediction, label or attribute column are left out. The `--output` file has this structure:

```json
{
  "fairmedfm_version": "0.2.0",
  "task": "cls",
  "input": "predictions.csv",
  "attributes": {
    "sex": {
      "n": 1000,
      "summary": {"overall-auc": 0.91, "auc-gap": 0.05, "eod": 0.96, "...": "..."},
      "overall": {"auc": 0.91, "acc@best_f1": 0.83, "...": "..."},
      "groups": {"F": {"n": 512, "auc": 0.93, "...": "..."}, "M": {"n": 488, "auc": 0.88, "...": "..."}}
    }
  }
}
```

The exit status is 2 when the input is invalid, for example a missing column or a group with only one label.
See [Fairness metrics](metrics.md) for the definitions.

## `fairmedfm run`

Runs a benchmark experiment. Requires `pip install "fairmedfm[cls]"` or `"fairmedfm[seg]"`; in a source
checkout, `python main.py` accepts the same arguments.

```bash
fairmedfm run --task cls --usage lp --dataset HAM10000 --model BiomedCLIP --sensitive_name Sex
fairmedfm run --help
```

Main options:

| Option | Values |
| --- | --- |
| `--task` | `cls` (classification) or `seg` (segmentation) |
| `--usage` | `lp` (linear probing), `clip-zs` (CLIP zero-shot), `clip-adapt` (CLIP adaptation), `seg2d` (2D promptable segmentation) |
| `--dataset` | `CXP`, `MIMIC_CXR`, `HAM10000`, `PAPILA`, `ADNI`, `COVID_CT_MD`, `FairVLMed10k`, `BREST`, `GF3300`, `HAM10000-Seg`, `FairSeg`, `montgomery`, `TUSC` |
| `--sensitive_name` | `Sex`, `Age`, `Race` or `Language`, if the dataset provides it |
| `--model` | See [Models](models.md) |
| `--method` | `erm` |
| `--prompt` | Segmentation prompt: `center`, `rand`, `rands` or `bbox` |
| `--sam_ckpt_path`, `--sam2_model_cfg` | Segmentation checkpoint and SAM2 config |
| `--img_size` | Segmentation input size (default 256; 1024 for SAM) |
| `--exp_path` | Output root (default `./output`) |
| `--random_seed` | Random seed (default 0) |
| `--no_cuda` | Run on the CPU |

Training options for `lp` and `clip-adapt`: `--total_epochs`, `--warmup_epochs`, `--blr`, `--min_lr`,
`--batch_size`, `--optimizer` (`sgd`, `adam`, `adamw`) and `--weight_decay`. See [Run experiments](benchmark.md)
for examples and outputs.

The parser also lists `resampling`, `group-dro` and `laftr` for `--method`, and `seg2d-*` usages; these are not
implemented in this release and stop with `NotImplementedError`.
