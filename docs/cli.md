---
title: FairMedFM command line reference
description: Reference for the fairmedfm score command (fairness metrics from tables of predictions, labels, masks and patient metadata) and the fairmedfm run command (benchmark experiments).
---

# Command line

The `fairmedfm` command has two subcommands: `score`, which computes fairness metrics from your predictions,
and `run`, which runs a benchmark experiment with a built-in foundation model.

## `fairmedfm score`

```text
fairmedfm score PREDICTIONS [--metadata FILE] [--on COLUMN] [--sensitive COLUMN ...] [options]
```

Reads a table with one row per sample (CSV, TSV, Parquet, Feather, JSON, JSON Lines or Excel; Parquet, Feather
and Excel need `fairmedfm[io]`), finds the prediction columns, evaluates every sensitive attribute and prints the
summaries as JSON. It prints which columns it used, and any warnings, to standard error.

### Data

| Option | Description |
| --- | --- |
| `PREDICTIONS` | The predictions table (`--input` and `--predictions` also work) |
| `--metadata FILE` | A second table with the sensitive attributes, joined to the predictions |
| `--on COLUMN` | Join column; `PRED_COLUMN=METADATA_COLUMN` when the names differ. Default: the single shared ID-like column (`id`, `image_id`, `filename`, `patient_id`, ...). Predictions without metadata are left out and counted. |
| `--sensitive COLUMN ...` | Sensitive attribute columns. Default: columns named `sex`, `gender`, `age`, `race`, `ethnicity`, `language`, `site`, `hospital`, `scanner`, ... |
| `--bins COLUMN=SPEC` | Group a numeric attribute: `age=40,60` (cut points: `<40`, `40-60`, `>=60`) or `age=q4` (quartiles). Repeatable. Required for numeric attributes with more than 20 distinct values. |
| `--intersectional` | Also evaluate the combination of the attributes |
| `--task` | `cls` or `seg`; by default inferred from the columns |

### Classification

| Option | Description |
| --- | --- |
| `--label COLUMN` | Ground truth. Default: a column named `label`, `target`, `y_true`, `gt`, `ground_truth`, `truth` or `y` |
| `--score COLUMN ...` | One column of positive-class probabilities or logits, or one column per class (e.g. softmax output). Default: a column named `prob`, `probability`, `score`, `y_score`, `y_prob`, `pred`, `logit`, ..., or numbered columns such as `prob_0 prob_1` |
| `--pos-label VALUE` | The positive class, when labels are not 0/1 (e.g. `malignant`). With more than two classes, that class is evaluated against the rest. |
| `--score-column INDEX` | Position (from 0) of the positive class among several `--score` columns. Default: the column whose name contains `--pos-label`, otherwise sorted class order as in scikit-learn. |
| `--score-type` | `auto` (default: probabilities if all scores are in [0, 1], otherwise logits), `probability` or `logit` |

### Segmentation

| Option | Description |
| --- | --- |
| `--dice COLUMN` | Per-sample Dice scores. Default: a column named `dice`, `dsc` or `dice_score` |
| `--pred-mask COLUMN`, `--true-mask COLUMN` | Columns of mask file paths, relative to the table's folder (`.npy`, `.npz`; `.png`, `.jpg`, `.tif`, `.nii`, `.nii.gz` with `fairmedfm[io]`). Defaults: `pred_mask`, `pred_path` and `gt_mask`, `true_mask`, `gt_path`, `mask`. |
| `--mask-label VALUE` | Class to evaluate in multi-class masks (default: any non-zero value) |
| `--mask-threshold` | Threshold for probability masks (default 0.5) |
| `--empty-score` | Dice when both masks are empty (default 1; the benchmark trainer uses 0) |

### Output

| Option | Description |
| --- | --- |
| `--output FILE` | `.json`: the full result with per-group metrics and warnings; `.csv`: the per-group table |
| `--format` | `json` (default) prints the summaries as JSON; `table` prints a readable table |

The `--output` JSON has this structure:

```json
{
  "fairmedfm_version": "0.3.0",
  "task": "cls",
  "metadata": {"n_samples": 1000, "input": "predictions.csv"},
  "overall": {"auc": 0.91, "acc@best_f1": 0.83, "...": "..."},
  "attributes": {
    "sex": {
      "n": 1000,
      "summary": {"overall-auc": 0.91, "auc-gap": 0.05, "eod": 0.96, "...": "..."},
      "overall": {"auc": 0.91, "...": "..."},
      "groups": {"F": {"n": 512, "auc": 0.93, "...": "...", "skipped": null}, "M": {"...": "..."}}
    }
  },
  "warnings": []
}
```

The exit status is 2 when the input cannot be used, for example a missing or ambiguous column; the message lists
the table's columns and the option to use. See [Evaluate your model](evaluate-your-model.md) for examples and
[Fairness metrics](metrics.md) for the definitions.

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
