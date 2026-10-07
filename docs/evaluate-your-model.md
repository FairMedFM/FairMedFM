---
title: Evaluate the fairness of your own classification or segmentation model
description: Score any binary classifier or segmentation model for fairness across sex, age, race or other groups. Pass labels, probabilities, logits, masks and metadata as you have them, in Python or from the command line.
---

# Evaluate your model

FairMedFM evaluates predictions, not models: run your model in its own environment and give FairMedFM its
outputs, the ground truth and the patients' attributes. This works for any binary classifier or segmentation
model, in medical imaging or any other domain.

```bash
pip install fairmedfm
pip install "fairmedfm[io]"   # optional: Parquet/Excel tables and PNG, TIFF or NIfTI masks
```

## Classification in Python

```python
import fairmedfm as fm

report = fm.evaluate(y_true=labels, y_score=scores, sensitive_features=meta[["sex", "age"]],
                     bins={"age": [40, 60]})
report.summary     # one row per attribute with the fairness metrics
report.by_group    # every group's sample count and metrics
report.to_json("fairness.json")
```

```text
             n  n_groups  overall-auc  worst-auc  auc-gap  acc-gap    eod     eo
attribute
sex        900         2       0.7939     0.7795   0.0576   0.1112 0.8541 0.0396
age        900         3       0.7939     0.7739   0.0414   0.0158 0.9593 0.0157
```

(Selected columns; `report.summary` also has the overall accuracy, BCE and ECE and the BCE and ECE gaps.)

Each argument accepts the forms your data is likely already in:

| Argument | Accepted forms |
| --- | --- |
| `y_true` | 0/1, booleans, one-hot rows, or any labels (e.g. `"malignant"`) with `pos_label="malignant"`. With more than two classes, `pos_label` evaluates that class against the rest. |
| `y_score` | Positive-class probabilities, logits, or a (samples, classes) matrix of probabilities or logits, e.g. softmax output. Use `score_column` to pick the positive class among more than two columns. |
| `sensitive_features` | A DataFrame (one column per attribute), a dict of name to values, a Series, or a single array. Values can be strings or numbers. |

Lists, NumPy arrays, pandas objects and PyTorch tensors all work, so you can pass model outputs directly:

```python
report = fm.evaluate(targets, torch.softmax(logits, dim=1), {"sex": sex, "site": site})
```

How the input is interpreted:

- **Scores.** Values within [0, 1] are read as probabilities; anything else as logits, converted with a sigmoid
  (one column) or softmax (several columns). `report.warnings` says when this happened. Use
  `score_type="probability"` or `"logit"` to force one reading.
- **Continuous attributes.** A numeric attribute with more than 20 distinct values must be grouped:
  `bins={"age": [40, 60]}` gives `<40`, `40-60` and `>=60`; `bins={"age": 4}` gives quartiles.
- **Missing values.** Samples with a missing attribute value are left out for that attribute only.
- **Small groups.** A group with only one label cannot have an AUC, TPR or TNR. It is listed in
  `report.by_group` with a `skipped` reason and left out of the gaps, and the other groups are still evaluated.
- **Intersections.** `intersectional=True` also evaluates combined groups such as `F & >=60`.

## Segmentation in Python

Give predicted and ground-truth masks and FairMedFM computes Dice per sample, or give Dice scores you already
have:

```python
report = fm.evaluate_segmentation(meta["sex"], pred_masks=pred_paths, true_masks=gt_paths)
report = fm.evaluate_segmentation(meta[["sex", "age"]], dice=dice_scores, bins={"age": [60]})
```

- **Masks** can be a list of arrays or tensors, a stacked array, or file paths: `.npy`, `.npz`, and with
  `fairmedfm[io]` also `.png`, `.jpg`, `.tif` and `.nii`/`.nii.gz`.
- **Foreground** is any non-zero value. For multi-class masks, pass `label=` to evaluate one class.
  Probability masks in [0, 1] are thresholded at `mask_threshold=0.5`.
- **Empty masks.** When both masks of a sample are empty, Dice is `empty_score=1.0`. The benchmark trainer's
  torchmetrics Dice gives 0 in this case; pass `empty_score=0` to match it.
- Samples with a NaN Dice score are left out with a warning.

## From the command line

`fairmedfm score` reads CSV, TSV, Parquet, Feather, JSON, JSON Lines and Excel tables. It recognizes common
column names and prints which columns it used:

```bash
fairmedfm score predictions.csv
```

```text
fairmedfm: sensitive attributes: sex
fairmedfm: classification: label column 'y_true', score column(s) 'y_score'
```

| Role | Recognized names (case-insensitive) | Option |
| --- | --- | --- |
| Label | `label`, `target`, `y_true`, `gt`, `ground_truth`, `truth`, `y` | `--label` |
| Score | `prob`, `probability`, `score`, `y_score`, `y_prob`, `pred`, `logit`, ... or `prob_0`, `prob_1`, ... | `--score` |
| Dice | `dice`, `dsc`, `dice_score` | `--dice` |
| Masks | `pred_mask`, `pred_path` / `gt_mask`, `true_mask`, `gt_path`, `mask` | `--pred-mask`, `--true-mask` |
| Attributes | `sex`, `gender`, `age`, `race`, `ethnicity`, `language`, `site`, `hospital`, `scanner`, ... | `--sensitive` |

Common situations:

```bash
# Patient attributes in a separate file, joined on an ID column
fairmedfm score predictions.parquet --metadata patients.csv --on image_id --sensitive sex age --bins age=40,60

# ID columns with different names in the two files
fairmedfm score preds.csv --metadata meta.csv --on filename=image_path

# Class names as labels and one probability column per class
fairmedfm score preds.csv --label diagnosis --pos-label malignant --score prob_benign prob_malignant

# Segmentation from mask files (paths relative to the table)
fairmedfm score masks.csv --pred-mask pred_path --true-mask gt_path --sensitive sex

# Combined groups, a readable table, and the per-group results as CSV
fairmedfm score preds.csv --sensitive sex race --intersectional --format table --output groups.csv
```

If a column is ambiguous or missing, the error lists the table's columns and the option to use. See the
[command line reference](cli.md) for every option.

## Next steps

- [Fairness metrics](metrics.md): what each number means.
- [Python API](api.md): every parameter of `evaluate` and `evaluate_segmentation`.
