---
title: Fairness metrics for classification and segmentation
description: Definitions of the FairMedFM fairness metrics - AUC, accuracy, BCE and ECE gaps, worst-group AUC, equal opportunity, equalized odds, Dice gap, Dice skewness and equity-scaled Dice.
---

# Fairness metrics

FairMedFM compares a model's performance across the groups of one sensitive attribute, for example female and
male patients. A **gap** is the largest group value minus the smallest one; 0 means all groups perform the same. Each metric is
available as a function (`fm.auc_gap(y_true, y_score, sensitive_features=group)`) and as a column of
`fm.evaluate(...).summary`.

## Binary classification

Computed by `fairmedfm.evaluate` and `fairmedfm score` for classification.

| Metric | Function | Definition | Fairer when |
| --- | --- | --- | --- |
| `overall-auc` | | ROC AUC on all samples | - |
| `overall-acc` | | Accuracy on all samples at the best-F1 threshold | - |
| `overall-bce` | | Binary cross-entropy on all samples | - |
| `overall-ece` | | Expected calibration error on all samples | - |
| `worst-auc` | `worst_group_auc` | Lowest AUC among the groups | higher |
| `auc-gap` | `auc_gap` | Largest minus smallest group AUC | lower |
| `acc-gap` | `accuracy_gap` | Largest minus smallest group accuracy, at the best-F1 threshold | lower |
| `bce-gap` | `bce_gap` | Largest minus smallest group binary cross-entropy | lower |
| `ece-gap` | `ece_gap` | Largest minus smallest group expected calibration error | lower |
| `eo` | `equal_opportunity_difference` | Equal opportunity gap: largest minus smallest group true positive rate (TPR) | lower |
| `eod` | `equalized_odds_score` | Equalized odds score: `1 - (TPR gap + TNR gap) / 2`, where TNR is the true negative rate | higher (1 = equal) |

Details:

- **Threshold.** Accuracy, TPR and TNR use the decision threshold that maximizes F1 on all samples (shared by all
  groups). The detailed results also report every metric at threshold 0.5 (keys without `@best_f1`).
- **ECE** uses 10 equal-width bins spanning the range of the predicted probabilities and weights each bin by its
  number of samples.
- **BCE** clamps log-probabilities at -100, like `torch.nn.BCELoss`, so a probability of exactly 0 or 1 gives a
  finite loss.
- **`eod` is a score, not a gap**: unlike the other fairness metrics, higher is fairer. It is not Fairlearn's
  `equalized_odds_difference` (0 is fair, the larger of the TPR and FPR gaps by default): on predictions
  binarized at the same threshold, `eod = 1 - equalized_odds_difference(..., agg="mean")`. See
  [FairMedFM, Fairlearn and AIF360](comparison.md#using-fairmedfm-with-fairlearn).

## Segmentation

Computed by `fairmedfm.evaluate_segmentation` and `fairmedfm score` for segmentation, from per-sample Dice scores
(given directly or computed from masks).

| Metric | Function | Definition | Fairer when |
| --- | --- | --- | --- |
| `mean_dice` | | Mean Dice over all samples | - |
| `min_dice` | `worst_group_dice` | Lowest group mean Dice (worst group) | higher |
| `max_dice` | | Highest group mean Dice (best group) | - |
| `delta_dice` | `dice_gap` | `max_dice - min_dice` | lower |
| `std_dice` | `dice_std` | Standard deviation of the group mean Dice scores | lower |
| `skewness_dice` | `dice_skewness` | `(1 - min_dice) / (1 - max_dice)`: how much larger the worst group's error is than the best group's | closer to 1 |
| `es_dice` | `equity_scaled_dice` | Equity-scaled Dice: `mean_dice / (1 + std_dice)` | higher |

`skewness_dice` is empty (`null` in JSON) when the best group's mean Dice is exactly 1. Attributes can have any
number of groups.

## Groups that cannot be evaluated

A classification group whose samples all have the same label has no AUC, TPR or TNR. `evaluate` lists it in
`by_group` with a `skipped` reason, leaves it out of the gaps and `worst-auc`, and adds a warning; the overall
metrics still include its samples. If fewer than two groups remain, the attribute's summary is empty. Samples
with a missing attribute value are left out for that attribute only.

## Consistency with the paper

The FairMedFM paper computed these metrics with the benchmark's original `utils/metrics.py`. The package
reproduces its results to within 1e-6, which continuous integration checks against stored reference values.
Differences from the original:

- Group values can be any strings or integers (originally 0, 1, ... only).
- Segmentation metrics use every group (originally only groups 0 and 1).
- Binary cross-entropy runs on the CPU in float64 (originally float32 on CUDA), which changes results by about 1e-7.
- A group with only one label raises a clear error.
