---
title: Fairness metrics for classification and segmentation
description: Definitions of the FairMedFM fairness metrics - AUC, accuracy, BCE and ECE gaps, worst-group AUC, equal opportunity, equalized odds, Dice gap, Dice skewness and equity-scaled Dice.
---

# Fairness metrics

FairMedFM compares a model's performance across the groups of one sensitive attribute, for example female and
male patients. A **gap** is the largest group value minus the smallest one; 0 means all groups perform the same.

## Binary classification

Computed by `fairmedfm score --task cls` and `fairmedfm.classification_fairness`.

| Metric | Definition | Fairer when |
| --- | --- | --- |
| `overall-auc` | ROC AUC on all samples | - |
| `overall-acc` | Accuracy on all samples at the best-F1 threshold | - |
| `overall-bce` | Binary cross-entropy on all samples | - |
| `overall-ece` | Expected calibration error on all samples | - |
| `worst-auc` | Lowest AUC among the groups | higher |
| `auc-gap` | Largest minus smallest group AUC | lower |
| `acc-gap` | Largest minus smallest group accuracy, at the best-F1 threshold | lower |
| `bce-gap` | Largest minus smallest group binary cross-entropy | lower |
| `ece-gap` | Largest minus smallest group expected calibration error | lower |
| `eo` | Equal opportunity gap: largest minus smallest group true positive rate (TPR) | lower |
| `eod` | Equalized odds score: `1 - (TPR gap + TNR gap) / 2`, where TNR is the true negative rate | higher (1 = equal) |

Details:

- **Threshold.** Accuracy, TPR and TNR use the decision threshold that maximizes F1 on all samples (shared by all
  groups). The detailed results also report every metric at threshold 0.5 (keys without `@best_f1`).
- **ECE** uses 10 equal-width bins spanning the range of the predicted probabilities and weights each bin by its
  number of samples.
- **BCE** clamps log-probabilities at -100, like `torch.nn.BCELoss`, so a probability of exactly 0 or 1 gives a
  finite loss.
- **`eod` is a score, not a gap**: unlike the other fairness metrics, higher is fairer.

## Segmentation

Computed by `fairmedfm score --task seg` and `fairmedfm.segmentation_fairness` from per-sample Dice scores.

| Metric | Definition | Fairer when |
| --- | --- | --- |
| `mean_dice` | Mean Dice over all samples | - |
| `min_dice` | Lowest group mean Dice (worst group) | higher |
| `max_dice` | Highest group mean Dice (best group) | - |
| `delta_dice` | `max_dice - min_dice` | lower |
| `std_dice` | Standard deviation of the group mean Dice scores | lower |
| `skewness_dice` | `(1 - min_dice) / (1 - max_dice)`: how much larger the worst group's error is than the best group's | closer to 1 |
| `es_dice` | Equity-scaled Dice: `mean_dice / (1 + std_dice)` | higher |

`skewness_dice` is `null` when the best group's mean Dice is exactly 1. Attributes can have any number of groups.

## Consistency with the paper

The FairMedFM paper computed these metrics with the benchmark's original `utils/metrics.py`. The package
reproduces its results to within 1e-6, which continuous integration checks against stored reference values.
Differences from the original:

- Group values can be any strings or integers (originally 0, 1, ... only).
- Segmentation metrics use every group (originally only groups 0 and 1).
- Binary cross-entropy runs on the CPU in float64 (originally float32 on CUDA), which changes results by about 1e-7.
- A group with only one label raises a clear error.
