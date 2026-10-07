---
title: FairMedFM Python API
description: Python API reference for fairmedfm.evaluate, fairmedfm.evaluate_segmentation and FairnessReport, plus the lower-level fairness metric functions.
---

# Python API

```python
import fairmedfm as fm

report = fm.evaluate(y_true, y_score, sensitive_features)                  # binary classification
report = fm.evaluate_segmentation(sensitive_features, dice=dice)           # segmentation from Dice scores
report = fm.evaluate_segmentation(sensitive_features, pred_masks=p, true_masks=t)   # ... or from masks
```

Both return a [`FairnessReport`](#fairmedfm.evaluation.FairnessReport) with pandas tables (`summary`,
`by_group`) and JSON export. See [Evaluate your model](evaluate-your-model.md) for examples and
[Fairness metrics](metrics.md) for the definitions.

## Fairness evaluation

::: fairmedfm.evaluation.evaluate

::: fairmedfm.evaluation.evaluate_segmentation

::: fairmedfm.evaluation.FairnessReport
    options:
      members: [to_dict, to_json]

## Building blocks

::: fairmedfm.metrics.binary_classification_report

::: fairmedfm.metrics.expected_calibration_error

::: fairmedfm.metrics.binary_cross_entropy

::: fairmedfm.metrics.find_threshold

## Earlier interfaces

These functions from version 0.1 remain available. They take one sensitive attribute, require 0/1 labels and
positive-class probabilities, and raise an error for a group with only one label.

::: fairmedfm.metrics.classification_fairness

::: fairmedfm.metrics.segmentation_fairness

`evaluate_binary`, `organize_results` and `evaluate_seg` keep the interfaces of the original benchmark code and
are used by its trainers. `evaluate_seg` returns `inf` for `skewness_dice` when the best group's mean Dice is 1 and
lets NaN Dice scores propagate, as the original did.

::: fairmedfm.metrics.evaluate_binary

::: fairmedfm.metrics.organize_results

::: fairmedfm.metrics.evaluate_seg
