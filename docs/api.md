---
title: FairMedFM Python API
description: Python API reference for fairmedfm.classification_fairness and fairmedfm.segmentation_fairness, plus the lower-level metric functions.
---

# Python API

```python
from fairmedfm import classification_fairness, segmentation_fairness
```

Both functions take array-likes (lists, NumPy arrays, pandas Series) and return plain Python dictionaries that
can be saved as JSON. See [Fairness metrics](metrics.md) for the definitions.

## Fairness evaluation

::: fairmedfm.metrics.classification_fairness

::: fairmedfm.metrics.segmentation_fairness

## Building blocks

::: fairmedfm.metrics.binary_classification_report

::: fairmedfm.metrics.expected_calibration_error

::: fairmedfm.metrics.binary_cross_entropy

::: fairmedfm.metrics.find_threshold

## Interfaces used by the benchmark trainers

`evaluate_binary`, `organize_results` and `evaluate_seg` keep the interfaces of the original benchmark code.
`evaluate_seg` returns `inf` for `skewness_dice` when the best group's mean Dice is 1 and lets NaN Dice scores
propagate, as the original did.

::: fairmedfm.metrics.evaluate_binary

::: fairmedfm.metrics.organize_results

::: fairmedfm.metrics.evaluate_seg
