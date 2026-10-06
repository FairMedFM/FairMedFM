# Changelog

## 0.1.0 (unreleased)

- First pip package: `pip install fairmedfm` provides the FairMedFM fairness metrics for any binary
  classification or segmentation model, with NumPy, pandas and scikit-learn only.
- `fairmedfm score` scores a CSV of per-sample predictions for one or more sensitive attributes;
  `fairmedfm.classification_fairness` and `fairmedfm.segmentation_fairness` provide the Python API.
- Metrics match the paper implementation in `utils/metrics.py` (checked against stored reference values).
  Differences: group labels can be any strings or integers, segmentation supports more than two groups,
  binary cross-entropy runs on the CPU, and a group with only one label raises an error instead of crashing.
