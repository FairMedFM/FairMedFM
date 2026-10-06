# Changelog

## Unreleased

- The benchmark code is now part of the package: `datasets`, `models`, `trainers`, `wrappers`, `utils` and
  `configs` moved under `src/fairmedfm/`. Import them as `fairmedfm.models`, `fairmedfm.datasets`, etc.; the old
  top-level `datasets` package no longer shadows Hugging Face `datasets`.
- `pip install "fairmedfm[cls]"` / `"fairmedfm[seg]"` installs the benchmark runner, and `fairmedfm run` takes the
  same arguments as `python main.py`, which still works in a source checkout. Configs are read from `./configs`
  in the working directory when present, otherwise from the package.
- The trainers use `fairmedfm.metrics`. Compared with the original `utils/metrics.py`: BCE is computed in float64
  on the CPU (differences around 1e-7), groups are taken in sorted order of their values (identical for 0..k-1),
  and segmentation summaries cover every group instead of only groups 0 and 1.
- Fix zero-shot tokenization for BiomedCLIP, which raised `UnboundLocalError` since the CONCH integration.
- Remove `ipdb` and `icecream` imports from library code.
- Add a documentation website, https://fairmedfm.github.io/FairMedFM/, with installation, evaluation, metric,
  command line, Python API, benchmark, model, dataset and FAQ pages, plus search and social metadata.

## 0.1.0 (2026-10-06)

- First pip package: `pip install fairmedfm` provides the FairMedFM fairness metrics for any binary
  classification or segmentation model, with NumPy, pandas and scikit-learn only.
- `fairmedfm score` scores a CSV of per-sample predictions for one or more sensitive attributes;
  `fairmedfm.classification_fairness` and `fairmedfm.segmentation_fairness` provide the Python API.
- Metrics match the paper implementation in `utils/metrics.py` (checked against stored reference values).
  Differences: group labels can be any strings or integers, segmentation supports more than two groups,
  binary cross-entropy runs on the CPU, and a group with only one label raises an error instead of crashing.
