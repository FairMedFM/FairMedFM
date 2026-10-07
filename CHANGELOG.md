# Changelog

## 0.3.0 (2026-10-06)

- Add `fairmedfm.evaluate` and `fairmedfm.evaluate_segmentation`, which take data as users have it: lists, NumPy,
  pandas or PyTorch; labels as 0/1, booleans, one-hot rows or class names (`pos_label`, one-vs-rest for more than
  two classes); scores as probabilities, logits or softmax matrices; several sensitive attributes at once as a
  DataFrame or dict, with `bins` for continuous attributes and optional intersectional groups; and segmentation
  from per-sample Dice or from masks (arrays or `.npy`, `.npz`, `.png`, `.tif`, `.nii.gz` files). They return a
  `FairnessReport` with pandas `summary` and `by_group` tables and JSON export. Groups with only one label and
  missing attribute values are reported and left out instead of failing the whole evaluation.
- `fairmedfm score` takes the predictions table as an argument, reads CSV, TSV, Parquet, Feather, JSON, JSON Lines
  and Excel, recognizes common column names, joins a separate metadata table (`--metadata`, `--on`), and adds
  `--pos-label`, multiple `--score` columns, `--bins`, `--intersectional`, mask columns, `--format table` and CSV
  output. The earlier options still work.
- Add the `io` extra (pyarrow, Pillow, nibabel, openpyxl) for Parquet, Feather and Excel tables and image and
  NIfTI masks.
- Remove the broken `test.py` and `run_test_seg.sh` scripts.

## 0.2.0 (2026-10-06)

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
- Add a documentation website, https://nanboy-ronan.github.io/FairMedFM-page/docs/, with installation, evaluation, metric,
  command line, Python API, benchmark, model, dataset and FAQ pages, plus search and social metadata.

## 0.1.0 (2026-10-06)

- First pip package: `pip install fairmedfm` provides the FairMedFM fairness metrics for any binary
  classification or segmentation model, with NumPy, pandas and scikit-learn only.
- `fairmedfm score` scores a CSV of per-sample predictions for one or more sensitive attributes;
  `fairmedfm.classification_fairness` and `fairmedfm.segmentation_fairness` provide the Python API.
- Metrics match the paper implementation in `utils/metrics.py` (checked against stored reference values).
  Differences: group labels can be any strings or integers, segmentation supports more than two groups,
  binary cross-entropy runs on the CPU, and a group with only one label raises an error instead of crashing.
