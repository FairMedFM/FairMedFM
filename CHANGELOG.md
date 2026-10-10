# Changelog

## 0.5.0 (2026-10-10)

- **License**: the code, including this package, is now Apache-2.0 (previously CC BY 4.0, which is not meant for
  software). The documentation and figures stay CC BY 4.0.
- **The benchmark runner is a separate package.** `fairmedfm` now contains only the fairness metrics, `evaluate`
  and `fairmedfm score`. The datasets, models, trainers and wrappers moved from `fairmedfm.models`,
  `fairmedfm.datasets`, ... (0.2 to 0.4) to `fairmedfm_bench`, installed from GitHub with
  `pip install "fairmedfm-bench @ git+https://github.com/FairMedFM/FairMedFM#subdirectory=benchmark"` (add `[seg]`
  for segmentation). The `fairmedfm[cls]` and `fairmedfm[seg]` extras no longer exist. `fairmedfm run` still works
  once the runner is installed, and otherwise prints the install command; `python main.py` works in a checkout.
- pandas inputs with the same index labels in a different order are now paired by index, with a note in
  `report.warnings`. Before, they were paired by position, which silently mixed up samples when, for example,
  predictions were shuffled and the metadata table was not.
- The top-level functions are visible to type checkers and editors (they were typed as `Any`), and the
  single-metric functions show their arguments in help, editors and the API reference.
- Single-metric functions compute only what they report: `auc_gap` is about 6 times faster.
- `classification_fairness` and `segmentation_fairness` are deprecated (`FutureWarning`; removal in 1.0). Use
  `evaluate`, `evaluate_segmentation` or the single metrics.
- `find_threshold` and `expected_calibration_error` are reimplemented from their definitions; results are unchanged.
- Benchmark runner: drop the `trans-utils` dependency (it pulled in pyradiomics, which has no wheels after Python
  3.9); C2L ResNets with `pretrained=True` work again with current torchvision; a missing dataset config or a
  sensitive attribute without a test split now fails with an explanation; `--if_wandb False` is no longer read as
  True; the learning-rate schedule is reimplemented with the same values.
- Documentation: how-to recipes, a comparison with Fairlearn and AIF360, and `llms-full.txt` plus Markdown copies
  of every page for AI assistants.

## 0.4.0 (2026-10-06)

- Add one function per fairness metric, in the style of `sklearn.metrics`: `auc_gap`, `worst_group_auc`,
  `accuracy_gap`, `bce_gap`, `ece_gap`, `equal_opportunity_difference`, `equalized_odds_score`, `dice_gap`,
  `worst_group_dice`, `dice_std`, `dice_skewness` and `equity_scaled_dice`. Each takes
  `(y_true, y_score, *, sensitive_features)` (or Dice scores or masks), returns a float equal to the matching
  `evaluate` column, compares combined groups when given several attributes, and works as a scikit-learn scorer
  with metadata routing.
- Lead the README and documentation with these functions on plain arrays, training-loop logging and scikit-learn
  model selection; the command line is presented as one option for prediction files.

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
