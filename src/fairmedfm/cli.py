"""Command line interface: ``fairmedfm score`` and ``fairmedfm run``."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__

EPILOG = """examples:
  fairmedfm score predictions.csv --sensitive sex age --bins age=40,60
  fairmedfm score predictions.parquet --metadata patients.csv --on image_id --sensitive sex race
  fairmedfm score preds.csv --label diagnosis --pos-label malignant --score prob_benign prob_malignant
  fairmedfm score dice.csv --sensitive sex
  fairmedfm score masks.csv --pred-mask pred_path --true-mask gt_path --sensitive sex

Columns are found by common names (label/target/y_true, prob/score/y_score/logit, prob_0 prob_1 ..., dice,
pred_mask/gt_mask) or set with --label, --score, --dice, --pred-mask and --true-mask. Tables can be CSV, TSV,
Parquet, Feather, JSON, JSON Lines or Excel."""

ROLE_NAMES = {
    "label": ["label", "labels", "y_true", "ytrue", "target", "targets", "gt", "ground_truth", "groundtruth",
              "truth", "true_label", "y"],
    "score": ["prob", "probs", "probability", "proba", "y_prob", "y_proba", "y_score", "score", "scores",
              "pred_prob", "prediction", "predictions", "pred", "y_pred", "output", "logit", "logits", "p"],
    "dice": ["dice", "dsc", "dice_score", "dice_coefficient"],
    "pred_mask": ["pred_mask", "mask_pred", "prediction_mask", "predicted_mask", "pred_mask_path", "pred_path",
                  "segmentation"],
    "true_mask": ["true_mask", "gt_mask", "mask_gt", "label_mask", "ground_truth_mask", "gt_mask_path",
                  "gt_path", "mask", "target_mask"],
}
SCORE_PREFIXES = "prob|probs|probability|proba|score|scores|logit|logits|p|y_prob|y_score|pred|output"
SENSITIVE_NAMES = ["sex", "gender", "age", "age_group", "race", "ethnicity", "language", "site", "hospital",
                   "scanner", "insurance", "skin_type", "fitzpatrick"]
ID_NAMES = ["id", "image_id", "image", "img_id", "filename", "file", "file_name", "path", "image_path",
            "case_id", "patient_id", "subject_id", "sample_id", "study_id", "name"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fairmedfm", description="Fairness evaluation for classification and "
                                     "segmentation models.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Compute fairness metrics from a table of per-sample predictions.",
                                description="Compute fairness metrics from a table of per-sample predictions.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    score.add_argument("predictions", nargs="?", type=Path, help="table with one row per sample")
    score.add_argument("--input", "--predictions", dest="input", type=Path, help=argparse.SUPPRESS)
    data = score.add_argument_group("data")
    data.add_argument("--metadata", type=Path, help="separate table with the sensitive attributes, joined on --on")
    data.add_argument("--on", help="join column, or PRED_COLUMN=METADATA_COLUMN when the names differ "
                                   "(default: the one shared ID-like column)")
    data.add_argument("--sensitive", nargs="+", metavar="COLUMN",
                      help="sensitive attribute columns (default: columns named sex, gender, age, race, ...)")
    data.add_argument("--bins", action="append", default=[], metavar="COLUMN=SPEC",
                      help="group a numeric attribute: age=40,60 (cut points) or age=q4 (quartiles); repeatable")
    data.add_argument("--intersectional", action="store_true", help="also evaluate the combination of attributes")
    data.add_argument("--task", choices=["auto", "cls", "seg"], default="auto",
                      help="cls: binary classification; seg: segmentation (default: from the columns)")
    cls = score.add_argument_group("classification")
    cls.add_argument("--label", "--label-col", dest="label", metavar="COLUMN", help="ground-truth column")
    cls.add_argument("--score", "--prob-col", dest="score", nargs="+", metavar="COLUMN",
                     help="score column(s): one positive-class probability or logit, or one column per class")
    cls.add_argument("--pos-label", help="positive class in the label column, if labels are not 0/1")
    cls.add_argument("--score-type", choices=["auto", "probability", "logit"], default="auto",
                     help="how to read scores (default: probabilities if within [0, 1], otherwise logits)")
    cls.add_argument("--score-column", type=int, metavar="INDEX",
                     help="position (from 0) of the positive class among several --score columns (default: the "
                          "column named after --pos-label, else sorted class order)")
    seg = score.add_argument_group("segmentation")
    seg.add_argument("--dice", "--dice-col", dest="dice", metavar="COLUMN", help="per-sample Dice column")
    seg.add_argument("--pred-mask", metavar="COLUMN", help="column of predicted mask file paths")
    seg.add_argument("--true-mask", metavar="COLUMN", help="column of ground-truth mask file paths")
    seg.add_argument("--mask-label", help="class value to evaluate in multi-class masks (default: any non-zero)")
    seg.add_argument("--mask-threshold", type=float, default=0.5, help="threshold for probability masks")
    seg.add_argument("--empty-score", type=float, default=1.0,
                     help="Dice when both masks are empty (default 1; the benchmark trainer uses 0)")
    out = score.add_argument_group("output")
    out.add_argument("--output", type=Path, help="write the full result: .json, or .csv for the per-group table")
    out.add_argument("--format", choices=["json", "table"], default="json", help="what to print (default: json)")
    commands.add_parser("run", add_help=False,
                        help="Run a FairMedFM benchmark experiment with a built-in foundation model (needs the "
                             "benchmark runner from GitHub); see fairmedfm run --help.")
    return parser


BENCHMARK_URL = "git+https://github.com/FairMedFM/FairMedFM#subdirectory=benchmark"


def _run(argv: List[str]) -> None:
    try:
        from fairmedfm_bench.run import main as run_main
        run_main(argv)
    except ModuleNotFoundError as exc:
        if exc.name == "fairmedfm_bench":
            sys.exit("fairmedfm: error: fairmedfm run needs the benchmark runner, which is installed from GitHub:\n"
                     f'  pip install "fairmedfm-bench @ {BENCHMARK_URL}"        # classification\n'
                     f'  pip install "fairmedfm-bench[seg] @ {BENCHMARK_URL}"   # + segmentation')
        sys.exit(f"fairmedfm: error: missing module {exc.name!r}. Models whose packages are not on PyPI need a "
                 "separate install; see https://fairmedfm.github.io/FairMedFM/docs/models/")


def score(args: argparse.Namespace):
    """Read the tables, work out the columns, and evaluate. Returns the report and notes for the user."""
    from ._inputs import read_table
    from .evaluation import evaluate, evaluate_segmentation

    path = args.predictions or args.input
    if path is None:
        raise ValueError("give the predictions table, e.g. fairmedfm score predictions.csv --sensitive sex")
    table = read_table(path)
    notes: List[str] = []
    if args.metadata:
        table = _join(table, read_table(args.metadata), args.on, notes)
    columns = list(table.columns)

    sensitive = args.sensitive or _find_sensitive(columns)
    notes.append(f"sensitive attributes: {', '.join(sensitive)}")
    _require(columns, sensitive, "--sensitive")
    bins = dict(_parse_bins(spec) for spec in args.bins)

    dice = args.dice or _find(columns, "dice", required=False)
    pred_mask = args.pred_mask or _find(columns, "pred_mask", required=False)
    true_mask = args.true_mask or _find(columns, "true_mask", required=False)
    label = args.label or _find(columns, "label", required=False)
    scores = args.score or _find_scores(columns)
    task = args.task
    if task == "auto":
        has_seg, has_cls = bool(dice or (pred_mask and true_mask)), bool(label and scores)
        if has_seg and has_cls:
            raise ValueError("the table has both classification and segmentation columns; pass --task cls or seg")
        if not has_seg and not has_cls:
            raise ValueError("no prediction columns found; name them with --label and --score (classification) "
                             f"or --dice / --pred-mask and --true-mask (segmentation). Columns: {columns}")
        task = "seg" if has_seg else "cls"

    if task == "cls":
        label = label or _find(columns, "label")
        assert label is not None  # _find raises when the column is required and missing
        if not scores:
            raise ValueError(f"no score column found; name it with --score. Columns: {columns}")
        _require(columns, [label, *scores], "--label/--score")
        notes.append(f"classification: label column {label!r}, score column(s) {', '.join(map(repr, scores))}")
        rows = table.dropna(subset=[label, *scores])
        if len(rows) < len(table):
            notes.append(f"{len(table) - len(rows)} rows with a missing label or score were left out")
        y_score = rows[scores[0]] if len(scores) == 1 else rows[scores].to_numpy()
        score_column = args.score_column
        if score_column is None and args.pos_label is not None and len(scores) > 1:
            named = [i for i, c in enumerate(scores) if _normalize(args.pos_label) in _normalize(c)]
            if len(named) == 1:
                score_column = named[0]
                notes.append(f"positive class {args.pos_label!r}: score column {scores[score_column]!r}")
        report = evaluate(rows[label], y_score, rows[sensitive], pos_label=_typed(args.pos_label, rows[label]),
                          score_type=args.score_type, score_column=score_column, bins=bins,
                          intersectional=args.intersectional)
    else:
        if dice:
            _require(columns, [dice], "--dice")
            notes.append(f"segmentation: Dice column {dice!r}")
            report = evaluate_segmentation(table[sensitive], dice=table[dice], bins=bins,
                                           intersectional=args.intersectional)
        else:
            if not (pred_mask and true_mask):
                raise ValueError("segmentation needs --dice, or both --pred-mask and --true-mask")
            _require(columns, [pred_mask, true_mask], "--pred-mask/--true-mask")
            notes.append(f"segmentation: masks from columns {pred_mask!r} and {true_mask!r}")
            base = Path(path).resolve().parent
            report = evaluate_segmentation(table[sensitive], pred_masks=table[pred_mask].map(lambda p: base / p),
                                           true_masks=table[true_mask].map(lambda p: base / p),
                                           label=_typed_number(args.mask_label),
                                           mask_threshold=args.mask_threshold, empty_score=args.empty_score,
                                           bins=bins, intersectional=args.intersectional)
    report.metadata["input"] = str(path)
    return report, notes


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["run"]:
        # The benchmark runner has its own argument parser.
        _run(argv[1:])
        return
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report, notes = score(args)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"fairmedfm: error: {_cli_message(str(exc))}\n")
    for note in notes:
        print(f"fairmedfm: {note}", file=sys.stderr)
    for warning in report.warnings:
        print(f"fairmedfm: warning: {warning}", file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            report.by_group.to_csv(args.output)
        else:
            report.to_json(args.output)
    if args.format == "table":
        print(report.summary.to_string(float_format=lambda v: f"{v:.4f}"))
    else:
        result = report.to_dict()
        json.dump({name: value["summary"] for name, value in result["attributes"].items()}, sys.stdout, indent=2)
        sys.stdout.write("\n")


def _cli_message(message: str) -> str:
    """Python API wording in error messages, translated to the command-line options."""
    message = re.sub(r"bins=\{'([^']+)': \[40, 60\]\} \(cut points\) or bins=\{'[^']+': 4\} \(quantiles\)",
                     r"--bins \1=40,60 (cut points) or --bins \1=q4 (quartiles)", message)
    for api, option in [("pos_label", "--pos-label"), ("score_column", "--score-column"),
                        ("score_type='probability'", "--score-type probability"), ("y_true", "the label column"),
                        ("y_score", "the score column")]:
        message = message.replace(api, option)
    return message


def _normalize(name: str) -> str:
    return re.sub(r"[\s\-.]+", "_", str(name).strip().lower())


def _find(columns: List[str], role: str, required: bool = True) -> Optional[str]:
    names = ROLE_NAMES[role]
    matches = [c for c in columns if _normalize(c) in names]
    if len(matches) > 1:
        raise ValueError(f"several columns could be the {role.replace('_', ' ')}: {matches}; choose one with "
                         f"--{role.replace('_', '-')}")
    if not matches and required:
        raise ValueError(f"no {role.replace('_', ' ')} column found; name it with --{role.replace('_', '-')}. "
                         f"Columns: {columns}")
    return matches[0] if matches else None


def _find_scores(columns: List[str]) -> Optional[List[str]]:
    single = _find(columns, "score", required=False)
    if single:
        return [single]
    numbered: Dict[str, List[Tuple[int, str]]] = {}
    for column in columns:
        match = re.fullmatch(rf"({SCORE_PREFIXES})_?(\d+)", _normalize(column))
        if match:
            numbered.setdefault(match.group(1), []).append((int(match.group(2)), column))
    if len(numbered) > 1:
        raise ValueError(f"several groups of score columns: {sorted(numbered)}; choose with --score")
    if numbered:
        return [column for _, column in sorted(next(iter(numbered.values())))]
    return None


def _find_sensitive(columns: List[str]) -> List[str]:
    found = [c for c in columns if _normalize(c) in SENSITIVE_NAMES]
    if not found:
        raise ValueError(f"no sensitive attribute columns found; name them with --sensitive. Columns: {columns}")
    return found


def _require(columns: List[str], wanted: List[str], option: str) -> None:
    missing = [c for c in wanted if c not in columns]
    if missing:
        raise ValueError(f"no column(s) {', '.join(missing)} (from {option}); available: {', '.join(columns)}")


def _join(predictions, metadata, on: Optional[str], notes: List[str]):
    if on and "=" in on:
        left, right = on.split("=", 1)
    elif on:
        left = right = on
    else:
        shared = [c for c in predictions.columns if c in set(metadata.columns)]
        ids = [c for c in shared if _normalize(c) in ID_NAMES]
        candidates = ids if ids else shared
        if len(candidates) != 1:
            raise ValueError(f"cannot tell which column joins the two tables (shared: {shared}); pass --on COLUMN "
                             "or --on PRED_COLUMN=METADATA_COLUMN")
        left = right = candidates[0]
    _require(list(predictions.columns), [left], "--on")
    _require(list(metadata.columns), [right], "--on")
    if metadata[right].duplicated().any():
        raise ValueError(f"metadata column {right!r} has duplicate values; it must identify each sample once")
    keys_p, keys_m = predictions[left].astype(str), metadata[right].astype(str)
    metadata = metadata.assign(**{right: keys_m})
    joined = predictions.assign(**{left: keys_p}).merge(metadata, left_on=left, right_on=right, how="inner",
                                                        suffixes=("", "_metadata"))
    unmatched = int((~keys_p.isin(set(keys_m))).sum())
    notes.append(f"joined predictions and metadata on {left!r}={right!r}: {len(joined)} rows")
    if unmatched:
        notes.append(f"{unmatched} prediction rows had no metadata and were left out")
    return joined


def _parse_bins(spec: str) -> Tuple[str, Any]:
    """COLUMN=40,60 (cut points) or COLUMN=q4 (quantile groups)."""
    column, _, value = spec.partition("=")
    value = value.strip().lower()
    try:
        if value.startswith("q"):
            return column, int(value[1:])
        return column, [float(p) for p in value.split(",") if p.strip()]
    except ValueError:
        pass
    raise ValueError(f"--bins {spec!r}: use COLUMN=40,60 (cut points) or COLUMN=q4 (quantile groups)")


def _typed(value: Optional[str], column) -> Any:
    """The --pos-label string as the label column's type (e.g. 1 for an integer column)."""
    if value is None:
        return None
    if column.dtype.kind == "b":
        return value.lower() in ("1", "true", "yes")
    if column.dtype.kind in "iuf":
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def _typed_number(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


if __name__ == "__main__":
    main()
