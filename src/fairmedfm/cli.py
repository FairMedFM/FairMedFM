"""Command line interface: ``fairmedfm score`` and ``fairmedfm run``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__

EPILOG = """examples:
  fairmedfm score --task cls --input predictions.csv --sensitive sex age
  fairmedfm score --task seg --input dice.csv --sensitive sex --output fairness.json

classification input: one row per sample with columns prob (positive-class probability),
label (0 or 1) and one column per sensitive attribute.
segmentation input: one row per sample with columns dice (Dice score in [0, 1]) and one column
per sensitive attribute."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fairmedfm", description="Group fairness evaluation for binary "
                                     "classification and segmentation models.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Compute fairness metrics from a CSV of per-sample predictions.",
                                description="Compute fairness metrics from a CSV of per-sample predictions.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    score.add_argument("--task", required=True, choices=["cls", "seg"],
                       help="cls: binary classification; seg: segmentation")
    score.add_argument("--input", required=True, type=Path, help="CSV file, one row per sample")
    score.add_argument("--sensitive", required=True, nargs="+", metavar="COLUMN",
                       help="one or more sensitive attribute columns, e.g. sex age race")
    score.add_argument("--prob-col", default="prob", help="positive-class probability column (cls; default: prob)")
    score.add_argument("--label-col", default="label", help="ground-truth label column (cls; default: label)")
    score.add_argument("--dice-col", default="dice", help="Dice score column (seg; default: dice)")
    score.add_argument("--output", type=Path, help="also write the full result to this JSON file")
    commands.add_parser("run", add_help=False,
                        help="Run a benchmark experiment with a built-in foundation model (needs fairmedfm[cls] or "
                             "fairmedfm[seg]); see fairmedfm run --help.")
    return parser


def _run(argv: List[str]) -> None:
    try:
        from .run import main as run_main
        run_main(argv)
    except ModuleNotFoundError as exc:
        sys.exit(f"fairmedfm: error: missing module {exc.name!r}. fairmedfm run needs the benchmark dependencies: "
                 "pip install 'fairmedfm[cls]' for classification or 'fairmedfm[seg]' for segmentation.")


def score(task: str, input_path: Path, sensitive: List[str], prob_col: str = "prob", label_col: str = "label",
          dice_col: str = "dice") -> Dict[str, Any]:
    import pandas as pd

    from .metrics import classification_fairness, segmentation_fairness

    table = pd.read_csv(input_path)
    needed = [prob_col, label_col] if task == "cls" else [dice_col]
    missing = [column for column in needed + sensitive if column not in table.columns]
    if missing:
        raise ValueError(f"{input_path} has no column(s) {', '.join(missing)}; "
                         f"available: {', '.join(map(str, table.columns))}")
    attributes = {}
    for column in sensitive:
        rows = table.dropna(subset=needed + [column])
        if task == "cls":
            result = classification_fairness(rows[prob_col], rows[label_col], rows[column].astype(str))
        else:
            result = segmentation_fairness(rows[dice_col], rows[column].astype(str))
        attributes[column] = {"n": len(rows), **result}
    return {"fairmedfm_version": __version__, "task": task, "input": str(input_path), "attributes": attributes}


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["run"]:
        # The benchmark runner has its own argument parser.
        _run(argv[1:])
        return
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = score(args.task, args.input, args.sensitive, args.prob_col, args.label_col, args.dice_col)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"fairmedfm: error: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    summaries = {column: value["summary"] for column, value in result["attributes"].items()}
    json.dump(summaries, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
