"""One fairness number per call, in the style of ``sklearn.metrics``.

Each function compares a model's performance across groups and returns a float, so it can be logged during
training, used for model selection or wrapped with ``sklearn.metrics.make_scorer``. The values are the same as
the corresponding columns of :func:`fairmedfm.evaluate` and :func:`fairmedfm.evaluate_segmentation`.

When ``sensitive_features`` has several columns, groups are their combinations (for example ``F & >=60``).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Iterable, List, Optional

import pandas as pd

from . import _inputs
from .evaluation import _evaluate, evaluate_segmentation

__all__ = [
    "auc_gap", "worst_group_auc", "accuracy_gap", "bce_gap", "ece_gap", "equal_opportunity_difference",
    "equalized_odds_score", "dice_gap", "worst_group_dice", "dice_std", "dice_skewness", "equity_scaled_dice",
]


# The per-group metrics each summary column needs, so that one function does not compute all of them.
_NEEDS = {"auc-gap": {"auc"}, "worst-auc": {"auc"}, "acc-gap": {"acc@best_f1"}, "bce-gap": {"bce"},
          "ece-gap": {"ece"}, "eo": {"tpr@best_f1"}, "eod": {"tpr@best_f1", "tnr@best_f1"}}


def _classification(column: str, y_true: Any, y_score: Any, sensitive_features: Any, **options: Any) -> float:
    notes: List[str] = []
    y_true, y_score, sensitive_features = _inputs.align_by_index(
        {"y_true": y_true, "y_score": y_score, "sensitive_features": sensitive_features}, notes)
    attribute = _one_attribute(sensitive_features, len(_inputs.to_numpy(y_true, "y_true")), options.pop("bins", None))
    _keep_pairing(attribute, y_true, y_score)
    report = _evaluate(y_true, y_score, attribute, wanted=_NEEDS[column], **options)
    report.warnings[:0] = notes
    return _value(report, column)


def _segmentation(column: str, dice: Any, sensitive_features: Any, **options: Any) -> float:
    notes: List[str] = []
    dice, pred_masks, true_masks, sensitive_features = _inputs.align_by_index(
        {"dice": dice, "pred_masks": options.get("pred_masks"), "true_masks": options.get("true_masks"),
         "sensitive_features": sensitive_features}, notes)
    if pred_masks is not None:
        options["pred_masks"] = pred_masks
    if true_masks is not None:
        options["true_masks"] = true_masks
    n = len(_inputs.to_numpy(dice, "dice")) if dice is not None else len(_masks(options))
    attribute = _one_attribute(sensitive_features, n, options.pop("bins", None))
    _keep_pairing(attribute, dice, pred_masks, true_masks)
    report = evaluate_segmentation(attribute, dice=dice, **options)
    report.warnings[:0] = notes
    return _value(report, column)


def _keep_pairing(attribute: pd.Series, *inputs: Any) -> None:
    """Give the attribute, already paired with the inputs, their index so that it is not paired again."""
    index = next((i for i in map(_inputs.pandas_index, inputs) if i is not None), None)
    if index is not None:
        attribute.index = index


def _masks(options):
    masks = options.get("pred_masks")
    if masks is None:
        raise ValueError("give dice, or both pred_masks and true_masks")
    return masks if isinstance(masks, (list, tuple, pd.Series)) else _inputs.to_numpy(masks, "pred_masks")


def _one_attribute(sensitive_features: Any, n: int, bins) -> pd.Series:
    table = _inputs.sensitive_table(sensitive_features, n, bins)
    return table.iloc[:, 0] if table.shape[1] == 1 else _inputs.intersect(table)


def _value(report, column: str) -> float:
    for message in report.warnings:
        if "skewness_dice" in message and column != "skewness_dice":
            continue  # only relevant to dice_skewness
        warnings.warn(message, UserWarning, stacklevel=4)  # point at the caller of the public function
    value = report.summary.iloc[0].get(column, math.nan)
    return float(value) if value is not None else math.nan


_CLS_ARGS = """
    Args:
        y_true: ground truth per sample (0/1, booleans, one-hot rows, or any labels with ``pos_label``).
        y_score: positive-class probabilities, logits, or a (samples, classes) matrix such as ``predict_proba``
            or softmax output.
        sensitive_features: group of each sample, e.g. sex or site: an array, Series, list, or a DataFrame or dict
            of several attributes (their combinations are compared).
        pos_label, score_type, score_column, bins: as in :func:`fairmedfm.evaluate`.
"""


def auc_gap(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None, score_type: str = "auto",
            score_column: Optional[int] = None, bins=None) -> float:
    """Largest minus smallest group ROC AUC (0 is fair)."""
    return _classification("auc-gap", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


def worst_group_auc(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None,
                    score_type: str = "auto", score_column: Optional[int] = None, bins=None) -> float:
    """Lowest group ROC AUC (higher is better)."""
    return _classification("worst-auc", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


def accuracy_gap(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None,
                 score_type: str = "auto", score_column: Optional[int] = None, bins=None) -> float:
    """Largest minus smallest group accuracy at the threshold with the best overall F1 (0 is fair)."""
    return _classification("acc-gap", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


def bce_gap(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None, score_type: str = "auto",
            score_column: Optional[int] = None, bins=None) -> float:
    """Largest minus smallest group binary cross-entropy (0 is fair)."""
    return _classification("bce-gap", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


def ece_gap(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None, score_type: str = "auto",
            score_column: Optional[int] = None, bins=None) -> float:
    """Largest minus smallest group expected calibration error (0 is fair)."""
    return _classification("ece-gap", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


def equal_opportunity_difference(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None,
                                 score_type: str = "auto", score_column: Optional[int] = None, bins=None) -> float:
    """Largest minus smallest group true positive rate at the best-F1 threshold (``eo``; 0 is fair)."""
    return _classification("eo", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


def equalized_odds_score(y_true: Any, y_score: Any, *, sensitive_features: Any, pos_label: Any = None,
                         score_type: str = "auto", score_column: Optional[int] = None, bins=None) -> float:
    """``1 - (TPR gap + TNR gap) / 2`` at the best-F1 threshold (``eod``; 1 is fair).

    A score, not a difference: unlike fairlearn's ``equalized_odds_difference`` (0 is fair), higher is fairer.
    """
    return _classification("eod", y_true, y_score, sensitive_features, pos_label=pos_label,
                           score_type=score_type, score_column=score_column, bins=bins)


_SEG_ARGS = """
    Args:
        dice: Dice score per sample. Alternatively pass ``pred_masks`` and ``true_masks`` (arrays, tensors or file
            paths) and Dice is computed per sample.
        sensitive_features: group of each sample, as in the classification functions.
        bins, pred_masks, true_masks, label, mask_threshold, empty_score: as in
            :func:`fairmedfm.evaluate_segmentation`.
"""


def dice_gap(dice: Any = None, *, sensitive_features: Any, bins=None, **masks: Any) -> float:
    """Largest minus smallest group mean Dice (``delta_dice``; 0 is fair)."""
    return _segmentation("delta_dice", dice, sensitive_features, bins=bins, **masks)


def worst_group_dice(dice: Any = None, *, sensitive_features: Any, bins=None, **masks: Any) -> float:
    """Lowest group mean Dice (``min_dice``; higher is better)."""
    return _segmentation("min_dice", dice, sensitive_features, bins=bins, **masks)


def dice_std(dice: Any = None, *, sensitive_features: Any, bins=None, **masks: Any) -> float:
    """Standard deviation of the group mean Dice scores (``std_dice``; 0 is fair)."""
    return _segmentation("std_dice", dice, sensitive_features, bins=bins, **masks)


def dice_skewness(dice: Any = None, *, sensitive_features: Any, bins=None, **masks: Any) -> float:
    """``(1 - worst group Dice) / (1 - best group Dice)`` (``skewness_dice``; 1 is fair, NaN if a group is perfect)."""
    return _segmentation("skewness_dice", dice, sensitive_features, bins=bins, **masks)


def equity_scaled_dice(dice: Any = None, *, sensitive_features: Any, bins=None, **masks: Any) -> float:
    """Mean Dice divided by ``1 +`` the standard deviation of group means (``es_dice``; higher is better)."""
    return _segmentation("es_dice", dice, sensitive_features, bins=bins, **masks)


def _add_arguments(functions: Iterable[Callable[..., float]], arguments: str) -> None:
    for function in functions:
        function.__doc__ = (function.__doc__ or "") + "\n" + arguments


_add_arguments([auc_gap, worst_group_auc, accuracy_gap, bce_gap, ece_gap, equal_opportunity_difference,
                equalized_odds_score], _CLS_ARGS)
_add_arguments([dice_gap, worst_group_dice, dice_std, dice_skewness, equity_scaled_dice], _SEG_ARGS)
