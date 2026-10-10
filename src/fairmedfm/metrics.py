"""Group fairness metrics for binary classification and segmentation.

The functions work on predictions from any model: export per-sample probabilities (classification) or
per-sample Dice scores (segmentation) together with a sensitive attribute such as sex, age group or race.
They reproduce the metrics reported in the FairMedFM paper, computed with NumPy, pandas and scikit-learn
only (no PyTorch or GPU).
"""
from __future__ import annotations

import warnings
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score

__all__ = [
    "classification_fairness",
    "segmentation_fairness",
    "binary_classification_report",
    "expected_calibration_error",
    "binary_cross_entropy",
    "find_threshold",
    "evaluate_binary",
    "organize_results",
    "evaluate_seg",
]


ALL_METRICS = ("auc", "acc", "bce", "ece", "tpr", "tnr", "tn", "fp", "fn", "tp")


def find_threshold(prob: ArrayLike, label: ArrayLike) -> float:
    """Decision threshold with the highest F1 score on these samples (the lowest such threshold on ties).

    Choosing the operating point by F1 follows the chest X-ray fairness evaluations of Seyyed-Kalantari et al.
    (Nature Medicine, 2021) and Zhang et al. (CHIL, 2022).
    """
    precision, recall, thresholds = precision_recall_curve(label, prob)
    # The curve ends with the point (recall 0, precision 1), which has no threshold.
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(thresholds[np.argmax(f1)])


def binary_cross_entropy(prob: ArrayLike, label: ArrayLike) -> float:
    """Mean binary cross-entropy, with log terms clamped at -100 as in ``torch.nn.BCELoss``."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=np.float64).ravel()
    with np.errstate(divide="ignore"):
        log_p = np.maximum(np.log(prob), -100)
        log_not_p = np.maximum(np.log(1 - prob), -100)
    return float(-np.mean(label * log_p + (1 - label) * log_not_p))


def expected_calibration_error(prob: ArrayLike, label: ArrayLike, num_bins: int = 10,
                               metric_variant: str = "abs", quantile_bins: bool = False) -> float:
    """Calibration error: the sample-weighted mean, over bins of ``prob``, of |mean prob - fraction positive|.

    Bins are ``num_bins`` equal-width intervals over the range of ``prob`` (``pandas.cut``), or equal-count bins
    with ``quantile_bins=True``. ``metric_variant="squared"`` averages squared differences and ``"rmse"`` takes
    the square root of that. See Guo et al. (2017), https://arxiv.org/abs/1706.04599, and Nixon et al. (2019),
    https://arxiv.org/abs/1904.01685.
    """
    if metric_variant not in ("abs", "squared", "rmse"):
        raise ValueError("metric_variant must be 'abs', 'squared' or 'rmse'")
    prob = np.asarray(prob, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=np.float64).ravel()
    bin_ids = (pd.qcut if quantile_bins else pd.cut)(prob, num_bins, labels=False)
    errors, sizes = [], []
    for bin_id in np.unique(bin_ids):
        in_bin = bin_ids == bin_id
        difference = prob[in_bin].mean() - label[in_bin].mean()
        errors.append(abs(difference) if metric_variant == "abs" else difference ** 2)
        sizes.append(in_bin.sum())
    error = float(np.average(errors, weights=sizes))
    return float(np.sqrt(error)) if metric_variant == "rmse" else error


def binary_classification_report(prob: ArrayLike, label: ArrayLike, threshold: float = 0.5,
                                 suffix: str = "") -> Dict[str, float]:
    """AUC, accuracy, BCE, ECE, TPR, TNR and confusion counts at one decision threshold."""
    return _report(np.asarray(prob), np.asarray(label), threshold, suffix, ALL_METRICS)


def _report(prob: np.ndarray, label: np.ndarray, threshold: float, suffix: str, names) -> Dict[str, float]:
    """The metrics in ``names`` (a subset of ``ALL_METRICS``), keyed with ``suffix``."""
    values: Dict[str, float] = {}
    if "auc" in names:
        values["auc"] = float(roc_auc_score(label, prob))
    if {"acc", "tpr", "tnr", "tn", "fp", "fn", "tp"} & set(names):
        tn, fp, fn, tp = confusion_matrix(label, (prob > threshold).astype(int), labels=[0, 1]).ravel()
        values.update(acc=float((tp + tn) / (tn + fp + fn + tp)), tpr=float(tp / (tp + fn)),
                      tnr=float(tn / (tn + fp)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    if "bce" in names:
        values["bce"] = binary_cross_entropy(prob, label)
    if "ece" in names:
        values["ece"] = expected_calibration_error(prob, label)
    return {f"{name}{suffix}": values[name] for name in ALL_METRICS if name in names}


def _threshold_reports(prob: np.ndarray, label: np.ndarray, group: Optional[np.ndarray] = None,
                      values: Optional[List[Any]] = None, wanted: Optional[Collection[str]] = None
                      ) -> Tuple[Dict[str, float], Dict[Any, Dict[str, float]]]:
    """Metrics on all samples and on each group, at threshold 0.5 and at the best overall F1 threshold.

    ``wanted`` limits the work to some keys, e.g. ``{"auc", "tpr@best_f1"}``; by default every metric is computed.
    ``values`` are the groups to report (default: all values of ``group``, sorted).
    """
    wanted = None if wanted is None else set(wanted)
    thresholds = []
    for suffix in ("", "@best_f1"):
        names = ALL_METRICS if wanted is None else [m for m in ALL_METRICS if f"{m}{suffix}" in wanted]
        if names:
            threshold = 0.5 if suffix == "" else find_threshold(prob, label)
            thresholds.append((threshold, suffix, names))
    overall: Dict[str, float] = {}
    for threshold, suffix, names in thresholds:
        overall.update(_report(prob, label, threshold, suffix, names))
    if group is None:
        return overall, {}
    if values is None:
        values = list(np.unique(group))
    reports: Dict[Any, Dict[str, float]] = {value: {} for value in values}
    for threshold, suffix, names in thresholds:
        for value in values:
            mask = group == value
            reports[value].update(_report(prob[mask], label[mask], threshold, suffix, names))
    return overall, reports


def _check_binary_inputs(prob: np.ndarray, label: np.ndarray, group: np.ndarray) -> None:
    if not (len(prob) == len(label) == len(group)):
        raise ValueError(f"prob, label and group must have the same length, got {len(prob)}, {len(label)} "
                         f"and {len(group)}")
    if prob.ndim != 1:
        raise ValueError("prob must be one probability per sample (the positive-class probability)")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("label must contain only 0 and 1; multi-class labels are not supported yet")
    if np.isnan(prob).any() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("prob must contain probabilities in [0, 1]")
    for value in np.unique(group):
        present = np.unique(label[group == value])
        if len(present) < 2:
            raise ValueError(f"group '{value}' contains only label {present[0]}; every group needs positive "
                             "and negative samples to compute AUC, TPR and TNR")
    if len(np.unique(group)) < 2:
        raise ValueError("the sensitive attribute needs at least two groups")


def _summary(overall: Mapping[str, float], subgroup: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    """The FairMedFM summary columns whose inputs are present in ``overall`` and ``subgroup``."""
    def gap(key: str) -> float:
        return float(max(subgroup[key]) - min(subgroup[key]))

    columns = {
        "overall-auc": (["auc"], [], lambda: float(overall["auc"])),
        "overall-acc": (["acc@best_f1"], [], lambda: float(overall["acc@best_f1"])),
        "overall-bce": (["bce"], [], lambda: float(overall["bce"])),
        "overall-ece": (["ece"], [], lambda: float(overall["ece"])),
        "worst-auc": ([], ["auc"], lambda: float(min(subgroup["auc"]))),
        "auc-gap": ([], ["auc"], lambda: gap("auc")),
        "acc-gap": ([], ["acc@best_f1"], lambda: gap("acc@best_f1")),
        "bce-gap": ([], ["bce"], lambda: gap("bce")),
        "ece-gap": ([], ["ece"], lambda: gap("ece")),
        "eod": ([], ["tpr@best_f1", "tnr@best_f1"],
                lambda: float(1 - (gap("tpr@best_f1") + gap("tnr@best_f1")) / 2)),
        "eo": ([], ["tpr@best_f1"], lambda: gap("tpr@best_f1")),
    }
    return {name: value() for name, (in_overall, in_groups, value) in columns.items()
            if all(k in overall for k in in_overall) and all(k in subgroup for k in in_groups)}


def classification_fairness(prob: ArrayLike, label: ArrayLike, group: Sequence[Any]) -> Dict[str, Any]:
    """Fairness of a binary classifier across the groups of one sensitive attribute.

    Args:
        prob: positive-class probability for each sample.
        label: ground-truth label (0 or 1) for each sample.
        group: sensitive attribute value for each sample, e.g. ``"F"``/``"M"`` or age bins. Two or more
            groups; every group must contain both labels.

    Deprecated since 0.5 and to be removed in 1.0: use :func:`fairmedfm.evaluate`, which accepts more input forms
    and reports groups it cannot evaluate instead of failing, or a single metric such as :func:`fairmedfm.auc_gap`.

    Returns:
        ``summary``: the FairMedFM fairness metrics (``overall-auc``, ``worst-auc``, ``auc-gap``, ``acc-gap``,
        ``bce-gap``, ``ece-gap``, ``eod``, ``eo`` and the overall accuracy, BCE and ECE). Gaps are the largest
        minus the smallest group value. ``eod`` is ``1 - (TPR gap + TNR gap) / 2`` (higher is fairer) and
        ``eo`` is the TPR gap; both use the threshold with the best overall F1, as does accuracy.
        ``overall``: metrics on all samples at threshold 0.5 and at the best-F1 threshold (``@best_f1``).
        ``groups``: the same metrics for each group, keyed by group value as a string, with sample counts.
    """
    _deprecated("classification_fairness", "fairmedfm.evaluate(label, prob, group)")
    prob_array = np.asarray(prob, dtype=np.float64)
    label_array = np.asarray(label)
    group_array = np.asarray(group)
    _check_binary_inputs(prob_array, label_array, group_array)
    overall, by_value = _threshold_reports(prob_array, label_array, group_array)
    groups, reports = list(by_value), list(by_value.values())
    subgroup = {key: [report[key] for report in reports] for key in reports[0]}
    return {
        "summary": _summary(overall, subgroup),
        "overall": overall,
        "groups": {str(value): {"n": int((group_array == value).sum()), **report}
                   for value, report in zip(groups, reports)},
    }


def segmentation_fairness(dice: ArrayLike, group: Sequence[Any]) -> Dict[str, Any]:
    """Fairness of a segmentation model from per-sample Dice scores across the groups of one attribute.

    Args:
        dice: Dice similarity coefficient in [0, 1] for each sample (image or volume).
        group: sensitive attribute value for each sample; two or more groups.

    Deprecated since 0.5 and to be removed in 1.0: use :func:`fairmedfm.evaluate_segmentation` or a single metric
    such as :func:`fairmedfm.dice_gap`.

    Returns:
        ``summary``: ``mean_dice`` (over all samples), ``min_dice``/``max_dice`` (worst and best group mean),
        ``delta_dice`` (max minus min), ``skewness_dice`` (``(1 - min) / (1 - max)``; ``None`` when the best
        group has a mean Dice of exactly 1), ``std_dice`` (standard deviation of the group means) and
        ``es_dice`` (equity-scaled Dice, ``mean_dice / (1 + std_dice)``).
        ``groups``: mean Dice and sample count per group, keyed by group value as a string.
    """
    _deprecated("segmentation_fairness", "fairmedfm.evaluate_segmentation(group, dice=dice)")
    dice_array = np.asarray(dice, dtype=np.float64).ravel()
    if np.isnan(dice_array).any() or (dice_array < 0).any() or (dice_array > 1).any():
        raise ValueError("dice must contain values in [0, 1]")
    result = _segmentation(dice_array, group)
    if not np.isfinite(result["summary"]["skewness_dice"]):
        result["summary"]["skewness_dice"] = None
    return result


def _deprecated(name: str, replacement: str) -> None:
    warnings.warn(f"fairmedfm.{name} is deprecated and will be removed in fairmedfm 1.0; use {replacement}",
                  FutureWarning, stacklevel=3)


def _segmentation(dice: np.ndarray, group: Sequence[Any]) -> Dict[str, Any]:
    group_array = np.asarray(group).ravel()
    if len(dice) != len(group_array):
        raise ValueError(f"dice and group must have the same length, got {len(dice)} and {len(group_array)}")
    values = list(np.unique(group_array))
    if len(values) < 2:
        raise ValueError("the sensitive attribute needs at least two groups")
    means = np.array([dice[group_array == value].mean() for value in values])
    min_dice, max_dice = float(means.min()), float(means.max())
    std_dice = float(means.std())
    mean_dice = float(dice.mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        skewness = float(np.divide(1 - min_dice, 1 - max_dice))
    return {
        "summary": {
            "mean_dice": mean_dice,
            "min_dice": min_dice,
            "max_dice": max_dice,
            "delta_dice": max_dice - min_dice,
            "skewness_dice": skewness,
            "std_dice": std_dice,
            "es_dice": mean_dice / (1 + std_dice),
        },
        "groups": {str(value): {"n": int((group_array == value).sum()), "mean_dice": float(mean)}
                   for value, mean in zip(values, means)},
    }


# Interfaces used by the FairMedFM training code; kept for existing callers.

def evaluate_binary(pred: ArrayLike, Y: ArrayLike, A: Sequence[Any]
                    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Overall metrics and per-group metric lists (groups in sorted order)."""
    pred_array, label_array, group_array = np.asarray(pred, dtype=np.float64), np.asarray(Y), np.asarray(A)
    _check_binary_inputs(pred_array, label_array, group_array)
    overall, by_value = _threshold_reports(pred_array, label_array, group_array)
    reports = list(by_value.values())
    return overall, {key: [report[key] for report in reports] for key in reports[0]}


def organize_results(overall_metrics: Mapping[str, float], subgroup_metrics: Mapping[str, Sequence[float]]
                     ) -> Dict[str, float]:
    """Fairness summary from the output of :func:`evaluate_binary`."""
    return _summary(overall_metrics, subgroup_metrics)


def evaluate_seg(dsc_list: ArrayLike, sensitive_list: Sequence[Any]) -> Dict[str, Any]:
    """Segmentation fairness summary as logged by the trainers; see :func:`segmentation_fairness`.

    Unlike :func:`segmentation_fairness`, ``skewness_dice`` is ``inf`` when the best group has a mean Dice of 1,
    and NaN Dice scores propagate into the means instead of raising.
    """
    return _segmentation(np.asarray(dsc_list, dtype=np.float64).ravel(), sensitive_list)["summary"]
