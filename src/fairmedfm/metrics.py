"""Group fairness metrics for binary classification and segmentation.

The functions work on predictions from any model: export per-sample probabilities (classification) or
per-sample Dice scores (segmentation) together with a sensitive attribute such as sex, age group or race.
They reproduce the metrics reported in the FairMedFM paper, computed with NumPy, pandas and scikit-learn
only (no PyTorch or GPU).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
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


# Adapted from https://github.com/LalehSeyyed/Underdiagnosis_NatMed/blob/main/CXP/classification/predictions.py
# and https://github.com/MLforHealth/CXR_Fairness/blob/master/cxr_fairness/metrics.py
def find_threshold(prob: Sequence[float], label: Sequence[int]) -> float:
    """Return the decision threshold with the highest F1 score (the first one on ties)."""
    precision, recall, thresholds = precision_recall_curve(label, prob)
    f1 = np.multiply(2, np.divide(np.multiply(precision, recall), np.add(recall, precision) + 1e-8))
    # The last precision/recall point (recall 0) has no threshold.
    best = np.where(f1[:-1] == f1[:-1].max())[0]
    return float(thresholds[best[0]])


def binary_cross_entropy(prob: Sequence[float], label: Sequence[int]) -> float:
    """Mean binary cross-entropy, with log terms clamped at -100 as in ``torch.nn.BCELoss``."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    label = np.asarray(label, dtype=np.float64).ravel()
    with np.errstate(divide="ignore"):
        log_p = np.maximum(np.log(prob), -100)
        log_not_p = np.maximum(np.log(1 - prob), -100)
    return float(-np.mean(label * log_p + (1 - label) * log_not_p))


def expected_calibration_error(prob: Sequence[float], label: Sequence[int], num_bins: int = 10,
                               metric_variant: str = "abs", quantile_bins: bool = False) -> float:
    """Calibration error over ``num_bins`` equal-width bins spanning the range of ``prob``.

    See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685. Adapted from
    https://github.com/MLforHealth/CXR_Fairness/blob/c2a0e884171d6418e28d59dca1ccfb80a3f125fe/cxr_fairness/metrics.py#L1557
    """
    if metric_variant == "abs":
        transform = np.abs
    elif metric_variant in ("squared", "rmse"):
        transform = np.square
    else:
        raise ValueError("metric_variant must be 'abs', 'squared' or 'rmse'")
    prob = np.asarray(prob)
    cut = pd.qcut if quantile_bins else pd.cut
    bin_ids = cut(prob, num_bins, labels=False, retbins=False)
    df = pd.DataFrame({"prob": prob, "label": np.asarray(label), "bin_id": bin_ids})
    bins = df.groupby("bin_id").agg(prob_mean=("prob", "mean"), label_mean=("label", "mean"),
                                    bin_size=("prob", "size"))
    weight = bins.bin_size / df.shape[0]
    result = np.average(transform(bins.prob_mean - bins.label_mean).values, weights=weight)
    if metric_variant == "rmse":
        result = np.sqrt(result)
    return float(result)


def binary_classification_report(prob: Sequence[float], label: Sequence[int], threshold: float = 0.5,
                                 suffix: str = "") -> Dict[str, float]:
    """AUC, accuracy, BCE, ECE, TPR, TNR and confusion counts at one decision threshold."""
    prob = np.asarray(prob)
    label = np.asarray(label)
    tn, fp, fn, tp = confusion_matrix(label, (prob > threshold).astype(int), labels=[0, 1]).ravel()
    return {
        f"auc{suffix}": float(roc_auc_score(label, prob)),
        f"acc{suffix}": float((tp + tn) / (tn + fp + fn + tp)),
        f"bce{suffix}": binary_cross_entropy(prob, label),
        f"ece{suffix}": expected_calibration_error(prob, label),
        f"tpr{suffix}": float(tp / (tp + fn)),
        f"tnr{suffix}": float(tn / (tn + fp)),
        f"tn{suffix}": int(tn),
        f"fp{suffix}": int(fp),
        f"fn{suffix}": int(fn),
        f"tp{suffix}": int(tp),
    }


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


def _per_group_reports(prob: np.ndarray, label: np.ndarray, group: np.ndarray
                       ) -> Tuple[Dict[str, float], List[Any], List[Dict[str, float]]]:
    thresholds = [(0.5, ""), (find_threshold(prob, label), "@best_f1")]
    overall: Dict[str, float] = {}
    groups = list(np.unique(group))
    reports: List[Dict[str, float]] = [{} for _ in groups]
    for threshold, suffix in thresholds:
        overall.update(binary_classification_report(prob, label, threshold, suffix))
        for report, value in zip(reports, groups):
            mask = group == value
            report.update(binary_classification_report(prob[mask], label[mask], threshold, suffix))
    return overall, groups, reports


def _summary(overall: Mapping[str, float], subgroup: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    def gap(key: str) -> float:
        return float(max(subgroup[key]) - min(subgroup[key]))

    tpr, tnr = subgroup["tpr@best_f1"], subgroup["tnr@best_f1"]
    return {
        "overall-auc": float(overall["auc"]),
        "overall-acc": float(overall["acc@best_f1"]),
        "overall-bce": float(overall["bce"]),
        "overall-ece": float(overall["ece"]),
        "worst-auc": float(min(subgroup["auc"])),
        "auc-gap": gap("auc"),
        "acc-gap": gap("acc@best_f1"),
        "bce-gap": gap("bce"),
        "ece-gap": gap("ece"),
        "eod": float(1 - ((max(tpr) - min(tpr)) + (max(tnr) - min(tnr))) / 2),
        "eo": gap("tpr@best_f1"),
    }


def classification_fairness(prob: Sequence[float], label: Sequence[int], group: Sequence[Any]) -> Dict[str, Any]:
    """Fairness of a binary classifier across the groups of one sensitive attribute.

    Args:
        prob: positive-class probability for each sample.
        label: ground-truth label (0 or 1) for each sample.
        group: sensitive attribute value for each sample, e.g. ``"F"``/``"M"`` or age bins. Two or more
            groups; every group must contain both labels.

    Returns:
        ``summary``: the FairMedFM fairness metrics (``overall-auc``, ``worst-auc``, ``auc-gap``, ``acc-gap``,
        ``bce-gap``, ``ece-gap``, ``eod``, ``eo`` and the overall accuracy, BCE and ECE). Gaps are the largest
        minus the smallest group value. ``eod`` is ``1 - (TPR gap + TNR gap) / 2`` (higher is fairer) and
        ``eo`` is the TPR gap; both use the threshold with the best overall F1, as does accuracy.
        ``overall``: metrics on all samples at threshold 0.5 and at the best-F1 threshold (``@best_f1``).
        ``groups``: the same metrics for each group, keyed by group value as a string, with sample counts.
    """
    prob_array = np.asarray(prob, dtype=np.float64)
    label_array = np.asarray(label)
    group_array = np.asarray(group)
    _check_binary_inputs(prob_array, label_array, group_array)
    overall, groups, reports = _per_group_reports(prob_array, label_array, group_array)
    subgroup = {key: [report[key] for report in reports] for key in reports[0]}
    return {
        "summary": _summary(overall, subgroup),
        "overall": overall,
        "groups": {str(value): {"n": int((group_array == value).sum()), **report}
                   for value, report in zip(groups, reports)},
    }


def segmentation_fairness(dice: Sequence[float], group: Sequence[Any]) -> Dict[str, Any]:
    """Fairness of a segmentation model from per-sample Dice scores across the groups of one attribute.

    Args:
        dice: Dice similarity coefficient in [0, 1] for each sample (image or volume).
        group: sensitive attribute value for each sample; two or more groups.

    Returns:
        ``summary``: ``mean_dice`` (over all samples), ``min_dice``/``max_dice`` (worst and best group mean),
        ``delta_dice`` (max minus min), ``skewness_dice`` (``(1 - min) / (1 - max)``; ``None`` when the best
        group has a mean Dice of exactly 1), ``std_dice`` (standard deviation of the group means) and
        ``es_dice`` (equity-scaled Dice, ``mean_dice / (1 + std_dice)``).
        ``groups``: mean Dice and sample count per group, keyed by group value as a string.
    """
    dice_array = np.asarray(dice, dtype=np.float64).ravel()
    group_array = np.asarray(group).ravel()
    if len(dice_array) != len(group_array):
        raise ValueError(f"dice and group must have the same length, got {len(dice_array)} and {len(group_array)}")
    if np.isnan(dice_array).any() or (dice_array < 0).any() or (dice_array > 1).any():
        raise ValueError("dice must contain values in [0, 1]")
    values = list(np.unique(group_array))
    if len(values) < 2:
        raise ValueError("the sensitive attribute needs at least two groups")
    means = np.array([dice_array[group_array == value].mean() for value in values])
    min_dice, max_dice = float(means.min()), float(means.max())
    std_dice = float(means.std())
    mean_dice = float(dice_array.mean())
    return {
        "summary": {
            "mean_dice": mean_dice,
            "min_dice": min_dice,
            "max_dice": max_dice,
            "delta_dice": max_dice - min_dice,
            "skewness_dice": (1 - min_dice) / (1 - max_dice) if max_dice < 1 else None,
            "std_dice": std_dice,
            "es_dice": mean_dice / (1 + std_dice),
        },
        "groups": {str(value): {"n": int((group_array == value).sum()), "mean_dice": float(mean)}
                   for value, mean in zip(values, means)},
    }


# Interfaces used by the FairMedFM training code; kept for existing callers.

def evaluate_binary(pred: Sequence[float], Y: Sequence[int], A: Sequence[Any]
                    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Overall metrics and per-group metric lists (groups in sorted order)."""
    pred_array, label_array, group_array = np.asarray(pred, dtype=np.float64), np.asarray(Y), np.asarray(A)
    _check_binary_inputs(pred_array, label_array, group_array)
    overall, _, reports = _per_group_reports(pred_array, label_array, group_array)
    return overall, {key: [report[key] for report in reports] for key in reports[0]}


def organize_results(overall_metrics: Mapping[str, float], subgroup_metrics: Mapping[str, Sequence[float]]
                     ) -> Dict[str, float]:
    """Fairness summary from the output of :func:`evaluate_binary`."""
    return _summary(overall_metrics, subgroup_metrics)


def evaluate_seg(dsc_list: Sequence[float], sensitive_list: Sequence[Any]) -> Dict[str, Any]:
    """Segmentation fairness summary; see :func:`segmentation_fairness`."""
    return segmentation_fairness(dsc_list, sensitive_list)["summary"]
