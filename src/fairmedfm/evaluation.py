"""High-level fairness evaluation that accepts data in whatever form users already have it."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from . import _inputs
from .metrics import _segmentation, _summary, _threshold_reports

__all__ = ["FairnessReport", "evaluate", "evaluate_segmentation"]


class FairnessReport:
    """Result of :func:`evaluate` or :func:`evaluate_segmentation`.

    Attributes:
        task: ``"cls"`` (classification) or ``"seg"`` (segmentation).
        summary: one row per sensitive attribute with the fairness metrics (see the metric reference).
        by_group: one row per (attribute, group) with the group's sample count and metrics. Groups that could not
            be evaluated have a ``skipped`` reason and are left out of the attribute's gaps.
        overall: metrics on all samples.
        warnings: notes about how the input was interpreted or which groups were skipped.
        metadata: how the input was read, e.g. the score transform or the positive label.
    """

    def __init__(self, task: str, summary: pd.DataFrame, by_group: pd.DataFrame, overall: Dict[str, Any],
                 attribute_overall: Dict[str, Dict[str, Any]], warnings: List[str], metadata: Dict[str, Any]):
        self.task = task
        self.summary = summary
        self.by_group = by_group
        self.overall = overall
        self.warnings = warnings
        self.metadata = metadata
        self._attribute_overall = attribute_overall

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible dictionary; NaN becomes ``None``."""
        from . import __version__
        attributes = {}
        for attribute, row in self.summary.iterrows():
            groups = self.by_group.loc[attribute]
            entry = {"n": int(row["n"]), "summary": _clean(row.drop(["n", "n_groups"]).to_dict()),
                     "groups": {str(g): _clean(values.to_dict()) for g, values in groups.iterrows()}}
            if attribute in self._attribute_overall:
                entry["overall"] = _clean(self._attribute_overall[attribute])
            attributes[str(attribute)] = entry
        return {"fairmedfm_version": __version__, "task": self.task, "metadata": _clean(self.metadata),
                "overall": _clean(self.overall), "attributes": attributes, "warnings": list(self.warnings)}

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """JSON text of :meth:`to_dict`; also written to ``path`` if given."""
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text + "\n")
        return text

    def __repr__(self) -> str:
        lines = [f"FairnessReport({self.task}, attributes: {', '.join(map(str, self.summary.index))})",
                 self.summary.to_string(float_format=lambda v: f"{v:.4f}")]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        notes = "".join(f"<p><b>warning:</b> {w}</p>" for w in self.warnings)
        return self.summary.to_html(float_format=lambda v: f"{v:.4f}") + notes


def evaluate(y_true: Any, y_score: Any, sensitive_features: Any, *, pos_label: Any = None,
             score_type: str = "auto", score_column: Optional[int] = None, bins: Optional[_inputs.Bins] = None,
             intersectional: bool = False) -> FairnessReport:
    """Fairness of a binary classifier across the groups of one or more sensitive attributes.

    Args:
        y_true: ground truth per sample: 0/1, booleans, one-hot rows, or any labels together with ``pos_label``.
        y_score: model output per sample: positive-class probabilities, logits, or a (samples, classes) array of
            probabilities or logits. Lists, NumPy arrays, pandas objects and PyTorch tensors all work.
        sensitive_features: a pandas DataFrame (one column per attribute), a dict of name to values, a Series, or
            one array. Values can be strings or numbers; missing values exclude the sample for that attribute.
        pos_label: the positive class in ``y_true``. Needed when labels are not 0/1 or booleans; with more than two
            classes, the evaluation is one-vs-rest for this class.
        score_type: ``"auto"`` treats scores in [0, 1] as probabilities and anything else as logits (sigmoid for one
            column, softmax for several); ``"probability"`` or ``"logit"`` force one interpretation.
        score_column: column of the positive class when ``y_score`` has several columns. By default, columns are
            taken to be in sorted class order, as in scikit-learn's ``predict_proba``: the column of ``pos_label``
            among the sorted labels, or the second of two columns for 0/1 labels.
        bins: how to group continuous attributes, e.g. ``{"age": [40, 60]}`` (cut points: <40, 40-60, >=60) or
            ``{"age": 4}`` (quartiles). Numeric attributes with more than 20 distinct values must be binned.
        intersectional: also evaluate the combination of all attributes (e.g. ``"F & >=60"``).

    Returns:
        A :class:`FairnessReport`. ``report.summary`` has, per attribute, ``overall-auc``, ``overall-acc``,
        ``overall-bce``, ``overall-ece``, ``worst-auc``, ``auc-gap``, ``acc-gap``, ``bce-gap``, ``ece-gap``, ``eo``
        (TPR gap) and ``eod`` (``1 - (TPR gap + TNR gap) / 2``, where 1 is fair), as in the FairMedFM paper.
    """
    return _evaluate(y_true, y_score, sensitive_features, pos_label=pos_label, score_type=score_type,
                     score_column=score_column, bins=bins, intersectional=intersectional)


def _evaluate(y_true: Any, y_score: Any, sensitive_features: Any, *, pos_label: Any = None, score_type: str = "auto",
              score_column: Optional[int] = None, bins: Optional[_inputs.Bins] = None, intersectional: bool = False,
              wanted: Optional[Set[str]] = None) -> FairnessReport:
    """:func:`evaluate`, computing only the per-group metrics in ``wanted`` (e.g. ``{"auc"}``) if given.

    With ``wanted``, ``summary`` has only the columns those metrics determine and ``overall`` is empty.
    """
    warnings: List[str] = []
    y_true, y_score, sensitive_features = _inputs.align_by_index(
        {"y_true": y_true, "y_score": y_score, "sensitive_features": sensitive_features}, warnings)
    labels, label_info = _inputs.binary_labels(y_true, pos_label)
    if score_column is None and pos_label is not None:
        score_column = _inputs.class_column(y_true, y_score, pos_label)
    scores, score_info = _inputs.positive_scores(y_score, score_type, score_column, pos_label)
    if len(labels) != len(scores):
        raise ValueError(f"y_true has {len(labels)} samples but y_score has {len(scores)}")
    table = _inputs.sensitive_table(sensitive_features, len(labels), bins)
    if intersectional and table.shape[1] > 1:
        combined = _inputs.intersect(table)
        table[combined.name] = combined
    if score_info.get("score_transform"):
        warnings.append(f"y_score was read as logits and converted with {score_info['score_transform']}; "
                        "pass score_type='probability' if these are probabilities")
    _require_both_labels(labels, "y_true")
    overall = _threshold_reports(scores, labels)[0] if wanted is None else {}

    summaries: Dict[str, Dict[str, Any]] = {}
    group_rows: List[Dict[str, Any]] = []
    attribute_overall: Dict[str, Dict[str, Any]] = {}
    for attribute in table.columns:
        order = _inputs.group_order(table[attribute])
        group = _inputs.as_groups(table[attribute])
        present = np.array([g is not None for g in group])
        if (~present).any():
            warnings.append(f"{attribute}: {int((~present).sum())} samples with a missing value were left out")
        p, y, g = scores[present], labels[present], group[present]
        valid, skipped = [], {}
        for value in order:
            classes = np.unique(y[g == value])
            if len(classes) < 2:
                skipped[value] = f"only label {int(classes[0])}"
            else:
                valid.append(value)
        for value, reason in skipped.items():
            warnings.append(f"{attribute}={value}: skipped ({reason}; AUC, TPR and TNR need both labels)")
        if len(np.unique(y)) < 2 or len(valid) < 2:
            if len(valid) < 2:
                warnings.append(f"{attribute}: fewer than two groups could be evaluated; its summary is empty")
            summaries[attribute] = {"n": len(y), "n_groups": len(valid)}
            reports: Dict[Any, Dict[str, Any]] = {}
            attribute_overall[attribute] = {}
        else:
            attribute_overall[attribute], reports = _threshold_reports(p, y, g, valid, wanted)
            subgroup = {key: [reports[v][key] for v in valid] for key in reports[valid[0]]}
            summaries[attribute] = {"n": len(y), "n_groups": len(valid), **_summary(attribute_overall[attribute],
                                                                                     subgroup)}
        for value in order:
            row: Dict[str, Any] = {"attribute": attribute, "group": value, "n": int((g == value).sum())}
            row.update(reports.get(value, {}))
            row["skipped"] = skipped.get(value)
            group_rows.append(row)
    metadata = {**label_info, **score_info, "n_samples": len(labels)}
    return FairnessReport("cls", _summary_frame(summaries), _group_frame(group_rows), overall,
                          attribute_overall, warnings, metadata)


def evaluate_segmentation(sensitive_features: Any, *, dice: Any = None, pred_masks: Any = None,
                          true_masks: Any = None, label: Any = None, mask_threshold: float = 0.5,
                          empty_score: float = 1.0, bins: Optional[_inputs.Bins] = None,
                          intersectional: bool = False) -> FairnessReport:
    """Fairness of a segmentation model across the groups of one or more sensitive attributes.

    Give either per-sample Dice scores (``dice``) or predicted and ground-truth masks (``pred_masks`` and
    ``true_masks``), from which Dice is computed per sample.

    Args:
        sensitive_features: as in :func:`evaluate`.
        dice: Dice score in [0, 1] per sample (image or volume).
        pred_masks: one predicted mask per sample: a list of arrays or tensors, a stacked array, or file paths
            (``.npy``, ``.npz``, ``.png``, ``.jpg``, ``.tif``, ``.nii``, ``.nii.gz``; images and NIfTI need
            ``pip install "fairmedfm[io]"``). Probability masks in [0, 1] are thresholded at ``mask_threshold``.
        true_masks: one ground-truth mask per sample, in the same forms.
        label: the class to evaluate in multi-class masks; by default any non-zero value is foreground.
        mask_threshold: threshold for probability masks.
        empty_score: Dice when both masks of a sample are empty (default 1.0, a correct empty prediction). The
            benchmark trainer's ``torchmetrics`` Dice gives 0 in this case; pass ``empty_score=0`` to match it.
        bins: as in :func:`evaluate`.
        intersectional: as in :func:`evaluate`.

    Returns:
        A :class:`FairnessReport`. ``report.summary`` has, per attribute, ``mean_dice``, ``min_dice``,
        ``max_dice``, ``delta_dice``, ``skewness_dice``, ``std_dice`` and ``es_dice``.
    """
    if dice is not None and (pred_masks is not None or true_masks is not None):
        raise ValueError("give either dice or pred_masks and true_masks, not both")
    warnings: List[str] = []
    dice, pred_masks, true_masks, sensitive_features = _inputs.align_by_index(
        {"dice": dice, "pred_masks": pred_masks, "true_masks": true_masks, "sensitive_features": sensitive_features},
        warnings)
    metadata: Dict[str, Any] = {}
    if dice is not None:
        scores = _inputs.to_numpy(dice, "dice").astype(np.float64).ravel()
    elif pred_masks is not None and true_masks is not None:
        scores = _inputs.dice_scores(pred_masks, true_masks, label=label, mask_threshold=mask_threshold,
                                     empty_score=empty_score)
        metadata.update({"dice_from_masks": True, "empty_score": empty_score, "label": label})
    else:
        raise ValueError("give dice, or both pred_masks and true_masks")
    finite = ~np.isnan(scores)
    if (~finite).any():
        warnings.append(f"{int((~finite).sum())} samples with a NaN Dice score were left out")
    if ((scores[finite] < 0) | (scores[finite] > 1)).any():
        raise ValueError("dice must contain values in [0, 1]")
    table = _inputs.sensitive_table(sensitive_features, len(scores), bins)
    if intersectional and table.shape[1] > 1:
        combined = _inputs.intersect(table)
        table[combined.name] = combined
    overall = {"mean_dice": float(scores[finite].mean()) if finite.any() else math.nan, "n": int(finite.sum())}

    summaries, group_rows = {}, []
    for attribute in table.columns:
        group = _inputs.as_groups(table[attribute])
        keep = finite & np.array([g is not None for g in group])
        missing = int((finite & ~keep).sum())
        if missing:
            warnings.append(f"{attribute}: {missing} samples with a missing value were left out")
        d, g = scores[keep], group[keep]
        values = [v for v in _inputs.group_order(table[attribute]) if (g == v).any()]
        if len(values) < 2:
            warnings.append(f"{attribute}: fewer than two groups; its summary is empty")
            summaries[attribute] = {"n": len(d), "n_groups": len(values)}
            means = {v: float(d[g == v].mean()) for v in values}
        else:
            result = _segmentation(d, g.astype(str))
            summary = result["summary"]
            if not math.isfinite(summary["skewness_dice"]):
                summary["skewness_dice"] = math.nan
                warnings.append(f"{attribute}: skewness_dice is undefined because the best group's mean Dice is 1")
            summaries[attribute] = {"n": len(d), "n_groups": len(values), **summary}
            means = {v: result["groups"][str(v)]["mean_dice"] for v in values}
        for value in values:
            group_rows.append({"attribute": attribute, "group": value, "n": int((g == value).sum()),
                               "mean_dice": means[value], "skipped": None})
    return FairnessReport("seg", _summary_frame(summaries), _group_frame(group_rows), overall, {},
                          warnings, metadata)


def _require_both_labels(labels: np.ndarray, name: str) -> None:
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{name} contains only one class; fairness metrics need positive and negative samples")


def _summary_frame(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(summaries, orient="index")
    frame.index.name = "attribute"
    columns = ["n", "n_groups"] + [c for c in frame.columns if c not in ("n", "n_groups")]
    return frame[columns]


def _group_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).set_index(["attribute", "group"])
    columns = ["n"] + [c for c in frame.columns if c not in ("n", "skipped")] + ["skipped"]
    return frame[columns]


def _clean(values: Any) -> Any:
    if isinstance(values, dict):
        return {str(k): _clean(v) for k, v in values.items()}
    if isinstance(values, (list, tuple)):
        return [_clean(v) for v in values]
    if isinstance(values, np.generic):
        values = values.item()
    if isinstance(values, float) and not math.isfinite(values):
        return None
    return values
