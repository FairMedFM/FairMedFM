"""Turn the many shapes users have their data in into the arrays the metrics need."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

Bins = Mapping[str, Union[int, Sequence[float]]]
# A numeric attribute with more distinct values than this is treated as continuous and must be binned.
MAX_NUMERIC_GROUPS = 20


def to_numpy(values: Any, name: str) -> np.ndarray:
    """NumPy array from a list, NumPy array, pandas object, PyTorch/TensorFlow/JAX tensor or similar."""
    if values is None:
        raise ValueError(f"{name} is required")
    if hasattr(values, "detach") and hasattr(values, "cpu"):  # PyTorch
        values = values.detach().cpu().numpy()
    elif isinstance(values, (pd.Series, pd.DataFrame, pd.Index)):
        values = values.to_numpy()
    elif hasattr(values, "to_numpy") and not isinstance(values, np.ndarray):  # polars and others
        values = values.to_numpy()
    elif hasattr(values, "numpy") and not isinstance(values, np.ndarray):  # TensorFlow
        values = values.numpy()
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError(f"{name} must have one value per sample, got a scalar")
    return array


def align_by_index(named: Dict[str, Any], notes: List[str]) -> Tuple[Any, ...]:
    """The values of ``named``, with pandas inputs reordered to the first one's index where that is clearly meant.

    Inputs are paired by position, as in scikit-learn. The exception: when two pandas inputs have the same unique
    index labels in a different order (e.g. predictions in one order and a metadata table in another), they are
    paired by index, as pandas would, and a note says so. Indexes with different labels stay positional.
    """
    values = dict(named)
    reference_name = next((name for name, value in values.items() if pandas_index(value) is not None), None)
    if reference_name is None:
        return tuple(values.values())
    reference = pandas_index(values[reference_name])
    if reference is None or not reference.is_unique:
        return tuple(values.values())
    for name, value in values.items():
        if name == reference_name:
            continue
        if isinstance(value, Mapping):
            reordered: Any = {key: _reorder(item, reference) for key, item in value.items()}
            changed = any(reordered[key] is not item for key, item in value.items())
        else:
            reordered = _reorder(value, reference)
            changed = reordered is not value
        if changed:
            notes.append(f"{name} was reordered to match the index of {reference_name} (same index labels in a "
                         "different order); samples are paired by index")
            values[name] = reordered
    return tuple(values.values())


def pandas_index(value: Any) -> Optional[pd.Index]:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.index
    if isinstance(value, Mapping):
        return next((item.index for item in value.values() if isinstance(item, (pd.Series, pd.DataFrame))), None)
    return None


def _reorder(value: Any, reference: pd.Index) -> Any:
    if not isinstance(value, (pd.Series, pd.DataFrame)):
        return value
    index = value.index
    if index.equals(reference) or len(index) != len(reference) or not index.is_unique or not index.isin(reference).all():
        return value
    return value.loc[reference]


def binary_labels(y_true: Any, pos_label: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """0/1 labels from booleans, 0/1, any two labels plus pos_label, or several classes plus pos_label."""
    labels = to_numpy(y_true, "y_true")
    if labels.ndim == 2 and labels.shape[1] > 1:  # one-hot
        labels = labels.argmax(axis=1)
    labels = labels.reshape(len(labels), -1)
    if labels.shape[1] != 1:
        raise ValueError(f"y_true must have one label per sample, got shape {to_numpy(y_true, 'y_true').shape}")
    labels = labels[:, 0]
    info: Dict[str, Any] = {}
    if pos_label is not None:
        if not np.any(labels == pos_label):
            raise ValueError(f"pos_label {pos_label!r} does not occur in y_true; values: {_preview(np.unique(labels))}")
        info["pos_label"] = _plain(pos_label)
        if len(np.unique(labels)) > 2:
            info["one_vs_rest"] = True
        return (labels == pos_label).astype(int), info
    if labels.dtype == bool:
        return labels.astype(int), info
    values = np.unique(labels)
    if np.isin(values, [0, 1]).all() and values.dtype.kind in "biuf":
        return labels.astype(int), info
    raise ValueError(f"y_true has values {_preview(values)}; pass pos_label to say which value is the positive class "
                     "(other values are treated as negative)")


def positive_scores(y_score: Any, score_type: str = "auto", score_column: Optional[int] = None,
                    pos_label: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Positive-class probabilities from probabilities, logits, or a (samples, classes) array of either."""
    if score_type not in ("auto", "probability", "logit"):
        raise ValueError("score_type must be 'auto', 'probability' or 'logit'")
    scores = to_numpy(y_score, "y_score").astype(np.float64)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if np.isnan(scores).any():
        raise ValueError("y_score contains NaN")
    info: Dict[str, Any] = {}
    if scores.ndim == 1:
        is_probability = score_type == "probability" or (score_type == "auto" and _in_unit_interval(scores))
        if score_type == "probability" and not _in_unit_interval(scores):
            raise ValueError("y_score has values outside [0, 1] but score_type='probability'")
        if not is_probability:
            scores = 1 / (1 + np.exp(-scores))
            info["score_transform"] = "sigmoid"
        return scores, info
    if scores.ndim != 2:
        raise ValueError(f"y_score must be one score per sample or a (samples, classes) array, got shape {scores.shape}")
    n_classes = scores.shape[1]
    if score_column is not None:
        column = int(score_column)
    elif n_classes == 2:
        column = 1
    elif isinstance(pos_label, (int, np.integer)) and not isinstance(pos_label, bool):
        column = int(pos_label)
    else:
        raise ValueError(f"y_score has {n_classes} columns; pass score_column, the column of the positive class")
    if not 0 <= column < n_classes:
        raise ValueError(f"score_column {column} is out of range for {n_classes} columns")
    rows_are_probabilities = _in_unit_interval(scores) and np.allclose(scores.sum(axis=1), 1, atol=1e-3)
    if score_type == "logit" or (score_type == "auto" and not rows_are_probabilities):
        shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
        scores = shifted / shifted.sum(axis=1, keepdims=True)
        info["score_transform"] = "softmax"
    info["score_column"] = column
    return scores[:, column], info


def class_column(y_true: Any, y_score: Any, pos_label: Any) -> Optional[int]:
    """Column of pos_label in a (samples, classes) score matrix, assuming sorted class order."""
    shape = getattr(y_score, "shape", None) or np.shape(y_score)
    if len(shape) != 2 or shape[1] < 2:
        return None
    labels = to_numpy(y_true, "y_true")
    if labels.ndim != 1:
        return None
    classes = np.unique(labels)
    if len(classes) != shape[1] or pos_label not in set(classes.tolist()):
        return None
    return int(np.flatnonzero(classes == pos_label)[0])


def sensitive_table(sensitive_features: Any, n_samples: int, bins: Optional[Bins] = None) -> pd.DataFrame:
    """One column per sensitive attribute, as strings, with missing values kept as None."""
    if isinstance(sensitive_features, pd.DataFrame):
        table = sensitive_features.reset_index(drop=True).copy()
    elif isinstance(sensitive_features, pd.Series):
        table = sensitive_features.reset_index(drop=True).to_frame(sensitive_features.name or "sensitive")
    elif isinstance(sensitive_features, Mapping):
        table = pd.DataFrame({str(k): to_numpy(v, f"sensitive_features[{k!r}]").ravel()
                              for k, v in sensitive_features.items()})
    else:
        array = to_numpy(sensitive_features, "sensitive_features")
        if array.ndim == 1:
            table = pd.DataFrame({"sensitive": array})
        elif array.ndim == 2:
            table = pd.DataFrame(array, columns=[f"sensitive_{i}" for i in range(array.shape[1])])
        else:
            raise ValueError(f"sensitive_features must be 1D or 2D, got shape {array.shape}")
    table.columns = [str(c) for c in table.columns]
    if table.shape[1] == 0:
        raise ValueError("sensitive_features has no attributes")
    if len(table) != n_samples:
        raise ValueError(f"sensitive_features has {len(table)} rows but there are {n_samples} samples")
    bins = dict(bins or {})
    unknown = sorted(set(bins) - set(table.columns))
    if unknown:
        raise ValueError(f"bins refers to unknown attribute(s) {unknown}; attributes: {list(table.columns)}")
    out = {}
    for column in table.columns:
        values = table[column]
        if column in bins:
            out[column] = _bin(values, bins[column], column)
            continue
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
        if numeric and values.nunique(dropna=True) > MAX_NUMERIC_GROUPS:
            raise ValueError(f"attribute {column!r} looks continuous ({values.nunique()} distinct values); group it "
                             f"with bins={{{column!r}: [40, 60]}} (cut points) or bins={{{column!r}: 4}} (quantiles)")
        out[column] = values.map(_group_name)
    return pd.DataFrame(out)


def group_order(values: pd.Series) -> List[str]:
    """Groups of one attribute in a natural order: bin order, numeric order, then alphabetical."""
    present = values.dropna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [str(c) for c in values.cat.categories if (present == c).any()]

    def key(value: str):
        try:
            return (0, float(value), "")
        except ValueError:
            return (1, 0.0, value)
    return sorted({str(v) for v in present}, key=key)


def as_groups(values: pd.Series) -> np.ndarray:
    """Group names as an object array, with None for missing values."""
    return np.array([None if pd.isna(v) else str(v) for v in values], dtype=object)


def intersect(table: pd.DataFrame) -> pd.Series:
    """Combined attribute, e.g. 'F & 60+', missing when any part is missing."""
    combined = table.astype(object).apply(lambda row: None if row.isna().any() else " & ".join(row.astype(str)),
                                          axis=1)
    combined.name = " & ".join(table.columns)
    return combined


def dice_scores(pred_masks: Any, true_masks: Any, *, label: Any = None, mask_threshold: float = 0.5,
                empty_score: float = 1.0) -> np.ndarray:
    """Per-sample Dice between predicted and ground-truth masks (arrays, tensors or file paths)."""
    preds, truths = _mask_list(pred_masks, "pred_masks"), _mask_list(true_masks, "true_masks")
    if len(preds) != len(truths):
        raise ValueError(f"pred_masks has {len(preds)} samples but true_masks has {len(truths)}")
    scores = np.empty(len(preds))
    for i, (pred, truth) in enumerate(zip(preds, truths)):
        p = _foreground(load_mask(pred), label, mask_threshold, binary_probabilities=True)
        t = _foreground(load_mask(truth), label, mask_threshold, binary_probabilities=False)
        if p.shape != t.shape:
            raise ValueError(f"sample {i}: predicted mask has shape {p.shape} but ground truth has shape {t.shape}")
        total = p.sum() + t.sum()
        scores[i] = empty_score if total == 0 else 2 * np.logical_and(p, t).sum() / total
    return scores


def load_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        return read_array_file(Path(mask))
    return np.squeeze(to_numpy(mask, "mask"))


def read_array_file(path: Path) -> np.ndarray:
    name = path.name.lower()
    if not path.exists():
        raise FileNotFoundError(f"mask file not found: {path}")
    if name.endswith(".npy"):
        return np.squeeze(np.load(path))
    if name.endswith(".npz"):
        with np.load(path) as data:
            return np.squeeze(data[data.files[0]])
    if name.endswith((".nii", ".nii.gz")):
        nibabel = _optional("nibabel", "NIfTI masks")
        return np.squeeze(np.asanyarray(nibabel.load(str(path)).dataobj))
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")):
        image = _optional("PIL.Image", "image masks")
        with image.open(path) as img:
            array = np.asarray(img)
        return array if array.ndim == 2 else array[..., 0]  # grayscale from the first channel
    raise ValueError(f"unsupported mask file {path}; use .npy, .npz, .png, .jpg, .tif or .nii/.nii.gz")


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """A table from CSV, TSV, Parquet, Feather, JSON, JSON Lines or Excel, chosen by file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path)
    if name.endswith((".tsv", ".tsv.gz", ".tab", ".txt")):
        return pd.read_csv(path, sep="\t")
    if name.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True)
    if name.endswith(".json"):
        return pd.read_json(path)
    if name.endswith((".parquet", ".pq")):
        _optional("pyarrow", "Parquet files")
        return pd.read_parquet(path)
    if name.endswith((".feather", ".arrow")):
        _optional("pyarrow", "Feather files")
        return pd.read_feather(path)
    if name.endswith((".xlsx", ".xls")):
        _optional("openpyxl", "Excel files")
        return pd.read_excel(path)
    raise ValueError(f"unsupported table {path}; use .csv, .tsv, .parquet, .feather, .json, .jsonl or .xlsx")


def _mask_list(masks: Any, name: str) -> List[Any]:
    if masks is None:
        raise ValueError(f"{name} is required")
    if isinstance(masks, (str, Path)):
        raise ValueError(f"{name} must be one mask per sample (a list, an array or a column of file paths)")
    if isinstance(masks, pd.Series):
        return list(masks)
    if isinstance(masks, (list, tuple)):
        return list(masks)
    array = to_numpy(masks, name)
    return [array[i] for i in range(len(array))]


def _foreground(mask: np.ndarray, label: Any, threshold: float, binary_probabilities: bool) -> np.ndarray:
    if label is not None:
        return mask == label
    if mask.dtype == bool:
        return mask
    if binary_probabilities and mask.dtype.kind == "f" and mask.size and mask.min() >= 0 and mask.max() <= 1:
        return mask >= threshold
    return mask > 0


def _bin(values: pd.Series, spec: Union[int, Sequence[float]], column: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() < values.notna().sum():
        raise ValueError(f"attribute {column!r} has non-numeric values and cannot be binned")
    if isinstance(spec, (int, np.integer)):
        if spec < 2:
            raise ValueError(f"bins for {column!r} must be at least 2 quantile groups")
        cut_points: Any = np.unique(np.quantile(numeric.dropna(), np.linspace(0, 1, int(spec) + 1)[1:-1]))
    else:
        cut_points = spec
    edges = sorted(float(e) for e in cut_points)
    if not edges:
        raise ValueError(f"bins for {column!r} must contain at least one cut point")
    labels = [f"<{edges[0]:.4g}"] + [f"{a:.4g}-{b:.4g}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]:.4g}"]
    return pd.cut(numeric, [-np.inf] + edges + [np.inf], right=False, labels=labels)


def _group_name(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _in_unit_interval(values: np.ndarray) -> bool:
    return bool(values.size) and values.min() >= 0 and values.max() <= 1


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _preview(values: np.ndarray, limit: int = 8) -> str:
    shown = ", ".join(repr(_plain(v)) for v in values[:limit])
    return f"[{shown}{', ...' if len(values) > limit else ''}]"


def _optional(module: str, purpose: str):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        raise ImportError(f"reading {purpose} needs an optional dependency: pip install \"fairmedfm[io]\"") from None


def warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)
