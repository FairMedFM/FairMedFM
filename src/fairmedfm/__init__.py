"""FairMedFM: group fairness evaluation for binary classification and segmentation models."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fairmedfm")
except PackageNotFoundError:  # source checkout that has not been installed
    __version__ = "0+unknown"

_GROUP_METRICS = (
    "auc_gap", "worst_group_auc", "accuracy_gap", "bce_gap", "ece_gap", "equal_opportunity_difference",
    "equalized_odds_score", "dice_gap", "worst_group_dice", "dice_std", "dice_skewness", "equity_scaled_dice",
)
__all__ = ["evaluate", "evaluate_segmentation", "FairnessReport", *_GROUP_METRICS, "classification_fairness",
           "segmentation_fairness", "__version__"]


def __getattr__(name):
    # Imported on demand so that `import fairmedfm` and `fairmedfm --version` stay fast.
    if name in ("evaluate", "evaluate_segmentation", "FairnessReport"):
        from fairmedfm import evaluation
        return getattr(evaluation, name)
    if name in _GROUP_METRICS:
        from fairmedfm import group_metrics
        return getattr(group_metrics, name)
    if name in ("classification_fairness", "segmentation_fairness"):
        from fairmedfm import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module 'fairmedfm' has no attribute {name!r}")
