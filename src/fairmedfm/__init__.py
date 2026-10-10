"""FairMedFM: group fairness evaluation for binary classification and segmentation models."""
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

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


if TYPE_CHECKING:  # what the lazy imports below provide, for type checkers and editors
    from .evaluation import FairnessReport, evaluate, evaluate_segmentation
    from .group_metrics import (accuracy_gap, auc_gap, bce_gap, dice_gap, dice_skewness, dice_std, ece_gap,
                                equal_opportunity_difference, equalized_odds_score, equity_scaled_dice,
                                worst_group_auc, worst_group_dice)
    from .metrics import classification_fairness, segmentation_fairness


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
