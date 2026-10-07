"""FairMedFM: group fairness evaluation for binary classification and segmentation models."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fairmedfm")
except PackageNotFoundError:  # source checkout that has not been installed
    __version__ = "0+unknown"

__all__ = ["evaluate", "evaluate_segmentation", "FairnessReport", "classification_fairness",
           "segmentation_fairness", "__version__"]


def __getattr__(name):
    # Imported on demand so that `import fairmedfm` and `fairmedfm --version` stay fast.
    if name in ("evaluate", "evaluate_segmentation", "FairnessReport"):
        from fairmedfm import evaluation
        return getattr(evaluation, name)
    if name in ("classification_fairness", "segmentation_fairness"):
        from fairmedfm import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module 'fairmedfm' has no attribute {name!r}")
