import importlib


def _optional_model(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as exc:
        missing_error = exc

        class MissingModel:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"{class_name} could not be imported because an optional "
                    f"dependency is missing: {missing_error}"
                ) from missing_error

        MissingModel.__name__ = class_name
        return MissingModel


try:
    from .sam_builder import build_sammed2d, build_tinysam
except ImportError:
    build_sammed2d = None
    build_tinysam = None


BiomedCLIP = _optional_model("models.biomed_clip", "BiomedCLIP")
BLIP = _optional_model("models.blip", "BLIP")
BLIP2 = _optional_model("models.blip2", "BLIP2")
C2L = _optional_model("models.c2l", "C2L")
CLIP = _optional_model("models.clip", "CLIP")
DINOv2 = _optional_model("models.dinov2", "DINOv2")
MedCLIP = _optional_model("models.medclip", "MedCLIP")
MedLVM = _optional_model("models.medlvm", "MedLVM")
MedMAE = _optional_model("models.medmae", "MedMAE")
MoCoCXR = _optional_model("models.moco_cxr", "MoCoCXR")
PubMedCLIP = _optional_model("models.pubmed_clip", "PubMedCLIP")
PLIP = _optional_model("models.plip", "PLIP")
RADDINO = _optional_model("models.rad_dino", "RADDINO")
SigLIP = _optional_model("models.siglip", "SigLIP")
MedSigLIP = _optional_model("models.medsiglip", "MedSigLIP")


# from models.medklip.model_MedKLIP import MedKLIP
