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


AIMv2 = _optional_model("models.aimv2", "AIMv2")
BiomedCLIP = _optional_model("models.biomed_clip", "BiomedCLIP")
BLIP = _optional_model("models.blip", "BLIP")
BLIP2 = _optional_model("models.blip2", "BLIP2")
C2L = _optional_model("models.c2l", "C2L")
CLIP = _optional_model("models.clip", "CLIP")
CONCH = _optional_model("models.conch", "CONCH")
DINOv2 = _optional_model("models.dinov2", "DINOv2")
DINOv3 = _optional_model("models.dinov3", "DINOv3")
ProvGigaPath = _optional_model("models.gigapath", "ProvGigaPath")
MedCLIP = _optional_model("models.medclip", "MedCLIP")
MedGemma = _optional_model("models.medgemma", "MedGemma")
MedLVM = _optional_model("models.medlvm", "MedLVM")
MedMAE = _optional_model("models.medmae", "MedMAE")
Merlin = _optional_model("models.merlin_ct", "Merlin")
MoCoCXR = _optional_model("models.moco_cxr", "MoCoCXR")
PubMedCLIP = _optional_model("models.pubmed_clip", "PubMedCLIP")
PLIP = _optional_model("models.plip", "PLIP")
RADDINO = _optional_model("models.rad_dino", "RADDINO")
RETFound = _optional_model("models.retfound", "RETFound")
SigLIP = _optional_model("models.siglip", "SigLIP")
SigLIP2 = _optional_model("models.siglip2", "SigLIP2")
MedSigLIP = _optional_model("models.medsiglip", "MedSigLIP")
UNI2 = _optional_model("models.uni2", "UNI2")
Virchow2 = _optional_model("models.virchow2", "Virchow2")


# from models.medklip.model_MedKLIP import MedKLIP
