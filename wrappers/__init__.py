import importlib


def _optional_wrapper(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as exc:
        missing_error = exc

        class MissingWrapper:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"{class_name} could not be imported because an optional "
                    f"dependency is missing: {missing_error}"
                ) from missing_error

        MissingWrapper.__name__ = class_name
        return MissingWrapper


CLIPWrapper = _optional_wrapper("wrappers.clip", "CLIPWrapper")
LinearProbeWrapper = _optional_wrapper("wrappers.linear_probe", "LinearProbeWrapper")
LoRAWrapper = _optional_wrapper("wrappers.lora", "LoRAWrapper")
MedSAM2Wrapper = _optional_wrapper("wrappers.medsam2", "MedSAM2Wrapper")
SAMWrapper = _optional_wrapper("wrappers.sam", "SAMWrapper")
