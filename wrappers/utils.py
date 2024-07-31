import json

import torch

import models
from utils.static import CLIP_MODELS
from utils.tokenizer import tokenize_text


def get_warpped_model(args, model, data_engine=None):

    if args.task == "cls":
        if args.usage == "lp":
            from wrappers import LinearProbeWrapper

            model_warpped = LinearProbeWrapper(model)
        elif args.usage in ["clip-zs", "clip-adapt"]:
            assert args.model in CLIP_MODELS, f"{args.usage} is not applicable for {args.model}"
            from wrappers import CLIPWrapper

            text = tokenize_text(args, args.data_setting["class_names"])
            text_features = model.encode_text(text.to(args.device))

            model_warpped = CLIPWrapper(model, text_features)
        elif args.usage == "lora":
            from wrappers import LoRAWrapper

            model_setting = args.model_setting

            assert (
                "lora_targets" in model_setting.keys()
            ), f"LoRA is not applicable for {args.model}, either because it's not a ViT-based model or it's not supported in the current version"

            model_warpped = LoRAWrapper(
                model, lora_targets=model_setting["lora_targets"])
        else:
            raise NotImplementedError()

    elif args.task == "seg":
        from wrappers import SAMWrapper
        model_warpped = SAMWrapper(model, data_engine=data_engine)
    else:
        raise NotImplementedError

    return model_warpped
