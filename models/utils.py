import json

import torch
from models.sam_builder.build_sammed2d import sam_model_registry1
from models.sam_builder.build_tinysam import sam_model_registry2
from segment_anything.build_sam import sam_model_registry

import models


def get_model(args):
    if args.task == "cls":
        model_setting = args.model_setting

        model_name = getattr(models, args.model)

        model = model_name()

        if model_setting is not None and "pretrained_path" in model_setting.keys():
            model.from_pretrained(model_setting["pretrained_path"])

    elif args.task == "seg":
        sam_checkpoint = args.resume_path

        if args.model in ["SAM", "MedSAM", "MedSAMAdaptor"]:
            model = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        elif args.model in ["SAMMed2D", "FT-SAM"]:
            model = sam_model_registry1['vit_b'](args)
        elif args.model in ["TinySAM", "MobileSAM"]:
            model = sam_model_registry2['vit_t'](checkpoint=sam_checkpoint)
        else:
            raise ValueError("Invalid model name!")

    else:
        raise NotImplementedError("Task type not supported!")

    return model
