from fairmedfm_bench import models
from fairmedfm_bench.models.medsam2 import build_medsam2


def get_model(args):
    if args.task == "cls":
        model_setting = args.model_setting

        model_name = getattr(models, args.model)

        pretrained_path = None
        if model_setting is not None and "pretrained_path" in model_setting.keys():
            pretrained_path = model_setting["pretrained_path"]

        backbone_kwarg_models = [
            "SigLIP", "MedSigLIP", "RADDINO", "DINOv3", "MedGemma",
            "SigLIP2", "AIMv2", "Virchow2", "UNI2", "ProvGigaPath", "CONCH",
        ]

        if args.model in backbone_kwarg_models and pretrained_path is not None:
            model = model_name(backbone=pretrained_path)
        else:
            model = model_name()

        if pretrained_path is not None and args.model not in backbone_kwarg_models:
            model.from_pretrained(model_setting["pretrained_path"])

    elif args.task == "seg":
        sam_checkpoint = args.sam_ckpt_path
        if sam_checkpoint == None and args.model not in ["SAM3"]:
            raise ValueError(
                "SAM checkpoint path is required for segmentation task!")
        # ic(sam_checkpoint)
        if args.model in ["SAM", "MedSAM", "MedSAMAdaptor"]:
            from segment_anything.build_sam import sam_model_registry
            model = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        elif args.model in ["SAMMed2D", "FT-SAM"]:
            from fairmedfm_bench.models.sam_builder.build_sammed2d import sam_model_registry1
            model = sam_model_registry1['vit_b'](args)
        elif args.model in ["TinySAM", "MobileSAM"]:
            from fairmedfm_bench.models.sam_builder.build_tinysam import sam_model_registry2
            model = sam_model_registry2['vit_t'](checkpoint=sam_checkpoint)
        elif args.model in ["MedSAM2", "SAM2"]:
            model = build_medsam2(args)
        elif args.model in ["SAM3", "MedicalSAM3"]:
            from fairmedfm_bench.models.sam3 import build_sam3
            model = build_sam3(args)
        else:
            raise ValueError("Invalid model name!")

        print(f"Weights of model: #{args.model}# is loaded!")

    else:
        raise NotImplementedError("Task type not supported!")

    return model
