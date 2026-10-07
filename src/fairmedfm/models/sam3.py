"""Builders for the official SAM 3 and Medical SAM3 image checkpoints."""

import torch


def build_sam3(args):
    try:
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise ImportError(
            "SAM3 requires the official sam3 package. Install it from "
            "https://github.com/facebookresearch/sam3 in a compatible environment."
        ) from exc

    if args.model == "MedicalSAM3" and args.prompt != "bbox":
        raise ValueError("MedicalSAM3 currently supports --prompt bbox only.")

    interactive = args.model == "SAM3"
    if args.model == "MedicalSAM3" and not args.sam_ckpt_path:
        raise ValueError("MedicalSAM3 requires --sam_ckpt_path to its 2D checkpoint.")

    # The medical checkpoint contains the detector weights without the
    # `detector.` prefix used by Meta's released SAM 3 checkpoint.
    model = build_sam3_image_model(
        device=str(args.device),
        checkpoint_path=args.sam_ckpt_path if interactive else None,
        load_from_HF=interactive and not args.sam_ckpt_path,
        enable_inst_interactivity=interactive,
    )
    if not interactive:
        checkpoint = torch.load(args.sam_ckpt_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("MedicalSAM3 checkpoint has no model state dictionary.")
        state = {
            key.removeprefix("detector."): value
            for key, value in state.items()
            if not key.startswith("tracker.")
        }
        model_keys = set(model.state_dict())
        matched = model_keys.intersection(state)
        if len(matched) < 0.95 * len(model_keys):
            raise ValueError(
                f"MedicalSAM3 checkpoint matches only {len(matched)}/{len(model_keys)} "
                "model tensors; check that this is a 2D Medical SAM3 checkpoint."
            )
        model.load_state_dict(state, strict=False)
    return model.eval()
