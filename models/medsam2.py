def build_sam2(args):
    """Builds a SAM2-family model (vanilla SAM2 or MedSAM2).

    Both share the same build path; only the config/checkpoint passed via
    --sam2_model_cfg / --sam_ckpt_path differ.
    """
    try:
        from sam2.build_sam import build_sam2 as _build_sam2
    except ImportError as exc:
        raise ImportError(
            "SAM2 requires a SAM2-compatible installation. Install the "
            "SAM2/MedSAM2 package, then pass --sam2_model_cfg and "
            "--sam_ckpt_path."
        ) from exc

    if not args.sam2_model_cfg:
        raise ValueError(
            "SAM2 requires --sam2_model_cfg, for example a SAM2.1/MedSAM2 "
            "YAML config path."
        )
    if not args.sam_ckpt_path:
        raise ValueError("SAM2 requires --sam_ckpt_path.")

    return _build_sam2(
        args.sam2_model_cfg,
        args.sam_ckpt_path,
        device=args.device,
    )


# Backward-compatible alias: MedSAM2 is built the exact same way as SAM2.
build_medsam2 = build_sam2
