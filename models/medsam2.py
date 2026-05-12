def build_medsam2(args):
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise ImportError(
            "MedSAM2 requires a SAM2-compatible installation. Install the "
            "MedSAM2/SAM2 package, then pass --sam2_model_cfg and "
            "--sam_ckpt_path."
        ) from exc

    if not args.sam2_model_cfg:
        raise ValueError(
            "MedSAM2 requires --sam2_model_cfg, for example a SAM2.1/MedSAM2 "
            "YAML config path."
        )
    if not args.sam_ckpt_path:
        raise ValueError("MedSAM2 requires --sam_ckpt_path.")

    return build_sam2(
        args.sam2_model_cfg,
        args.sam_ckpt_path,
        device=args.device,
    )
