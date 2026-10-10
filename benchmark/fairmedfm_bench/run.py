"""Run a FairMedFM benchmark experiment: ``fairmedfm run`` (or ``python main.py`` in a checkout)."""
import json
import os
import random
from pathlib import Path

from fairmedfm_bench import parse_args

os.environ["WANDB_DISABLED"] = "true"


def config_path(kind, name):
    """A config from ./configs in the working directory if present, otherwise the packaged one."""
    local = Path.cwd() / "configs" / kind / f"{name}.json"
    return local if local.exists() else Path(__file__).resolve().parent / "configs" / kind / f"{name}.json"


def create_exerpiment_setting(args):
    import torch

    from fairmedfm_bench.utils import basics

    # get hash
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr

    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.usage,
        args.method,
        args.dataset,
        args.model,
        args.sensitive_name,
        f"seed{args.random_seed}",
    )

    args.resume_path = args.save_folder
    basics.creat_folder(args.save_folder)

    data_path = config_path("datasets", args.dataset)
    if not data_path.exists():
        raise FileNotFoundError(f"no dataset config for {args.dataset}: add configs/datasets/{args.dataset}.json "
                                "in the working directory (see the benchmark documentation)")
    data_setting = json.loads(data_path.read_text())
    data_setting["augment"] = False
    split_key = f"test_{args.sensitive_name.lower()}_meta_path"
    if split_key not in data_setting:
        available = sorted(k[len("test_"):-len("_meta_path")] for k in data_setting
                           if k.startswith("test_") and k.endswith("_meta_path") and k != "test_meta_path")
        raise ValueError(f"{data_path} has no {split_key!r}: {args.dataset} has test splits for "
                         f"{', '.join(available) or 'no sensitive attribute'}, not {args.sensitive_name}")
    data_setting["test_meta_path"] = data_setting[split_key]
    if args.pos_class is not None:
        data_setting["pos_class"] = args.pos_class
    args.data_setting = data_setting

    # Models without a config (for example CLIP) need no pretrained path or LoRA targets.
    model_path = config_path("models", args.model)
    args.model_setting = json.loads(model_path.read_text()) if model_path.exists() else None

    return args


def main(argv=None):
    args = parse_args.collect_args(argv)

    import numpy as np
    import torch

    from fairmedfm_bench.datasets.utils import get_dataset
    from fairmedfm_bench.models.utils import get_model
    from fairmedfm_bench.trainers.utils import get_trainer
    from fairmedfm_bench.utils import basics
    from fairmedfm_bench.wrappers.utils import get_warpped_model

    args = create_exerpiment_setting(args)

    logger = basics.setup_logger(
        "train", args.save_folder, "history.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data, train_dataloader, train_meta = get_dataset(args, split="train")
    test_data, test_dataloader, test_meta = get_dataset(args, split="test")
    model = get_model(args).to(args.device)

    # ic(train_data, train_dataloader, train_meta)
    # ic(test_data, test_dataloader, test_meta)
    # ic(model)

    if args.task == "cls":
        model = get_warpped_model(args, model).to(args.device)
    elif args.task == "seg":
        model = get_warpped_model(args, model, test_data).to(
            args.device)  # SAMLearner

    trainer = get_trainer(args, model, logger, test_dataloader)

    if args.usage == "clip-zs":
        logger.info("CLIP Zero-shot performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, "clip_zs_final"))
        return

    elif args.usage == "clip-adapt":
        logger.info("CLIP-Adaptor performance:")
        trainer.init_optimizers()
        trainer.train(train_dataloader)
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, "clip_adaptor_final"))
        return

    elif args.usage == "lp":
        logger.info("Linear probing performance:")
        trainer.init_optimizers()
        trainer.train(train_dataloader)
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, "lp_final"))
        return

    elif args.usage == "seg2d":
        logger.info(f"2D SegFM using {args.prompt} prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, args.prompt))
        return

    # elif args.usage == "seg2d-rands":
    #     logger.info("2D SegFM using 5 random points prompt performance:")
    #     trainer.evaluate(test_dataloader, save_path=os.path.join(
    #         args.save_folders, "rands"))
    #     exit(0)

    # elif args.usage == "seg2d-bbox":
    #     logger.info("2D SegFM using 1 bounding box prompt performance:")
    #     trainer.evaluate(test_dataloader, save_path=os.path.join(
    #         args.save_folder, "bbox"))
    #     exit(0)

    elif args.usage == "seg3d-center":
        # TODO
        logger.info("3D SegFM using 1 center point prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, "center"))
        return

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
