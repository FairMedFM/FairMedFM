import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

import parse_args
from datasets.utils import get_dataset
from models.utils import get_model
from trainers.utils import get_trainer
from utils import basics
from wrappers.utils import get_warpped_model

os.environ["WANDB_DISABLED"] = "true"


def create_exerpiment_setting(args):
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

    try:
        with open(f"configs/datasets/{args.dataset}.json", "r") as f:
            data_setting = json.load(f)
            data_setting["augment"] = False
            data_setting["test_meta_path"] = data_setting[
                f"test_{str.lower(args.sensitive_name)}_meta_path"]
            args.data_setting = data_setting

            if args.pos_class is not None:
                args.data_setting["pos_class"] = args.pos_class
    except:
        args.data_setting = None

    try:
        with open(f"configs/models/{args.model}.json", "r") as f:
            args.model_setting = json.load(f)
    except:
        args.model_setting = None

    return args


if __name__ == "__main__":
    args = parse_args.collect_args()
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
        exit(0)

    elif args.usage == "clip-adapt":
        logger.info("CLIP-Adaptor performance:")
        trainer.init_optimizers()
        trainer.train(train_dataloader)
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, "clip_adaptor_final"))
        exit(0)

    elif args.usage == "lp":
        logger.info("Linear probing performance:")
        trainer.init_optimizers()
        trainer.train(train_dataloader)
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, "lp_final"))
        exit(0)

    elif args.usage == "seg2d":
        logger.info(f"2D SegFM using {args.prompt} prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args.save_folder, args.prompt))
        exit(0)

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
        exit(0)

    else:
        raise NotImplementedError
