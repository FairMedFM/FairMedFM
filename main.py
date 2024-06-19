import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import hashlib
import time
from utils import basics
import parse_args
from tqdm import tqdm

from datasets.utils import get_dataset
from models.utils import get_model
from wrappers.utils import get_warpped_model
from trainers.utils import get_trainer


def create_exerpiment_setting(args):
    # get hash
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr
    # opt["lr"] = opt["blr"] * opt["batch_size"] / 256

    args.save_folder = os.path.join(
        args.exp_path,
        args.task,
        args.usage,
        args.dataset,
        args.model,
        args.sensitive_name,
        f"seed{args.random_seed}",
    )

    args["resume_path"] = args["save_folder"]
    basics.creat_folder(args["save_folder"])

    try:
        with open(f"configs/datasets/{args.dataset}.json", "r") as f:
            data_setting = json.load(f)
            data_setting["augment"] = False
            data_setting["test_meta_path"] = data_setting[f"test_{str.lower(args["sensitive_name"])}_meta_path"]
            args.data_setting = data_setting
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
        "train", args["save_folder"], "history.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    train_data, train_dataloader, train_meta = get_dataset(args, split="train")
    test_data, test_dataloader, test_meta = get_dataset(args, split="test")
    model = get_model(args).to(args.device)

    if args.task == "cls":
        model = get_warpped_model(args, model).to(args.device)
    elif args.task == "seg":
        model = get_warpped_model(args, model, test_data).to(
            args.device)  # SAMLearner

    trainer = get_trainer(args, model, logger)

    if args.usage == "clip-zs":
        logger.info("Zero-shot performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "zs"))
        exit(0)

    elif args.usage == "seg2d-center":
        logger.info("2D SegFM using 1 center point prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "center"))
        exit(0)
        # for seg, teh test_dataloader is not used.

    elif args.usage == "seg2d-rand":
        logger.info("2D SegFM using 1 random point prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "rand"))
        exit(0)

    elif args.usage == "seg2d-rands":
        logger.info("2D SegFM using 5 random points prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "rands"))
        exit(0)

    elif args.usage == "seg2d-bbox":
        logger.info("2D SegFM using 1 bounding box prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "bbox"))
        exit(0)

    elif args.usage == "seg3d-center":
        # TODO
        logger.info("3D SegFM using 1 center point prompt performance:")
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "center"))
        exit(0)

    logger.info("Start training")
    trainer.train(train_dataloader, test_dataloader)

    logger.info("Final results:")
    trainer.evaluate(test_dataloader, save_path=os.path.join(
        args["save_folder"], "lp_final"))
