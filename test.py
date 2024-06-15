import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import hashlib
import time
from util import basics
import parse_args
from tqdm import tqdm
from cached_path import set_cache_dir


from datasets.utils import get_dataset
from models.utils import get_model, tokenize_text
from trainers.base import LPTrainer


def create_exerpiment_setting(opt):

    # get hash
    opt["device"] = torch.device("cuda" if opt["cuda"] else "cpu")
    opt["lr"] = opt["blr"]
    # opt["lr"] = opt["blr"] * opt["batch_size"] / 256

    opt["save_folder"] = os.path.join(
        opt["exp_path"],
        opt["experiment"],
        f"seed{opt['random_seed']}",
        opt["dataset_name"],
        opt["model"],
        opt["sensitive_name"],
    )

    if opt["resume_path"] == "":
        opt["resume_path"] = os.path.join(
            opt["exp_path"],
            opt["experiment"],
            f"seed{opt['random_seed']}",
            opt["dataset_name"],
            opt["model"],
            "Sex",
        )

    basics.creat_folder(opt["save_folder"])

    with open("configs/datasets.json", "r") as f:
        data_path = json.load(f)

    try:
        data_setting = data_path[opt["dataset_name"]]
        data_setting["augment"] = False
        data_setting["test_meta_path"] = data_setting[f"test_{opt['sensitive_name'].lower()}_meta_path"]
    except:
        data_setting = {}

    opt["data_setting"] = data_setting

    with open("configs/clip.json", "r") as f:
        try:
            opt["clip_setting"] = json.load(f)[opt["dataset_name"]]
        except:
            opt["clip_setting"] = {}

    with open("configs/models.json", "r") as f:
        try:
            opt["model_setting"] = json.load(f)[opt["model"]]
        except:
            opt["model_setting"] = {}

    return opt


if __name__ == "__main__":
    # set_cache_dir("/research/d5/gds/yzhong22/misc/cache")

    opt = parse_args.collect_args()
    opt = create_exerpiment_setting(opt)

    logger = basics.setup_logger("train", opt["save_folder"], "test.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(opt)

    torch.manual_seed(opt["random_seed"])
    np.random.seed(opt["random_seed"])

    # _, train_dataloader, _ = get_dataset(opt, split="train")
    _, test_dataloader, _ = get_dataset(opt, split="test")
    model = get_model(opt).to(opt["device"])
    text = tokenize_text(opt, test_dataloader.dataset.get_class_names())

    trainer = LPTrainer(opt, model, text, logger)
    logger.info("Zero-shot performance:")
    trainer.evaluate(test_dataloader, save_path=os.path.join(opt["save_folder"], "zs"))

    ckpt_path = os.path.join(opt["resume_path"], "ckpt.pth")
    logger.info(f"Load checkpoint from: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)

    logger.info("Final results:")
    trainer.evaluate(test_dataloader, save_path=os.path.join(opt["save_folder"], "lp_final"))
