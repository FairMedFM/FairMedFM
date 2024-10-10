import argparse
import os
from ast import parse


def collect_args():
    parser = argparse.ArgumentParser()

    # experiments

    parser.add_argument("--task", default="cls", choices=["cls", "seg"])
    parser.add_argument(
        "--usage",
        type=str,
        default='clip-zs',
        choices=["lp", "clip-zs", "clip-adapt", "seg2d-rand",
                 "seg2d-rands", "seg2d-center", "seg2d-bbox", "seg2d"],
    )
    parser.add_argument("--method", default="erm",
                        choices=["erm", "resampling", "group-dro", "laftr"])
    parser.add_argument(
        "--dataset",
        default="CXP",
        choices=[
            "CXP",
            "MIMIC_CXR",
            "HAM10000",
            "PAPILA",
            "ADNI",
            "COVID_CT_MD",
            "FairVLMed10k",
            "BREST",
            "GF3300",
            "HAM10000-Seg",
            "FairSeg",
            "montgomery",
            "TUSC"
        ],
    )
    parser.add_argument("--sensitive_name", default="Sex",
                        choices=["Sex", "Age", "Race", "Language"])
    parser.add_argument("--is_3d", action="store_true")
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--wandb_name", type=str, default="baseline")
    parser.add_argument("--if_wandb", type=bool, default=False)

    parser.add_argument("--resume_path", type=str, default="",
                        help="explicitly indentify checkpoint path to resume.")

    # training
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--optimizer", default="adamw",
                        choices=["sgd", "adam", "adamw"])
    parser.add_argument("--blr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--fixed_lr", action="store_true")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-4, help="weight decay for optimizer")
    parser.add_argument("--lr_decay_rate", type=float,
                        default=0.1, help="decay rate of the learning rate")
    parser.add_argument("--lr_decay_period", type=float,
                        default=10, help="decay period of the learning rate")
    parser.add_argument("--total_epochs", type=int,
                        default=100, help="total training epochs")
    parser.add_argument("--early_stopping", type=int,
                        default=1, help="early stopping epochs")
    parser.add_argument("--test_mode", type=bool,
                        default=False, help="if using test mode")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    parser.add_argument("--no_cls_balance",
                        dest="cls_balance", action="store_false")

    # network
    parser.add_argument(
        "--model",
        default="BiomedCLIP",
        choices=[
            "BiomedCLIP",
            "PubMedCLIP",
            "MedCLIP",
            "CLIP",
            "BLIP",
            "BLIP2",
            "DINOv2",
            "MedLVM",
            "C2L",
            "MedMAE",
            "MoCoCXR",
            "SAM",
            "MedSAM",
            "SAMMed2D",
            "FT-SAM",
            "TinySAM",
            "MobileSAM"
        ],
    )
    parser.add_argument("--context_length", default=77)

    # testing
    parser.add_argument("--hash_id", type=str, default="")

    # strategy for validation
    parser.add_argument(
        "--val_strategy",
        type=str,
        default="loss",
        choices=["loss", "worst_auc"],
        help="strategy for selecting val model",
    )

    parser.set_defaults(cuda=True)

    # logging
    parser.add_argument("--log_freq", type=int, default=50,
                        help="logging frequency (step)")
    parser.add_argument("--exp_path", type=str, default="./output")

    # segment_specific
    parser.add_argument("--pos_class", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--sam_ckpt_path", type=str)
    parser.add_argument("--prompt", type=str,
                        choices=["bbox", "rand", "rands", "center"])

    args = parser.parse_args()
    return args
