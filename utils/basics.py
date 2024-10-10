import csv
import json
import logging
import os
import pickle
import random
import time
from importlib import import_module

import numpy as np
import pandas as pd
import torch


def save_results(t_predictions, tol_target, s_prediction, tol_sensitive, path):
    np.save(os.path.join(path, "tpredictions.npy"), np.asarray(t_predictions))
    np.save(os.path.join(path, "ttargets.npy"), np.asarray(tol_target))
    np.save(os.path.join(path, "spredictions.npy"), np.asarray(s_prediction))
    np.save(os.path.join(path, "stargets.npy"), np.asarray(tol_sensitive))


def save_result_csv(log_dict, path):
    with open(path + "/results.csv", "w") as f:
        w = csv.DictWriter(f, log_dict.keys())
        w.writeheader()
        w.writerow(log_dict)


def add_dict_prefix(dicts, prefix):
    new_dict = {}
    for k, v in dicts.items():
        new_dict[prefix + k] = dicts[k]
    return new_dict


def get_model(opt, wandb):
    mod = import_module("models" + "." + opt["experiment"])
    model_name = getattr(mod, opt["experiment"])
    model = model_name(opt, wandb)
    return model


def avg_eval(val_df, opt, mode="val"):
    val_df = val_df.reset_index(drop=True)

    mean_df = val_df.mean()
    std_df = val_df.std()
    sem_df = val_df.sem()
    ci95_hi = pd.DataFrame(mean_df + 1.96 * sem_df).transpose()
    ci95_lo = pd.DataFrame(mean_df - 1.96 * sem_df).transpose()
    mean_df = pd.DataFrame(mean_df).transpose()
    std_df = pd.DataFrame(std_df).transpose()

    stat = pd.concat([mean_df, std_df, ci95_hi, ci95_lo]
                     ).reset_index(drop=True)
    stat = stat.rename(index={0: "mean", 1: "std", 2: "ci95_hi", 3: "ci95_lo"})
    save_path = os.path.join(opt["save_folder"], opt["experiment"] +
                             "_" + opt["hash"] + "_" + mode + "_pred_stat.csv")
    stat.to_csv(save_path)
    return stat


def save_pkl(pkl_data, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(pkl_data, f)


def load_pkl(load_path):
    with open(load_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def save_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(load_path):
    with open(load_path, "r") as f:
        json_data = json.load(f)
    return json_data


def save_state_dict(state_dict, save_path):
    torch.save(state_dict, save_path)


def creat_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(
            root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    return lg
