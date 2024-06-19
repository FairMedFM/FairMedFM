import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import dice

from trainers.base import BaseTrainer
from utils.basics import creat_folder
from utils.metrics import evaluate_binary  # implement evaluate_seg
from utils.metrics import organize_results


class SegTrainer(BaseTrainer):
    def __init__(self, args, model, logger) -> None:
        super().__init__(args, model, logger)

    def evaluate(self, dataloader, save_path=None):
        self.model.eval()

        dsc_list = []
        sensitive_list = []

        for minibatch in dataloader:
            x = minibatch["image"].to(self.device)
            y = minibatch["label"].to(self.device)
            a = minibatch["sex"].to(self.device)  # modify sex with sensitive
            prompt_point = minibatch['prompt_point'].to(self.device)
            prompt_box = minibatch['prompt_box'].to(self.device)
            prompt_center = minibatch["prompt_center"].to(self.device)
            prompt_rands = minibatch["prompt_rands"].to(self.device)

            self.model.set_torch_image(
                x, (self.args.image_size, self.args.image_size))

            with torch.no_grad():
                f = self.model.encode({"img": x})

                if self.args.prompt == "rand":
                    new_outputs, new_iou = self.model.decode({
                        "features": f,
                        "prompt_point": prompt_point,
                        "prompt_box": prompt_box}, flag="point")
                elif self.args.prompt == "center":
                    new_outputs, new_iou = self.model.decode({
                        "features": f,
                        "prompt_point": prompt_center,
                        "prompt_box": prompt_box}, flag="point")
                elif self.args.prompt == "rands":
                    new_outputs, new_iou = self.model.decode({
                        "features": f,
                        "prompt_point": prompt_rands,
                        "prompt_box": prompt_box}, flag="point")
                elif self.args.prompt == "bbox":
                    new_outputs, new_iou = self.model.decode({
                        "features": f,
                        "prompt_point": prompt_box,
                        "prompt_box": prompt_box}, flag="bbox")

                mask = new_outputs[0][new_iou.argmax()]

                dice_metric = dice(mask, y, ignore_index=0)
            dsc_list.append(dice_metric.item())
            sensitive_list.append(a.cpu().numpy())

        # overall_metrics, subgroup_metrics = evaluate_seg(
        #     dsc_list, sensitive_list)
        overall_metrics, subgroup_metrics = None, None
        organized_metrics = organize_results(overall_metrics, subgroup_metrics)
        self.logger.info(
            "----------------------------------------------".format(self.epoch))
        self.logger.info(
            "----------------eva epoch {}------------------".format(self.epoch))
        self.logger.info(
            "{}".format(
                ", ".join("{}: {}".format(k, v)
                          for k, v in organized_metrics.items()),
            )
        )
        self.logger.info("-----------------meta info-------------------")
        self.logger.info(
            "overall metrics: {}".format(
                ", ".join("{}: {}".format(k, v)
                          for k, v in overall_metrics.items()),
            )
        )
        self.logger.info(
            "subgroup metrics: {}".format(
                ", ".join("{}: {}".format(k, v)
                          for k, v in subgroup_metrics.items()),
            )
        )
        self.logger.info(
            "----------------------------------------------".format(self.epoch))
        if save_path is not None:
            creat_folder(save_path)

            with open(os.path.join(save_path, "metrics.pkl"), "wb") as f:
                pickle.dump(
                    {"epoch": self.epoch, "overall": overall_metrics, "subgroup": subgroup_metrics}, f)

            with open(os.path.join(save_path, "predictions.pkl"), "wb") as f:
                pickle.dump(
                    {"epoch": self.epoch, "dice": dsc_list}, f)
