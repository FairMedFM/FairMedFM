import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from trainers.base import BaseTrainer

from utils.metrics import evaluate_binary, organize_results
from utils.basics import creat_folder


class CLSTrainer(BaseTrainer):
    def __init__(self, args, model, logger) -> None:
        super().__init__(args, model, logger)

    def update_batch(self, minibatch):
        x = minibatch["img"].to(self.device)
        y = minibatch["label"].to(self.device)

        if self.args["is_3d"]:
            # input shape: B x N_slice x 3 x H x W
            # output shape: B x N_slice x N_classes
            # logits list for each slice
            logits_sliced = torch.stack([self.model(x[:, i]) for i in range(x.shape[1])], dim=1)
            prob_sliced = F.softmax(logits_sliced, dim=-1)
            indices = torch.argmax(prob_sliced[:, :, 1], dim=1)

            logits = torch.stack([logits_sliced[i, idx] for i, idx in enumerate(indices)])
            loss = F.cross_entropy(logits, y.long().squeeze(-1))
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits, y.long().squeeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def evaluate(self, dataloader, save_path=None):
        self.model.eval()

        logits_list = []
        prob_list = []
        target_list = []
        sensitive_list = []

        for minibatch in dataloader:
            x = minibatch["img"].to(self.device)
            y = minibatch["label"].to(self.device)
            a = minibatch["sensitive"].to(self.device)

            with torch.no_grad():
                if self.args["is_3d"]:
                    logits_sliced = torch.stack([self.model(x[:, i]) for i in range(x.shape[1])], dim=1)
                    prob_sliced = F.softmax(logits_sliced, dim=-1)
                    indices = torch.argmax(prob_sliced[:, :, 1], dim=1)

                    logits = torch.stack([logits_sliced[i, idx] for i, idx in enumerate(indices)])
                else:
                    logits = self.model(x)
            prob = F.softmax(logits, dim=-1)

            logits_list.append(logits)
            prob_list.append(prob)
            target_list.append(y)
            sensitive_list.append(a)

        logits_list = torch.concat(logits_list).squeeze().cpu().numpy()
        prob_list = torch.concat(prob_list).squeeze().cpu().numpy()
        target_list = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sensitive_list = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)

        overall_metrics, subgroup_metrics = evaluate_binary(prob_list[:, 1], target_list, sensitive_list)
        organized_metrics = organize_results(overall_metrics, subgroup_metrics)

        if self.args["early_stopping"] == 1:
            if len(self.last_five_auc) >= 5:
                self.last_five_auc.pop(0)

            self.last_five_auc.append(overall_metrics["auc"])

        self.logger.info("----------------------------------------------".format(self.epoch))
        self.logger.info("----------------eva epoch {}------------------".format(self.epoch))
        self.logger.info(
            "{}".format(
                ", ".join("{}: {}".format(k, v) for k, v in organized_metrics.items()),
            )
        )
        self.logger.info("-----------------meta info-------------------")
        self.logger.info(
            "overall metrics: {}".format(
                ", ".join("{}: {}".format(k, v) for k, v in overall_metrics.items()),
            )
        )
        self.logger.info(
            "subgroup metrics: {}".format(
                ", ".join("{}: {}".format(k, v) for k, v in subgroup_metrics.items()),
            )
        )
        self.logger.info("----------------------------------------------".format(self.epoch))

        # save predictions
        if save_path is not None:
            creat_folder(save_path)

            with open(os.path.join(save_path, "metrics.pkl"), "wb") as f:
                pickle.dump({"epoch": self.epoch, "overall": overall_metrics, "subgroup": subgroup_metrics}, f)

            with open(os.path.join(save_path, "predictions.pkl"), "wb") as f:
                pickle.dump({"epoch": self.epoch, "logits": logits_list, "label": target_list}, f)
