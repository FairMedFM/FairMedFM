import os

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.nn.functional as F

from utils.lr_sched import adjust_learning_rate


class BaseTrainer(object):
    def __init__(self, args, model, logger) -> None:
        self.args = args
        self.device = args.device

        self.model = model.to(self.device)

        self.logger = logger

        self.init()

    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()

        while self.epoch < self.total_epochs:
            adjust_learning_rate(self.optimizer, self.epoch + 1, self.args)

            loss = self.train_epoch(train_dataloader)
            self.logger.info("epoch {}: lr {}, loss {}".format(
                self.epoch, self.optimizer.param_groups[0]["lr"], loss))

            if val_dataloader is not None:
                self.evaluate(val_dataloader)

            # save model
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                },
                os.path.join(self.args.save_folder, "ckpt.pth"),
            )

            if self.args.early_stopping and val_dataloader is not None:
                if self.epoch > 10 and (max(self.last_five_auc) - min(self.last_five_auc) < 1e-5):
                    break

            self.epoch += 1

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model.load_state_dict(ckpt["model"])

    def train_epoch(self, train_dataloader):
        loss_epoch = []
        for minibatch in train_dataloader:
            loss_batch = self.update_batch(minibatch)

            loss_epoch.append(loss_batch.item())

        return np.mean(loss_epoch)

    def update_batch(self, minibatch):
        pass

    def init_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.args.weight_decay)
        self.optimizer = torch.optim.SGD(param_groups, lr=self.args.lr)

    def init(self):
        self.start_epoch = 0
        self.epoch = self.start_epoch

        self.total_epochs = self.args.total_epochs

        self.last_five_auc = []

    def evaluate(self, dataloader, save_path=None):
        pass
