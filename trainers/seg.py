import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base import BaseTrainer
from utils.basics import creat_folder
from utils.metrics import evaluate_binary, organize_results


class SegTrainer(BaseTrainer):
    def __init__(self, args, model, logger) -> None:
        super().__init__(args, model, logger)

    def evaluate(self, dataloader, save_path=None):
        # TODO:
        '''
        trainer.evaluate(test_dataloader, save_path=os.path.join(
            args["save_folder"], "rand"))
        '''
        pass
