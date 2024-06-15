import torch
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from datasets.base import BaseDataset


class HAM10000(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            return np.asarray(self.dataframe["sex"].values != "male").astype(np.float32)
        elif self.sens_name == "Age":
            # 0-60: 0, >60: 1
            age_binary = self.dataframe["age"].values.astype(np.int8)
            return np.asarray(age_binary >= 60).astype(np.float32)
        else:
            raise NotImplementedError

    def set_label(self):
        Y = self.dataframe["dx"].values.copy()
        Y[Y == "akiec"] = 1
        Y[Y == "mel"] = 1
        Y[Y != 1] = 0

        self.class_nums = 2

        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        if self.path_to_labels is not None:
            img = Image.fromarray(self.tol_images[idx])
        else:
            img = Image.open(os.path.join(self.path_to_images, f"{item['image_id']}.jpg"))

        if self.transform is not None:
            img = self.transform(img)

        return img
