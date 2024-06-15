import torch
import pickle
import numpy as np
from PIL import Image
import os
import pickle
from .base import BaseDataset


class FairVLMed10k(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            return np.asarray(self.dataframe["gender"].values != "male").astype(np.float32)
        elif self.sens_name == "Age":
            # 0-60: 0, >60: 1
            age_binary = self.dataframe["age"].values.astype(np.float32)
            return np.asarray(age_binary >= 60).astype(np.float32)
        elif self.sens_name == "Race":
            # white: 0, non-white: 1
            return np.asarray(self.dataframe["race"].values != "white").astype(np.float32)
        elif self.sens_name == "language":
            return np.asarray(self.dataframe["language_binary"].values != "english").astype(np.float32)
        else:
            raise NotImplementedError

    def set_label(self):
        Y = np.where(
            self.dataframe["glaucoma"] == "yes",
            1,
            0,
        )
        self.class_nums = 2

        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        if self.path_to_labels is not None:
            img = Image.fromarray(self.tol_images[idx])
        else:
            img = Image.open(os.path.join(self.path_to_images, f"{item['path']}")).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img
