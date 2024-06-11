import torch
import pickle
import numpy as np
from PIL import Image
import os
import pickle
from datasets.base import BaseDataset


class BREST(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            return np.asarray(self.dataframe["patient_sex"].values != 1).astype(np.float32)
        elif self.sens_name == "Age":
            # 0-60: 0, >60: 1
            return self.dataframe["age_binary"].values.astype(np.float32)
        else:
            raise NotImplementedError

    def set_label(self):
        Y = np.where(
            self.dataframe["DR_ICDR"] != 0,
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
            img = Image.open(os.path.join(self.path_to_images, f"{item['image_id']}.jpg")).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img
