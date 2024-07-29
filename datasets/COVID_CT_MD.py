import torch
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import SimpleITK as sitk

from datasets.base import BaseDataset


class COVID_CT_MD(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

        self.window = (-1250, 250)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            sa_array = np.asarray(self.dataframe["Patient Gender"].values != "M").astype(np.float32)
            class_0_count = np.sum(sa_array == 0)
            class_1_count = np.sum(sa_array == 1)
            total_count_sa = len(sa_array)
            weight_class_0 = total_count_sa / (2 * class_0_count)
            weight_class_1 = total_count_sa / (2 * class_1_count)
            self.class_weights_sa = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
            return sa_array
        elif self.sens_name == "Age":
            # 0-60: 0, >60: 1
            return self.dataframe["age_binary"].values.astype(np.float32)
        else:
            raise NotImplementedError

    def set_label(self):
        Y = np.where(
            self.dataframe["Diagnosis"] != "Normal",
            1,
            0,
        )

        self.class_nums = 2

        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        diagnosis = item["Diagnosis"]
        if diagnosis == "CAP":
            diagnosis = "Cap"

        if self.path_to_labels is not None:
            img = Image.fromarray(self.tol_images[idx])
        else:
            img = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.path_to_images, f"{diagnosis} Cases", f"{item['Folder']}.nii.gz"))
            ).astype(np.float32)

        # window cut of CT, using common lung window
        img = np.clip(img, self.window[0], self.window[1])
        img = (img - self.window[0]) / (self.window[1] - self.window[0]) * 255
        img = np.uint8(img)
        img = [Image.fromarray(img[i]).convert("RGB") for i in range(img.shape[0])]

        if self.transform is not None:
            img = [self.transform(x) for x in img]
            img = torch.stack(img)

        return img
