import torch
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import SimpleITK as sitk

from datasets.base import BaseDataset


class ADNI(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            sa_array = np.asarray(self.dataframe["Sex"].values != "M").astype(np.float32)
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
            self.dataframe["Group"] != "CN",
            1,
            0,
        )

        self.class_nums = 2

        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        if self.path_to_labels is not None:
            raise NotImplementedError
        else:
            img = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.path_to_images, f"{item['Image Data ID']}.nii.gz"))
            ).astype(np.float32)

        img_lst = []
        for i in range(img.shape[0]):
            slc = (img[i] - img[i].min()) / (img[i].max() - img[i].min() + 1e-5) * 255
            img_lst.append(Image.fromarray(np.uint8(slc)).convert("RGB"))

        if self.transform is not None:
            img_lst = [self.transform(x) for x in img_lst]

        img = torch.stack(img_lst)

        return img
