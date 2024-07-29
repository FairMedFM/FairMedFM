import torch
import pickle
import numpy as np
from PIL import Image
import os
import pickle
from datasets.base import BaseDataset


class MIMIC_CXR(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            sa_array = np.asarray(self.dataframe["gender"].values != "M").astype(np.float32)
            class_0_count = np.sum(sa_array == 0)
            class_1_count = np.sum(sa_array == 1)
            total_count_sa = len(sa_array)
            weight_class_0 = total_count_sa / (2 * class_0_count)
            weight_class_1 = total_count_sa / (2 * class_1_count)
            self.class_weights_sa = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
            return sa_array
        elif self.sens_name == "Age":
            # 0-60: 0, >60: 1
            age_binary = self.dataframe["anchor_age"].values.astype(np.int8)
            return np.asarray(age_binary >= 60).astype(np.float32)
        elif self.sens_name == "Race":
            # white: 0, non-white: 1
            return np.asarray(self.dataframe["race_binary"].values != "White").astype(np.float32)
        else:
            raise NotImplementedError

    def set_label(self):
        Y = np.asarray(self.dataframe["No Finding"].values).astype(np.float32)
        self.class_nums = 2

        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        dicom_id, subject_id, study_id = str(item["dicom_id"]), str(item["subject_id"]), str(item["study_id"])

        if self.path_to_labels is not None:
            img = Image.fromarray(self.tol_images[idx])
        else:
            img = Image.open(
                os.path.join(self.path_to_images, f"p{subject_id[:2]}/p{subject_id}/s{study_id}/", f"{dicom_id}.jpg")
            ).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img
