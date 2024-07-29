import torch
import numpy as np
import pickle

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_labels=None):
        super().__init__()

        # path_to_labels: reserve for seg

        assert path_to_images is not None, "path of raw images can not be none"

        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.sens_name = sens_name

        self.A = self.set_sensitive()
        self.Y = self.set_label()
        self.compute_class_weights()
        self.AY_proportion = None

    def __len__(self):
        return self.dataset_size

    def set_sensitive(self):
        pass

    def set_label(self):
        pass

    def get_sensitive(self, idx):
        return int(self.A[idx])

    def get_label(self, idx):
        return torch.FloatTensor([int(self.Y[idx])])

    def get_img(self, idx):
        pass

    def __getitem__(self, idx):
        return {
            "img": self.get_img(idx),
            "label": self.get_label(idx),
            "sensitive": self.get_sensitive(idx),
            "idx": idx,
        }

    def get_AY_proportions(self):
        if self.AY_proportion:
            return self.AY_proportion

        A = self.A.tolist()
        Y = self.Y.tolist()
        ttl = len(A)

        len_A0Y0 = len([ay for ay in zip(A, Y) if ay == (0, 0)])
        len_A0Y1 = len([ay for ay in zip(A, Y) if ay == (0, 1)])
        len_A1Y0 = len([ay for ay in zip(A, Y) if ay == (1, 0)])
        len_A1Y1 = len([ay for ay in zip(A, Y) if ay == (1, 1)])

        assert (len_A0Y0 + len_A0Y1 + len_A1Y0 + len_A1Y1) == ttl, "Problem computing train set AY proportion."
        A0Y0 = len_A0Y0 / ttl
        A0Y1 = len_A0Y1 / ttl
        A1Y0 = len_A1Y0 / ttl
        A1Y1 = len_A1Y1 / ttl

        self.AY_proportion = [[A0Y0, A0Y1], [A1Y0, A1Y1]]

        return self.AY_proportion

    def get_A_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[0][1], AY[1][0] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def get_Y_proportions(self):
        AY = self.get_AY_proportions()
        ret = [AY[0][0] + AY[1][0], AY[0][1] + AY[1][1]]
        np.testing.assert_almost_equal(np.sum(ret), 1.0)
        return ret

    def compute_class_weights(self):
        class_0_count = np.sum(self.Y == 0)
        class_1_count = np.sum(self.Y == 1)
        total_count_y = len(self.Y)
        weight_class_0 = total_count_y / (2 * class_0_count)
        weight_class_1 = total_count_y / (2 * class_1_count)
        self.class_weights_y = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
        return self.class_weights_y


class BaseSegDataset(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_labels=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_labels)
