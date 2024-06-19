"""
    Dataset preprocessing for KiTS2023 Dataset
"""
import json
import os

import cv2
import ipdb
import numpy as np
import pandas as pd
import SimpleITK as sitk
from einops import rearrange, repeat
from icecream import ic
from monai import transforms as mtrans
from tqdm import tqdm

LABEL_ID = [0,   1,  17,  33,  65,  97, 129, 193,
            257, 321, 385, 449, 451, 453, 465, 481, 513]


def relabel_mask(arr):
    arr[arr == 257] = 2
    arr[arr == 321] = 3
    arr[arr == 385] = 4
    arr[arr == 449] = 5
    arr[arr == 451] = 6
    arr[arr == 453] = 7
    arr[arr == 465] = 8
    arr[arr == 481] = 9
    arr[arr == 513] = 10

    return arr


if __name__ == "__main__":
    new_df = pd.DataFrame(
        columns=["index", "imagePath", "labelPath", "sex"])

    meta = pd.read_csv("../../dataset/IRCADb/overview.csv")

    for idx in tqdm(range(len(meta))):
        image = sitk.ReadImage(
            f"../../dataset/IRCADb/Dataset003_3D-IRCADb/imagesTr/{meta.iloc[idx]['name']}.nii.gz")
        label = sitk.ReadImage(
            f"../../dataset/IRCADb/Dataset003_3D-IRCADb/labelsTr/{meta.iloc[idx]['name']}.nii.gz")
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        label = relabel_mask(label)

        # normalize image
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)

        # slice-wise split
        for z in range(image.shape[0]):
            image_slice = image[z]
            label_slice = label[z]

            # ic(image_slice.shape)

            if np.sum(label_slice) > 0:
                image_slice = cv2.resize(
                    image_slice, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                label_slice = cv2.resize(
                    label_slice, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(
                    f"../../dataset/IRCADb/image/{meta.iloc[idx]['name']}_{z}.png", image_slice)
                cv2.imwrite(
                    f"../../dataset/IRCADb/label/{meta.iloc[idx]['name']}_{z}.png", label_slice)

                # save metadata
                new_df.loc[len(new_df)] = [len(
                    new_df), f"image/{meta.iloc[idx]['name']}_{z}.png", f"label/{meta.iloc[idx]['name']}_{z}.png", meta.iloc[idx]['sex']]

        # break


new_df["sexBinary"] = new_df["sex"]
new_df.to_csv("../../dataset/IRCADb/metadata.csv", index=False)
