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

LABEL_ID = [0, 1, 2, 3, 4, 5, 6, 7, 100, 201, 202, 203, 204, 205, 206, 207]

if __name__ == "__main__":
    new_df = pd.DataFrame(
        columns=["index", "imagePath", "labelPath",  "sex"])

    meta = pd.read_csv("../../dataset/SPIDER/overview.csv")

    for idx in tqdm(range(len(meta))):
        image = sitk.ReadImage(
            f"../../dataset/SPIDER/images/{meta.iloc[idx]['new_file_name']}.mha")
        label = sitk.ReadImage(
            f"../../dataset/SPIDER/masks/{meta.iloc[idx]['new_file_name']}.mha")
        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)

        # normalize image
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)

        # slice-wise split
        for z in range(image.shape[-1]):
            image_slice = image[:, :, z]
            label_slice = label[:, :, z]

            if np.sum(label_slice) > 0:
                image_slice = cv2.resize(
                    image_slice, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                label_slice = cv2.resize(
                    label_slice, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(
                    f"../../dataset/SPIDER/image/{meta.iloc[idx]['new_file_name']}_{z}.png", image_slice)
                cv2.imwrite(
                    f"../../dataset/SPIDER/label/{meta.iloc[idx]['new_file_name']}_{z}.png", label_slice)

                # save metadata
                new_df.loc[len(new_df)] = [len(
                    new_df), f"image/{meta.iloc[idx]['new_file_name']}_{z}.png", f"label/{meta.iloc[idx]['new_file_name']}_{z}.png", meta['sex'][0]]

        # break


new_df["sexBinary"] = new_df["sex"]
new_df.to_csv("../../dataset/SPIDER/metadata.csv", index=False)
