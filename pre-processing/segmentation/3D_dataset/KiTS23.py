"""
    Dataset preprocessing for KiTS2023 Dataset
"""
import json
import os

import cv2
import ipdb
import numpy as np
import pandas as pd
from einops import rearrange, repeat
from icecream import ic
from monai import transforms as mtrans
from tqdm import tqdm

LABEL_ID = [0, 1, 2, 3]
"""
0 for background, 1 for kidney, 2 for tumor, and 3 for cyst
"""


def process_one_volume(image, label, filename):
    sample_dict = {
        "image": image,
        "label": label
    }

    trans = mtrans.Compose([
        # (612, 512, 512)
        mtrans.LoadImaged(keys=["image", "label"]),
        mtrans.EnsureChannelFirstd(keys=["image", "label"]),
        mtrans.Orientationd(keys=["image", "label"], axcodes="RAS"),
        mtrans.Spacingd(keys=["image", "label"], pixdim=(
            1.0, 1.0, 1.0), mode="nearest"),
    ])
    sample_dict = trans(sample_dict)

    volume_shape = sample_dict["image"].shape  # (1, w, h, z)
    # ic(volume_shape, sample_dict["label"].shape)
    sample_dict["image"] = (sample_dict["image"] - sample_dict["image"].min()) / \
        (sample_dict["image"].max() - sample_dict["image"].min())

    available_slices = []
    for i in range(volume_shape[-1]):
        image = sample_dict["image"][0][:, :, i].numpy() * 255
        label = sample_dict["label"][0][:, :, i].numpy()
        if np.sum(label) > 0:
            available_slices.append(i)
            # ic(image.shape, label.shape)
            image = repeat(image, "w h -> w h c", c=3)
            image = cv2.resize(image, (1024, 1024),
                               interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (1024, 1024),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                f"../../dataset/KiTS2023/image/{filename}_{i}.png", image)
            cv2.imwrite(
                f"../../dataset/KiTS2023/label/{filename}_{i}.png", label)

    return available_slices


if __name__ == "__main__":
    new_df = pd.DataFrame(
        columns=["index", "imagePath", "labelPath",  "sex", "age"])

    with open("../../dataset/KiTS2023/kits23.json") as f:
        meta = json.load(f)

    for idx in tqdm(range(300)):
        image = os.path.join(
            "../../dataset/KiTS2023/imagesTr", f"case_{idx:05d}_0000.nii.gz")
        label = os.path.join(
            "../../dataset/KiTS2023/labelsTr", f"case_{idx:05d}.nii.gz")

        slice_idx = process_one_volume(image, label, f"case_{idx:05d}")
        # ic(slice_idx)

        for slice_id in slice_idx:
            new_df.loc[len(new_df)] = [len(new_df), f"image/case_{idx:05d}_{slice_id}.png",
                                       f"label/case_{idx:05d}_{slice_id}.png", meta[idx]["gender"], meta[idx]["age_at_nephrectomy"]]

    for idx in tqdm(range(400, 588)):
        image = os.path.join(
            "../../dataset/KiTS2023/imagesTr", f"case_{idx:05d}_0000.nii.gz")
        label = os.path.join(
            "../../dataset/KiTS2023/labelsTr", f"case_{idx:05d}.nii.gz")

        slice_idx = process_one_volume(image, label, f"case_{idx:05d}")
        # ic(slice_idx)

        for slice_id in slice_idx:
            new_df.loc[len(new_df)] = [len(new_df), f"image/case_{idx:05d}_{slice_id}.png",
                                       f"label/case_{idx:05d}_{slice_id}.png", meta[idx-100]["gender"], meta[idx-100]["age_at_nephrectomy"]]
new_df["sexBinary"] = new_df["sex"].apply(
    lambda x: "F" if x == "female" else "M")

new_df.to_csv("../../dataset/KiTS2023/metadata.csv", index=False)
