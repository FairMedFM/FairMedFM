"""
    Dataset preprocessing for CANDI Dataset
"""
import os

import cv2
import ipdb
import numpy as np
import pandas as pd
from einops import rearrange, repeat
from icecream import ic
from monai import transforms as mtrans
from tqdm import tqdm

LABEL_ID = [0.,  2.,  3.,  4.,  5.,  7.,  8., 10., 11., 12., 13., 14., 15., 16.,
            17., 18., 24., 26., 28., 30., 41., 42., 43., 44., 46., 47., 49., 50.,
            51., 52., 53., 54., 58., 60., 62.]

'''
2    左大脑白质 *
3    左大脑皮层 *
4    左侧脑室   *
5    左下侧脑室 
7    左小脑白质 
8    左小脑皮层 
10   左丘脑 
11   左尾状核 
12   左壳核 
13   左苍白球 
14   第三脑室 
15   第四脑室 
16   脑干 *
17   左海马 
18   左杏仁核 
24   脑脊液 
26   左伏隔区 
28   左腹侧丘脑 
29   左未确定区域 
30   左血管 
41   右大脑白质 *
42   右大脑皮层 *
43   右侧脑室   *
44   右下侧脑室 
46   右小脑白质 
47   右小脑皮层 
49   右丘脑 
50   右尾状核 
51   右壳核 
52   右苍白球 
53   右海马 
54   右杏仁核 
58   右伏隔区 
60   右腹侧丘脑 
61   右未确定区域 
62   右血管 
72   第五脑室 
77   白质低密度区 
85   视交叉
'''


def process_one_volume(image, label, filename):

    sample_dict = {
        "image": image,
        "label": label
    }

    trans = mtrans.Compose([
        # (256, 256, 128), (256, 256, 128)
        mtrans.LoadImaged(keys=["image", "label"]),
    ])
    sample_dict = trans(sample_dict)

    volume_shape = sample_dict["image"].shape
    # ic(volume_shape)
    sample_dict["image"] = (sample_dict["image"] - sample_dict["image"].min()) / \
        (sample_dict["image"].max() - sample_dict["image"].min())

    available_slices = []
    for i in range(volume_shape[-1]):
        image = sample_dict["image"][:, i].numpy() * 255
        label = sample_dict["label"][:, i].numpy()

        # ic(image.shape)
        if np.sum(label) > 0:
            available_slices.append(i)
            # ic(image.shape, label.shape)
            image = repeat(image, "w h -> w h c", c=3)
            image = cv2.resize(image, (1024, 1024),
                               interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (1024, 1024),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"../../dataset/CANDI/image/{filename}_{i}.png", image)
            cv2.imwrite(f"../../dataset/CANDI/label/{filename}_{i}.png", label)

    return available_slices


if __name__ == "__main__":
    if not os.path.exists("../../dataset/CANDI/image"):
        os.makedirs("../../dataset/CANDI/image")
    if not os.path.exists("../../dataset/CANDI/label"):
        os.makedirs("../../dataset/CANDI/label")

    new_df = pd.DataFrame(
        columns=["index", "imagePath", "labelPath",  "sex", "age"])
    for subset in ["BPDwithoutPsy_Basic_Demographics", "BPDwithPsy_Basic_Demographics", "HC_Basic_Demographics", "SS_Basic_Demographics"]:
        ic(subset)
        sample_meta = pd.read_csv(
            f"../../dataset/CANDI/raw/{subset}.csv")

        for idx in tqdm(range(len(sample_meta))):
            info = sample_meta.loc[idx]

            image = os.path.join(
                "../../dataset/CANDI/raw/", info["Subject"], info["Subject"] + "_procimg.nii.gz")
            label = os.path.join("../../dataset/CANDI/raw/",
                                 info["Subject"], info["Subject"] + "_seg.nii.gz")

            slice_idx = process_one_volume(image, label, info["Subject"])
            # ic(slice_idx)
            for slice_id in slice_idx:
                # ic([len(new_df), f"image/{info['Subject']}_{slice_id}.png",
                #     f"label/{info['Subject']}_{slice_id}.png", info["Gender"], info["Age"]])
                new_df.loc[len(new_df)] = [len(new_df), f"image/{info['Subject']}_{slice_id}.png",
                                           f"label/{info['Subject']}_{slice_id}.png", info["Gender"], info["Age"]]

    new_df["sexBinary"] = new_df["sex"].apply(
        lambda x: "F" if x == "female" else "M")

    new_df.to_csv("../../dataset/CANDI/metadata.csv", index=False)
