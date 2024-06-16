import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# split npz into image and label
# for i in tqdm(range(1, 10001)):
#     data = np.load(f"../../dataset/FairSeg/data_{i:06d}.npz")

#     # print(data["fundus"].shape)
#     # print(data["disc_cup_borders"].min(), data["disc_cup_borders"].max())

#     # plt.imshow(data["disc_cup_borders"])

#     cv2.imwrite(f"../../dataset/FairSeg/image/image_{i:06d}.png", data["fundus"])
#     cv2.imwrite(f"../../dataset/FairSeg/label/label_{i:06d}.png", -1 * data["disc_cup_borders"]*255 / 2)

# modify metadata


df = pd.read_csv("../../dataset/FairSeg/metatabel.csv", index_col=0)

print(df.head())

df["index"] = df.index
df["imagePath"] = df["index"].apply(lambda x: f"image/image_{x+1:06d}.png")
df["labelPath"] = df["index"].apply(lambda x: f"label/label_{x+1:06d}.png")
df["sexBinary"] = df["gender"].apply(lambda x: "M" if x == "Male" else "F")

new_df = df[["index", "imagePath", "labelPath", "sexBinary", "age",
             "race", "gender", "ethnicity", "language", "maritalstatus"]]

new_df.to_csv("../../dataset/FairSeg/metadata.csv", index=False)
