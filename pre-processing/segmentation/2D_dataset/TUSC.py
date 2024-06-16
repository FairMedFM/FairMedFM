import os

import pandas as pd

df = pd.read_csv("../../dataset/TUSC/meta.csv")
new_df = pd.DataFrame(
    columns=["index", "imagePath", "labelPath",  "sex", "age"])

print(df.head())

filelist = os.listdir("../../dataset/TUSC/image/")

print(filelist)
index = 0

for f in filelist:
    meta = df[df["annot_id"] == (f.split("_")[0] + '_')]
    new_df.loc[index] = [index, "image/" + f,
                         "label/" + f, meta["sex"].values[0], meta["age"].values[0]]

    index += 1

print(new_df)

new_df["sexBinary"] = new_df["sex"].apply(
    lambda x: "F" if x == "Female" else "M")
print(new_df)

new_df.to_csv("../../dataset/TUSC/metadata.csv", index=False)
