import pandas as pd

df = pd.read_csv("../../dataset/ISIC2018/HAM10000_metadata.csv")

print(df)
df.dropna(subset=["sex"], inplace=True)

df["imagePath"] = df["image_id"].apply(lambda x: "image/" + x + ".jpg")
df["labelPath"] = df["image_id"].apply(lambda x: "label/" + x + "_segmentation.png")
df["index"] = df.index
df["sexBinary"] = df["sex"].apply(lambda x: "F" if x == "female" else "M")

new_df = df[['index', 'imagePath', 'labelPath', 'sexBinary', 'age']]
new_df.to_csv("../../dataset/ISIC2018/metadata.csv", index=False)