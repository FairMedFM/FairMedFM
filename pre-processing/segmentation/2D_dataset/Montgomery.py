import os
import shutil

import cv2
import pandas as pd


# concat left and right masks
def concat_mask(msk0, msk1):
    msk = msk0 / 255 * 128 + msk1
    return msk


new_df = pd.DataFrame(
    columns=["index", "imagePath", "labelPath",  "sex", "age"])

filelist = os.listdir("../../dataset/montgomery/MontgomerySet/CXR_png/")

index = 0

for f in filelist:
    if ".png" in f:
        # copy image
        shutil.copyfile(os.path.join("../../dataset/montgomery/MontgomerySet/CXR_png/", f),
                        os.path.join("../../dataset/montgomery/image/", f))

        # merge mask
        msk0 = cv2.imread(os.path.join(
            "../../dataset/montgomery/MontgomerySet/ManualMask/leftMask/", f), cv2.IMREAD_GRAYSCALE)
        msk1 = cv2.imread(os.path.join(
            "../../dataset/montgomery/MontgomerySet/ManualMask/rightMask/", f), cv2.IMREAD_GRAYSCALE)

        msk = concat_mask(msk0, msk1)
        cv2.imwrite(os.path.join("../../dataset/montgomery/label/", f), msk)

        # extract metadata
        metafile = os.path.join(
            "../../dataset/montgomery/MontgomerySet/ClinicalReadings", f.replace(".png", ".txt"))

        with open(metafile, "r") as file:
            lines = file.readlines()
        print(lines)
        sex = lines[0][-3]
        age = int(lines[1][-6:-2])

        print(sex, age)

        new_df.loc[index] = [index, "image/" + f, "label/" + f, sex, age]
        index += 1

new_df.to_csv("../../dataset/montgomery/metadata.csv", index=False)
