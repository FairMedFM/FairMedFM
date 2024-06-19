import os

import ipdb
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt


# KiTS 2023
def get_bmi(x):
    x = x[6:16]
    bmi = kits_json[kits_json['case_id'] == x]['bmi']

    bmi = bmi.values[0]

    return bmi


kits_metadata = pd.read_csv(
    "~/Documents/NeurIPS2024_FairFM/dataset/KiTS2023/metadata.csv")
kits_json = pd.read_json(
    "~/Documents/NeurIPS2024_FairFM/dataset/KiTS2023/kits23.json")
kits_result_1 = pd.read_csv(
    "~/Documents/NeurIPS2024_FairFM/new_dice/KiTS2023-SAM-bbox-class-1-dice.csv")
kits_result_2 = pd.read_csv(
    "~/Documents/NeurIPS2024_FairFM/new_dice/KiTS2023-SAM-bbox-class-1-dice.csv")
kits_result_3 = pd.read_csv(
    "~/Documents/NeurIPS2024_FairFM/new_dice/KiTS2023-SAM-bbox-class-1-dice.csv")

ic(kits_metadata.head())
ic(len(kits_result_1), len(kits_result_2), len(kits_result_3))
ic(kits_json.keys())

# get bmi
kits_metadata['bmi'] = kits_metadata.imagePath.apply(lambda x: get_bmi(x))
kits_metadata['bmiCategorical'] = kits_metadata.bmi.apply(
    lambda x: 'underweight' if x < 18.5 else 'normal' if x < 25 else 'overweight' if x < 30 else 'obese')
kits_metadata.to_csv(
    "~/Documents/NeurIPS2024_FairFM/dataset/KiTS2023/metadata_refined.csv", index=False)


# plt.figure(figsize=(5, 5))
# kits_metadata.hist(column='bmi')
# plt.savefig('bmi_hist.png')
# ipdb.set_trace()
