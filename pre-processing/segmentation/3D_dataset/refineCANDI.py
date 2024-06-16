import os

import ipdb
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt

metadata = pd.read_csv(
    "~/Documents/NeurIPS2024_FairFM/dataset/CANDI/metadata.csv")

BPDwithoutPsy_Basic_Demographics = pd.read_csv(
    f"../../dataset/CANDI/raw/BPDwithoutPsy_Basic_Demographics.csv")
BPDwithPsy_Basic_Demographics = pd.read_csv(
    f"../../dataset/CANDI/raw/BPDwithPsy_Basic_Demographics.csv")
HC_Basic_Demographics = pd.read_csv(
    f"../../dataset/CANDI/raw/HC_Basic_Demographics.csv")
SS_Basic_Demographics = pd.read_csv(
    f"../../dataset/CANDI/raw/SS_Basic_Demographics.csv")

# ic(metadata.head())
metadata['subject'] = metadata['imagePath'].apply(
    lambda x: '_'.join(x[6:].split('_')[:2]))
ic(metadata.head())

# ic(BPDwithoutPsy_Basic_Demographics.head())
# ic(BPDwithPsy_Basic_Demographics.head())
# ic(HC_Basic_Demographics.head())
# ic(SS_Basic_Demographics.head())

# merge
demographics = pd.concat(
    [BPDwithoutPsy_Basic_Demographics, BPDwithPsy_Basic_Demographics, HC_Basic_Demographics, SS_Basic_Demographics])

ic(demographics.head())

metadata['handedness'] = metadata['subject'].apply(
    lambda x: demographics[demographics['Subject'] == x]['Handedness'].values[0])

ic(metadata)
metadata.to_csv(
    "~/Documents/NeurIPS2024_FairFM/dataset/CANDI/metadata-refined.csv")
