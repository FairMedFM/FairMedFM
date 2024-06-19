import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import time
from PIL import Image
from tqdm import tqdm

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# read metadata
path = "/media/yesindeed/WD5T/data/mimic-cxr-jpg/2.0.0/"

df = pd.read_csv(os.path.join(path, "mimic-cxr-2.0.0-metadata.csv"))


# resize all images to save storage
low_res_path = os.path.join(path, "small")

for i in tqdm(range(len(df))):
    item = df.iloc[i]
    dicom_id, subject_id, study_id = str(item["dicom_id"]), str(item["subject_id"]), str(item["study_id"])
    sub_folder = f"p{subject_id[:2]}"

    full_path = f"{sub_folder}/p{subject_id}/s{study_id}/"

    save_folder = os.path.join(low_res_path, full_path)
    if os.path.exists(os.path.join(save_folder, f"{dicom_id}.jpg")):
        continue

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img = Image.open(os.path.join(path, "files", full_path, f"{dicom_id}.jpg"))
    img = img.resize(size=(256, 256), resample=Image.BICUBIC)

    img.save(os.path.join(save_folder, f"{dicom_id}.jpg"))
