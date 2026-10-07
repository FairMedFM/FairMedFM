---
title: Medical imaging datasets in the FairMedFM fairness benchmark
description: Datasets used by FairMedFM - CheXpert, MIMIC-CXR, HAM10000, PAPILA, FairVLMed10k, Harvard-GF, BRSET, COVID-CT-MD, ADNI and segmentation datasets - with sensitive attributes, access and preprocessing.
---

# Datasets

FairMedFM evaluates fairness on medical imaging datasets that record patient attributes. Several datasets
require an application to their provider, so FairMedFM cannot redistribute them; the repository provides the
preprocessing scripts instead.

## Classification

| `--dataset` | Dataset | Modality | Sensitive attributes | Access |
| --- | --- | --- | --- | --- |
| `CXP` | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) ([demographics](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf)) | Chest X-ray | Sex, Age, Race | Application |
| `MIMIC_CXR` | [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | Chest X-ray | Sex, Age, Race | Credentialed (PhysioNet) |
| `HAM10000` | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | Dermoscopy | Sex, Age | [Preprocessed download](https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/HAM10000.zip) |
| `PAPILA` | [PAPILA](https://www.nature.com/articles/s41597-022-01388-1) | Fundus | Sex, Age | [Preprocessed download](https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/PAPILA.zip) |
| `FairVLMed10k` | FairVLMed10k | Fundus (SLO) | Sex, Age, Race, Language | Provider |
| `GF3300` | Harvard-GF (GF3300) | Fundus | Sex, Age, Race, Language | Provider |
| `BREST` | BRSET | Fundus | Sex, Age | Provider |
| `COVID_CT_MD` | [COVID-CT-MD](https://doi.org/10.6084/m9.figshare.12991592) | CT | Sex, Age | Public |
| `ADNI` | [ADNI 1.5T](https://ida.loni.usc.edu/login.jsp?project=ADNI) | Brain MRI | Sex | Application |

Use `--sensitive_name` with one of the dataset's attributes (`Sex`, `Age`, `Race` or `Language`).

## Segmentation

| `--dataset` | Dataset | Modality |
| --- | --- | --- |
| `TUSC` | [TUSC](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5) | Thyroid ultrasound |
| `HAM10000-Seg` | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | Dermoscopy |
| `FairSeg` | [Harvard-FairSeg](https://ophai.hms.harvard.edu/datasets/harvard-fairseg10k) | Fundus (SLO) |
| `montgomery` | [Montgomery County X-ray](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html) | Chest X-ray |

TUSC has a packaged config and records sex; the other segmentation datasets need a config in `./configs/datasets/` (see
[Run experiments](benchmark.md#current-limitations)). The paper also covers 3D segmentation datasets (KiTS2023,
IRCADb, CANDI and SPIDER), whose preprocessing scripts are in the repository.

## Preparing data

1. Download the data from the provider (or the preprocessed archive, where available) into `data/<dataset>/` in
   your working directory.
2. Run the dataset's preprocessing notebook from
   [`pre-processing/`](https://github.com/FairMedFM/FairMedFM/tree/main/pre-processing). The notebooks
   preprocess images where needed, extract the sensitive attributes, and for classification create train/test
   splits with balanced subgroups.
3. Check that the paths in the dataset config match your files, or override the config in
   `./configs/datasets/`.

## Using your own dataset

To measure fairness on your own data with your own model, you do not need a dataset config: export predictions
and use [`fairmedfm score`](evaluate-your-model.md).
