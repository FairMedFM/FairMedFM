# <div align =center><img src=https://raw.githubusercontent.com/FairMedFM/FairMedFM/main/figs/icon.png width=40> FairMedFM
## <div align =center> Fairness Benchmarking for Medical Imaging Foundation Models
![main](https://raw.githubusercontent.com/FairMedFM/FairMedFM/main/figs/main.png)

<p align="center">
  <a href="https://pypi.org/project/fairmedfm/"><img src="https://img.shields.io/pypi/v/fairmedfm.svg" alt="PyPI"></a>
  <a href="https://fairmedfm.github.io/FairMedFM/"><img src="https://img.shields.io/badge/docs-fairmedfm.github.io-teal.svg" alt="Documentation"></a>
  <a href="https://arxiv.org/abs/2407.00983"><img src="https://img.shields.io/badge/arXiv-2407.00983-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/FairMedFM/FairMedFM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg" alt="License"></a>
  <a href="https://pypi.org/project/fairmedfm/"><img src="https://img.shields.io/pypi/pyversions/fairmedfm.svg" alt="Python versions"></a>
  <img src="https://img.shields.io/github/stars/FairMedFM/FairMedFM?style=social" alt="Stars">
  <a href="https://github.com/ubc-tea/MedVLMBench"><img src="https://img.shields.io/badge/Companion-MedVLMBench-orange.svg" alt="MedVLMBench"></a>
</p>

**FairMedFM measures the fairness of any binary classification or segmentation model.** Give it per-sample
predictions and a sensitive attribute (sex, age group, race, site, ...) and it reports subgroup AUC, accuracy and
calibration gaps, equalized odds, and Dice disparities. The metrics come from the FairMedFM benchmark of 20
medical imaging foundation models, but they work for any model and any domain.

## Pip package: fairness metrics for any model

```bash
pip install fairmedfm
```

The package needs only NumPy, pandas and scikit-learn: no PyTorch, no GPU and no FairMedFM checkout. Run
your model in its own environment, save one row per sample, then score it. Results match the code used for the
FairMedFM paper. Full documentation: **[fairmedfm.github.io/FairMedFM](https://fairmedfm.github.io/FairMedFM/)**.

### Classification fairness

Save a CSV with the positive-class probability, the ground-truth label (0/1) and one column per sensitive
attribute:

```text
prob,label,sex,age
0.91,1,F,60+
0.12,0,M,<60
```

```bash
fairmedfm score --task cls --input predictions.csv --sensitive sex age --output fairness.json
```

```python
from fairmedfm import classification_fairness

result = classification_fairness(prob, label, sex)
result["summary"]   # overall-auc, worst-auc, auc-gap, acc-gap, ece-gap, bce-gap, eod, eo, ...
result["groups"]    # the same metrics for each group, with sample counts
```

### Segmentation fairness

Save a CSV with the Dice score of each image or volume and its sensitive attributes:

```text
dice,sex
0.87,F
0.79,M
```

```bash
fairmedfm score --task seg --input dice.csv --sensitive sex
```

```python
from fairmedfm import segmentation_fairness

segmentation_fairness(dice, sex)["summary"]   # mean_dice, min_dice, delta_dice, es_dice, ...
```

### Metrics

| Task | Metric | Definition |
| --- | --- | --- |
| Classification | `overall-auc`, `overall-acc`, `overall-bce`, `overall-ece` | AUC, accuracy, binary cross-entropy and expected calibration error (10 bins) on all samples |
| Classification | `worst-auc` | Lowest group AUC |
| Classification | `auc-gap`, `acc-gap`, `bce-gap`, `ece-gap` | Largest minus smallest group value |
| Classification | `eo` | Equal opportunity gap: largest minus smallest group true positive rate |
| Classification | `eod` | Equalized odds score: `1 - (TPR gap + TNR gap) / 2`; 1 means equal rates |
| Segmentation | `mean_dice` | Mean Dice over all samples |
| Segmentation | `min_dice`, `max_dice`, `delta_dice` | Worst and best group mean Dice, and their difference |
| Segmentation | `std_dice`, `skewness_dice` | Standard deviation of group means; `(1 - min_dice) / (1 - max_dice)` |
| Segmentation | `es_dice` | Equity-scaled Dice: `mean_dice / (1 + std_dice)` |

Accuracy, `eo` and `eod` use the decision threshold with the best overall F1; `result["overall"]` and
`result["groups"]` also report every metric at threshold 0.5. Sensitive attributes can have two or more groups,
and every classification group needs both positive and negative samples. Multi-class classification is not
supported yet: score each class one-vs-rest. To reproduce the benchmark itself (foundation models, datasets,
training), use the repository as described in [Installation](#installation).

## Table of Contents

- [Pip package: fairness metrics for any model](#pip-package-fairness-metrics-for-any-model)
- [Abstract](#abstract)
- [Key Findings](#key-findings)
- [Companion Benchmark: MedVLMBench](#companion-benchmark-medvlmbench)
- [Structure](#structure)
- [Schedule](#schedule)
- [Installation](#installation)
- [Data](#data)
- [Notebook Tutorial](#notebook-tutorial)
- [Running Experiment](#running-experiment)
- [Citation](#citation)

---

## Abstract
The advent of foundation models (FMs) in healthcare offers unprecedented opportunities to enhance medical diagnostics through automated classification and segmentation tasks. However, these models also raise significant concerns about their fairness, especially when applied to diverse and underrepresented populations in healthcare applications. Currently, there is a lack of comprehensive benchmarks, standardized pipelines, and easily adaptable libraries to evaluate and understand the fairness performance of FMs in medical imaging, leading to considerable challenges in formulating and implementing solutions that ensure equitable outcomes across diverse patient populations. To fill this gap, we introduce FairMedFM, a fairness benchmark for FM research in medical imaging. FairMedFM integrates with 17 popular medical imaging datasets, encompassing different modalities, dimensionalities, and sensitive attributes. It explores 20 widely used FMs, with various usages such as zero-shot learning, linear probing, parameter-efficient fine-tuning, and prompting in various downstream tasks -- classification and segmentation. Our exhaustive analysis evaluates the fairness performance over different evaluation metrics from multiple perspectives, revealing the existence of bias, varied utility-fairness trade-offs on different FMs, consistent disparities on the same datasets regardless FMs, and limited effectiveness of existing unfairness mitigation methods. 

## Key Findings

- **Bias is pervasive**: All 20 evaluated medical FMs exhibit measurable fairness disparities across sex, race, and age.
- **Utility–fairness trade-offs vary**: No single FM dominates both accuracy and fairness — the best-performing model is often not the fairest.
- **Dataset effect dominates model choice**: Disparities on the same dataset remain consistent regardless of which FM is used, pointing to data-driven rather than model-driven bias.
- **Mitigation has limited impact**: State-of-the-art debiasing algorithms provide only marginal fairness improvements.

## Companion Benchmark: MedVLMBench

> **Evaluating capability of medical VLMs?** See our companion benchmark [**MedVLMBench**](https://github.com/ubc-tea/MedVLMBench) — the first unified benchmark covering 30+ generalist and specialist VLMs (LLaVA, MedGemma, Qwen2-VL, o3, Gemini 2.5 Pro …) on VQA, diagnosis, and captioning tasks.

FairMedFM and MedVLMBench form a **two-part evaluation suite** for medical foundation models — capability and fairness, measured on the same models and datasets.

| | FairMedFM | [MedVLMBench](https://github.com/ubc-tea/MedVLMBench) |
|---|---|---|
| **Focus** | Fairness across sex, race, age | Capability: accuracy, AUROC, VQA scores |
| **Model paradigm** | Discriminative FMs (CLIP, SAM variants) | Generative VLMs + discriminative models |
| **Tasks** | Classification, Segmentation | VQA, Diagnosis, Captioning |
| **Scale** | 20 FMs · 17 datasets | 30+ VLMs · 14 datasets |

**Models evaluated in both**: BioMedCLIP · MedCLIP · PLIP · SigLIP · MedSigLIP · CLIP · BLIP · BLIP2 · PubMedCLIP

**Datasets in both**: HAM10000 · CheXpert · MIMIC-CXR · FairVLMed10k · GF3300 · PAPILA

## Structure

FairMedFM captures comprehensive modules for benchmarking the fairness of foundation models in medical image analysis.

![main](https://raw.githubusercontent.com/FairMedFM/FairMedFM/main/figs/package.png)

- **Dataloader**: provides a consistent interface for loading and processing imaging data across various modalities and dimensions, supporting both classification and segmentation tasks.
- **Model**: a one-stop library that includes implementations of the most popular pre-trained foundation models for medical image analysis.
- **Usage Wrapper**: encapsulates foundation models for various use cases and tasks, including linear probe, zero-shot inference, PEFT, promptable segmentation, etc.
- **Trainer**: offers a unified workflow for fine-tuning and testing wrapped models, and includes state-of-the-art unfairness mitigation algorithms.
- **Evaluation** includes a set of metrics and tools to visualize and analyze fairness across different tasks.

|        Tasks         | Supported Usages                                        |                       Supported Models                       |                      Supported Datasets                      |
| :------------------: | ------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Image Classification | Linear probe, zero-shot, CLIP adaptaion, PEFT           | CLIP, BLIP, BLIP2, MedCLIP, BiomedCLIP, PubMedCLIP, DINOv2, **DINOv3**, **AIMv2**, RAD-DINO, **RETFound**, C2L, LVM-Med, MedMAE, MoCo-CXR, PLIP, SigLIP, **SigLIP2**, MedSigLIP, **MedGemma**, **UNI2-h**, **Virchow2**, **Prov-GigaPath**, **CONCH**, **Merlin** | CheXpert, MIMIC-CXR, HAM10000, FairVLMed10k, GF3300, PAPILA, BRSET, COVID-CT-MD, ADNI-1.5T |
|  Image Segmentation  | Interactive segmentation prompted with boxes and points | SAM, MobileSAM, TinySAM, MedSAM, MedSAM2, **SAM2**, **SAM3**, **MedicalSAM3 (box)**, SAM-Med2D, FT-SAM | HAM10000, TUSC, FairSeg, Montgomery County X-ray, KiTS, CANDI, IRCADb, SPIDER |

> **Newly integrated foundation models (2023-2026)**
>
> - **General vision**: [DINOv3](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) (Meta), [SAM2](https://github.com/facebookresearch/sam2) (Meta — vanilla checkpoint, shares the MedSAM2 build path; point `--sam_ckpt_path`/`--sam2_model_cfg` at the official SAM2.1 weights/config), [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224) (Google), [AIMv2](https://huggingface.co/apple/aimv2-large-patch14-native) (Apple)
> - **Medical VLM**: [MedGemma](https://huggingface.co/google/medgemma-4b-pt) (Google — feature extraction via its SigLIP-based vision tower)
> - **Pathology foundation models** (new domain for this repo): [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (Mahmood Lab), [Virchow2](https://huggingface.co/paige-ai/Virchow2) (Paige), [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) (Providence), [CONCH](https://huggingface.co/MahmoodLab/conch) (Mahmood Lab, vision-language)
> - **Ophthalmology**: [RETFound](https://github.com/rmaphoh/RETFound_MAE) (Nature, 2023/2024) — pairs naturally with the existing PAPILA/GF3300 eye datasets
> - **3D CT**: [Merlin](https://github.com/StanfordMIMI/Merlin) (Stanford, 2024) — volumetric CT feature extractor for ADNI/COVID-CT-MD/KiTS-style 3D data
>
> **Gated repos** (request access on the model page, then `huggingface-cli login` / set `HF_TOKEN`): DINOv3, MedGemma, UNI2-h, Virchow2, Prov-GigaPath, CONCH, RETFound.
> **Extra install required**: SAM2 needs the [`sam2`](https://github.com/facebookresearch/sam2) package; UNI2-h/Virchow2/Prov-GigaPath need `timm>=1.0`; CONCH needs `pip install git+https://github.com/Mahmoodlab/CONCH.git`; Merlin needs `pip install merlin-vlm`.
> **SAM3 / MedicalSAM3 (2D)**: These use the official [SAM 3 image model](https://github.com/facebookresearch/sam3) API. SAM3 supports the existing point and box prompts; MedicalSAM3 currently supports boxes only. Meta's package requires Python 3.12+, PyTorch 2.7+, and a compatible CUDA setup, so use a separate environment if the main FairMedFM environment is older. Download [Medical SAM3's `checkpoint_2D.pt`](https://huggingface.co/ChongCong/Medical-SAM3/blob/main/checkpoint_2D.pt) separately. Neither checkpoint is part of `pretrained.zip`.
> **RETFound checkpoints**: unlike the other gated models here, RETFound ships raw `.pth` files (not a `transformers`-loadable repo) — download manually after access is granted and point `configs/models/RETFound.json`'s `pretrained_path` at the local file, the same convention used for MedMAE/MoCo-CXR/C2L.
>
> **Not integrated (previously miscredited in this table)**: SAM-Med3D, FastSAM3D, and SegVol were listed here before but had no corresponding code anywhere in the repo. We looked into adding them: SAM-Med3D is loadable via the third-party [`medim`](https://pypi.org/project/medim/) package, and SegVol via `AutoModel.from_pretrained("BAAI/SegVol", trust_remote_code=True)`, but both expose custom, undocumented inference APIs (3D sliding-window prompting, `forward_test()` with joint text/point/box prompts) that don't match this repo's `encode()`/`decode()` segmentation wrapper contract (see `wrappers/sam_model.py`, `wrappers/medsam2.py`). Wiring them in correctly needs a new wrapper/trainer path built against their actual source, not just their README — left as follow-up work rather than shipped half-verified. FastSAM3D additionally has no documented Python inference API at all.



## Schedule

- [x] Release the classification tasks.

- [x] Release the segmentation tasks.
  - [x] 2D dataset + 2D SAMs
  - [x] 3D dataset + 2D SAMs
  - [x] 3D dataset + 3D SAMs

- [x] Release more models 

- [x] Release the preprocessed datasets for classification.

- [x] Release examples and tutorials.

## Evolving
Your are welcome to post your thoughts about updated features and we will try to make this repo evolving as the development of more FMs.

## Installation

To score your own model's predictions, `pip install fairmedfm` is all you need (see
[above](#pip-package-fairness-metrics-for-any-model)). To run the benchmark with the built-in foundation models:

1. Install the benchmark runner (Python 3.10+). Install PyTorch for your CUDA version first if needed.

   ```bash
   pip install "fairmedfm[cls]"   # classification: linear probing, CLIP zero-shot and CLIP adaptation
   pip install "fairmedfm[seg]"   # segmentation with SAM-family models (includes [cls])
   ```

   A few models need packages that are not on PyPI: BLIP/BLIP2 (`salesforce-lavis`, which pins older
   dependencies, so use a separate environment), CONCH (`pip install git+https://github.com/Mahmoodlab/CONCH.git`),
   Merlin (`pip install merlin-vlm`), SAM2 (`sam2`) and SAM3 (`sam3`). The runner reports which package is missing.

2. Work in a directory that holds your data and checkpoints. Dataset and model configs refer to `./data/...` and
   `./pretrained/...`; a `configs/` folder in the working directory overrides the packaged configs.

   ```bash
   wget https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/pretrained.zip
   unzip pretrained.zip && rm -f pretrained.zip
   ```

3. Run an experiment with `fairmedfm run` (examples under [Running Experiment](#running-experiment)).

To use the environment from the paper instead, clone the repository and create the conda environment
(Python 3.8); `python main.py` then runs the same experiments as `fairmedfm run`:

```bash
git clone https://github.com/FairMedFM/FairMedFM.git && cd FairMedFM
conda env create -f environment.yml && conda activate fairmedfm
```

Our notebook tutorials also contains how to setup the environment in Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/linear_probing.ipynb)

## Data

You can either download our pre-processed data directly (see [next section](#use-our-pre-processed-data)) or pre-process customized data your self. However, not all dataset we used permit us to release the data on our end (e.g., dataset like MIMIC and ADNI requires the user go through their data usage application first). In such case, we cannot provide the download link of our preprocessed dataset for them, but we have the original dataset downloading link and our pre-process scripts released.

### Preprocess data on your own
We provide data preprocessing scripts for each datasets [here](./notebooks/preprocess). The data preprocessing contains 3 steps:

- (Optional) preprocess imaging data.
- Preprocess metadata and sensitive attributes.
- Split dataset into training set and test set with balanced subgroups (for classification only).

Our data is downloaded uisng the following links.

#### Classification Dataset

| Dataset         | Link                                                                                               |
|-----------------|------------------------------------------------------------------------------------------------------|
| **CheXpert**    | [Original data](https://stanfordmlgroup.github.io/competitions/chexpert/) <br> [Demographic data](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) |
| **MIMIC-CXR**   | [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)                                      |
| **PAPILA**      | [PAPILA](https://www.nature.com/articles/s41597-022-01388-1#Sec6)                                    |
| **HAM10000**    | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)          |
| **OCT**         | [OCT](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm)                                    |
| **OL3I**        | [OL3I](https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc)         |
| **COVID-CT-MD** | [COVID-CT-MD](https://doi.org/10.6084/m9.figshare.12991592)                                          |
| **ADNI**   | [ADNI-1.5T](https://ida.loni.usc.edu/login.jsp?project=ADNI)                                         |

#### Segmentation Dataset

| Dataset         | Link                                                                                               |
|-----------------|------------------------------------------------------------------------------------------------------|
| **HAM10000**    | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)|
| **TUSC**   | [TUSC](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5)                                      |
| **FairSeg**      | [FairSeg](https://ophai.hms.harvard.edu/datasets/harvard-fairseg10k)                                    |
| **Montgomery County X-ray**    | [Montgomery County X-ray](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html)          |
| **KiTS2023**         | [KiTS2023](https://kits-challenge.org/kits23/)                                    |
| **IRCADb**        | [IRCADb](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)         |
| **CANDI** | [CANDI](https://www.nitrc.org/projects/candi_share)                                          |
| **SPIDER**   | [SPIDER](http://spider.grand-challenge.org)|


### Use Our Pre-processed Data
We offer data downloading through the S3 link. We are working to build this feature now.
#### Classification Dataset

| Dataset         | Link                                                                                               |
|-----------------|------------------------------------------------------------------------------------------------------|
| **CheXpert**    | Requires application on original data provider. |
| **MIMIC-CXR**   | Requires application on original data provider.                   |
| **PAPILA**      | [PAPILA](https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/PAPILA.zip)                            |
| **HAM10000**    | [HAM10000](https://object-arbutus.alliancecan.ca/swift/v1/86581f3bb67c4c04bbccbcb839de730a/rjin/HAM10000.zip)         |
| **OCT**         | Waiting for more storage resources                   |
| **OL3I**        | Waiting for more storage resources         |
| **COVID-CT-MD** | Waiting for more storage resources                |
| **ADNI**   | Requires application on original data provider.

## Notebook Tutorial
We offer some examples of how to use our package through the notebook.

| Feature | Notebook  |
|-----------------|------------------------------------------------------------------------------------------------------|
| **Linear Probing**    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/linear_probing.ipynb) |
| **CLIP Zero-shot and Adaptor**   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/clip_downstream.ipynb) |
| **Segmentation**   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/segmentation.ipynb) |
<!-- | **More Coming Soon**   | TODO | -->

## Running Experiment

### Classification

We provide an example of running a linear-probe (classification) experiment of the CLIP model on the MIMIC-CXR dataset to evaluate fairness on sex. Run `fairmedfm run --help` or see [parse_args.py](https://github.com/FairMedFM/FairMedFM/blob/main/src/fairmedfm/parse_args.py) for all options. In a source checkout, `python main.py` accepts the same arguments.

```bash
fairmedfm run --task cls --usage lp --dataset CXP --sensitive_name Sex --method erm --total_epochs 100 --warmup_epochs 5 --blr 2.5e-4 --batch_size 128 --optimizer adamw --min_lr 1e-5 --weight_decay 0.05
```

### Segmentation (2D SAMs)

We also provide an example of using SAM with center point prompt on the TUSC dataset to evaluate fairness on sex.
Run `fairmedfm run --help` or see [parse_args.py](https://github.com/FairMedFM/FairMedFM/blob/main/src/fairmedfm/parse_args.py) for all options. In a source checkout, `python main.py` accepts the same arguments.

```bash
fairmedfm run --task seg --usage seg2d --dataset TUSC --sensitive_name Sex --method erm --batch_size 1 --pos_class 255 --model SAM --sam_ckpt_path ./weights/SAM.pth --img_size 1024 --prompt center
```

For the newer 2D models, install the official `sam3` package in a compatible environment with `pip install git+https://github.com/facebookresearch/sam3.git`, then run one of:

```bash
# SAM3: downloads Meta's gated checkpoint after Hugging Face access is granted.
fairmedfm run --task seg --usage seg2d --dataset TUSC --sensitive_name Sex --method erm --batch_size 1 --pos_class 255 --model SAM3 --img_size 1024 --prompt center

# MedicalSAM3: use the 2D Medical SAM3 checkpoint (not its 3D V2 checkpoint).
fairmedfm run --task seg --usage seg2d --dataset TUSC --sensitive_name Sex --method erm --batch_size 1 --pos_class 255 --model MedicalSAM3 --sam_ckpt_path /path/to/checkpoint_2D.pt --img_size 1024 --prompt bbox
```

The existing segmentation trainer is single-image only (`--batch_size 1`). Box and point prompts are derived from the ground-truth mask, so results are interactive segmentation scores, not unprompted segmentation scores. The new adapters restore the dataset's normalized BGR tensors to RGB pixels before SAM3 preprocessing.

MedicalSAM3 keeps detections above confidence 0.1 and uses the highest-scoring mask; an empty detection produces an empty mask. Run `python -m pytest -q tests/test_sam3.py` for checkpoint-free regression checks. These cover image conversion, prompt geometry, mask selection, empty detections, and checkpoint loading; actual checkpoint inference still needs validation in a SAM3-compatible environment.


## Acknowledgement

We thank [MEDFAIR](https://github.com/ys-zong/MEDFAIR) for their pioneering works on benchmarking fairness for medical image analysis, and [Slide-SAM](https://github.com/Curli-quan/Slide-SAM) for the SAM inference framework.

## License

This project is released under the CC BY 4.0 license. Please see the LICENSE file for more information.

## Citation
If you think our project is helpful and love our project, it's nice if you can cite us. Such supports will help us secure resources for further developing similar projects.

```bibtex
@article{jin2024fairmedfm,
  title={FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models},
  author={Jin, Ruinan and Xu, Zikang and Zhong, Yuan and Yao, Qiongsong and Dou, Qi and Zhou, S Kevin and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2407.00983},
  year={2024}
}
```

If you also use our companion benchmark **MedVLMBench** for capability evaluation, please cite:

```bibtex
@article{zhong2025can,
  title={Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights},
  author={Zhong, Yuan and Jin, Ruinan and Li, Xiaoxiao and Dou, Qi},
  journal={arXiv preprint arXiv:2506.17337},
  year={2025}
}
```
