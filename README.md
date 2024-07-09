# <div align =center><img src=./figs/icon.png width=40> FairMedFM
## <div align =center> Fairness Benchmarking for Medical Imaging Foundation Models
![main](./figs/main.png)

## Abstract
The advent of foundation models (FMs) in healthcare offers unprecedented opportunities to enhance medical diagnostics through automated classification and segmentation tasks. However, these models also raise significant concerns about their fairness, especially when applied to diverse and underrepresented populations in healthcare applications. Currently, there is a lack of comprehensive benchmarks, standardized pipelines, and easily adaptable libraries to evaluate and understand the fairness performance of FMs in medical imaging, leading to considerable challenges in formulating and implementing solutions that ensure equitable outcomes across diverse patient populations. To fill this gap, we introduce FairMedFM, a fairness benchmark for FM research in medical imaging. FairMedFM integrates with 17 popular medical imaging datasets, encompassing different modalities, dimensionalities, and sensitive attributes. It explores 20 widely used FMs, with various usages such as zero-shot learning, linear probing, parameter-efficient fine-tuning, and prompting in various downstream tasks -- classification and segmentation. Our exhaustive analysis evaluates the fairness performance over different evaluation metrics from multiple perspectives, revealing the existence of bias, varied utility-fairness trade-offs on different FMs, consistent disparities on the same datasets regardless FMs, and limited effectiveness of existing unfairness mitigation methods. 

## Structure

FairMedFM captures comprehensive modules for benchmarking the fairness of foundation models in medical image analysis.

![main](./figs/package.png)

- **Dataloader**: provides a consistent interface for loading and processing imaging data across various modalities and dimensions, supporting both classification and segmentation tasks.
- **Model**: a one-stop library that includes implementations of the most popular pre-trained foundation models for medical image analysis.
- **Usage Wrapper**: encapsulates foundation models for various use cases and tasks, including linear probe, zero-shot inference, PEFT, promptable segmentation, etc.
- **Trainer**: offers a unified workflow for fine-tuning and testing wrapped models, and includes state-of-the-art unfairness mitigation algorithms.
- **Evaluation** includes a set of metrics and tools to visualize and analyze fairness across different tasks.

|        Tasks         | Supported Usages                                        |                       Supported Models                       |                      Supported Datasets                      |
| :------------------: | ------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Image Classification | Linear probe, zero-shot, CLIP adaptaion, PEFT           | CLIP, BLIP, BLIP2, MedCLIP, BiomedCLIP, PubMedCLIP, DINOv2, C2L, LVM-Med, MedMAE, MoCo-CXR | CheXpert, MIMIC-CXR, HAM10000, FairVLMed10k, GF3300, PAPILA, BRSET, COVID-CT-MD, ADNI-1.5T |
|  Image Segmentation  | Interactive segmentation prompted with boxes and points | SAM, MobileSAM, TinySAM, MedSAM, SAM-Med2D, FT-SAM, SAM-Med3D, FastSAM3D, SegVol | HAM10000, TUSC, FairSeg, Montgomery County X-ray, KiTS, CANDI, IRCADb, SPIDER |



## Schedule

- [x] Release the classification tasks.

- [ ] Release the segmentation tasks.
  - [x] 2D dataset + 2D SAMs
  - [x] 3D dataset + 2D SAMs
  - [ ] 3D dataset + 3D SAMs

- [ ] Release more models 

- [ ] Release the preprocessed datasets.

- [ ] Release examples and tutorials.

## Installation

1. Download from github

   ```git
   git clone https://github.com/FairMedFM/FairMedFM.git
   cd FairMedFM
   ```

2. Creating conda environment

   ```
   conda env create -f environment.yaml
   conda activate fairmedfm
   ```

## Getting Started

### Data Preprocessing

We provide data preprocessing scripts for each datasets [here](./notebooks/preprocess). The data preprocessing contains 3 steps:

- (Optional) preprocess imaging data.
- Preprocess metadata and sensitive attributes.
- Split dataset into training set and test set with balanced subgroups.

### Running Experiment

We provide an example of running a linear-probe (classification) experiment of the CLIP model on the MIMIC-CXR dataset to evaluate fairness on sex. Please refer to [parse_args.py](./parse_args.py) for more details.

```bash
python main.py --task cls --usage lp --dataset CXP --sensitive_name Sex --method erm --total_epochs 100 --warmup_epochs 5 --blr 2.5e-4 --batch_size 128 --optimizer adamw --min_lr 1e-5 --weight_decay 0.05
```

## Acknowledgement

We thank [MEDFAIR](https://github.com/ys-zong/MEDFAIR) for their pioneering works on benchmarking fairness for medical image analysis, and [Slide-SAM](https://github.com/Curli-quan/Slide-SAM) for the SAM inference framework.

## License

This project is released under the CC BY 4.0 license. Please see the LICENSE file for more information.
