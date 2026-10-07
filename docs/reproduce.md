---
title: Reproduce the FairMedFM paper
description: Reproduce the FairMedFM paper experiments with the original Python 3.8 conda environment, the preprocessing notebooks and the Colab tutorials.
---

# Reproduce the paper

The FairMedFM paper's experiments ran in the conda environment from the repository (Python 3.8, PyTorch 2.3,
transformers 4.40). Use it to reproduce the published numbers:

```bash
git clone https://github.com/FairMedFM/FairMedFM.git
cd FairMedFM
conda env create -f environment.yml
conda activate fairmedfm
```

In this checkout, `python main.py` runs the same experiments as `fairmedfm run` with the same arguments, for example:

```bash
python main.py --task cls --usage lp --dataset CXP --sensitive_name Sex --method erm --total_epochs 100 \
  --warmup_epochs 5 --blr 2.5e-4 --batch_size 128 --optimizer adamw --min_lr 1e-5 --weight_decay 0.05
```

Prepare the data as described in [Datasets](datasets.md) and the checkpoints as described in
[Run experiments](benchmark.md#working-directory).

## Notebook tutorials

| Notebook | Colab |
| --- | --- |
| Linear probing | [Open in Colab](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/linear_probing.ipynb) |
| CLIP zero-shot and adaptation | [Open in Colab](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/clip_downstream.ipynb) |
| Segmentation | [Open in Colab](https://colab.research.google.com/github/FairMedFM/FairMedFM/blob/main/notebooks/segmentation.ipynb) |

## Metric consistency

Fairness metrics are computed by `fairmedfm.metrics`, which reproduces the paper's original implementation to
within 1e-6 (see [Fairness metrics](metrics.md#consistency-with-the-paper)). Results from newer environments
installed with `pip install "fairmedfm[cls]"` can differ slightly from the paper because of newer model and
library versions.
