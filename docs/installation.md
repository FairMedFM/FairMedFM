---
title: Install FairMedFM
description: Install the fairmedfm Python package for fairness metrics, or add the classification and segmentation extras to run the FairMedFM foundation model benchmark.
---

# Installation

FairMedFM supports Python 3.10 to 3.13 and is published on [PyPI](https://pypi.org/project/fairmedfm/).

## Fairness metrics only

```bash
pip install fairmedfm
```

This installs the metrics, the `fairmedfm score` command and the Python API, with NumPy, pandas and
scikit-learn as the only dependencies. To read Parquet, Feather or Excel tables and PNG, TIFF or NIfTI masks, add
the `io` extra:

```bash
pip install "fairmedfm[io]"
```

The base install is all you need to [evaluate your own model](evaluate-your-model.md),
on any operating system and without a GPU.

## Benchmark runner

To run the benchmark with the built-in foundation models, install an extra. Install PyTorch for your CUDA
version first (see [pytorch.org](https://pytorch.org/get-started/locally/)); otherwise pip installs the default
PyTorch build.

=== "Classification"

    ```bash
    pip install "fairmedfm[cls]"
    ```

    Linear probing, CLIP zero-shot and CLIP adaptation with the classification models.

=== "Segmentation"

    ```bash
    pip install "fairmedfm[seg]"
    ```

    Promptable segmentation with SAM-family models. Includes everything in `[cls]`.

The extras keep transformers below 5, albumentations below 2, torchmetrics below 1.7 and setuptools below 81,
because the benchmark code uses APIs that later versions removed.

### Models with separate installs

These models need packages that are not on PyPI or that conflict with the extras. FairMedFM reports which
package is missing when you select one of them.

| Model | Install |
| --- | --- |
| BLIP, BLIP2 | `pip install salesforce-lavis` in a separate environment (it pins older dependencies) |
| CONCH | `pip install git+https://github.com/Mahmoodlab/CONCH.git` |
| Merlin | `pip install merlin-vlm` |
| SAM2, MedSAM2 | Install [SAM2](https://github.com/facebookresearch/sam2) |
| SAM3, MedicalSAM3 | `pip install git+https://github.com/facebookresearch/sam3.git` (Python 3.12+, PyTorch 2.7+) |

Some models are gated on Hugging Face: request access on the model page, then run `huggingface-cli login`
or set `HF_TOKEN`. See [Models](models.md).

## Check the installation

```bash
fairmedfm --version
fairmedfm run --help      # needs [cls] or [seg]
```

## Paper environment

The paper's experiments used Python 3.8 and the conda environment in the repository. To use it, clone the
repository and run `python main.py`, which accepts the same arguments as `fairmedfm run`:

```bash
git clone https://github.com/FairMedFM/FairMedFM.git
cd FairMedFM
conda env create -f environment.yml
conda activate fairmedfm
python main.py --help
```

See [Reproduce the paper](reproduce.md).
