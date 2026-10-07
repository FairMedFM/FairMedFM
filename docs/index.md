---
title: Fairness evaluation for classification and segmentation models
description: Measure the fairness of any binary classification or segmentation model with pip install fairmedfm. Subgroup AUC, accuracy, calibration and Dice gaps, equalized odds, and a benchmark of 20 medical imaging foundation models.
hide:
  - navigation
---

<div class="hero" markdown>

# Fairness evaluation for any classification or segmentation model

<p class="lead">FairMedFM measures how a model's performance differs across groups such as sex, age, hospital or
scanner. Give it your model's predictions and the group of each sample; it reports subgroup AUC, accuracy and
calibration gaps, equal opportunity, equalized odds, and Dice disparities. It is also the FairMedFM benchmark of 20 medical imaging foundation models
on 17 datasets.</p>

[Get started](installation.md){ .md-button .md-button--primary }
[Evaluate your model](evaluate-your-model.md){ .md-button }

</div>

```python
# pip install fairmedfm
import fairmedfm as fm

fm.auc_gap(y_true, y_score, sensitive_features=group)      # one number, like sklearn.metrics
report = fm.evaluate(y_true, y_score, sensitive_features=df[["sex", "age"]], bins={"age": [40, 60]})
```

<div class="grid cards" markdown>

-   **Any model, any domain**

    ---

    Binary classifiers and segmentation models from any framework. Pass labels, probabilities, logits or masks
    as you have them; FairMedFM does not need to run your model.

-   **Fits your workflow**

    ---

    sklearn-style functions for training loops and model selection, pandas reports, or a command line for
    prediction files. Needs only NumPy, pandas and scikit-learn.

-   **Same numbers as the paper**

    ---

    The metrics are checked against the implementation behind the FairMedFM paper, in continuous integration.

-   **Full benchmark included**

    ---

    `pip install "fairmedfm[cls]"` or `"fairmedfm[seg]"` adds the benchmark runner: linear probing, CLIP
    zero-shot and adaptation, and promptable segmentation with SAM-family models.

</div>

## What it measures

| Task | Metrics |
| --- | --- |
| Binary classification | Overall AUC, accuracy, BCE and ECE; worst-group AUC; AUC, accuracy, BCE and ECE gaps between groups; equal opportunity (TPR gap) and equalized odds |
| Segmentation | Mean Dice; worst and best group Dice and their gap; standard deviation and skewness across groups; equity-scaled Dice |

See [Fairness metrics](metrics.md) for exact definitions.

## Two ways to use FairMedFM

**Evaluate your own model.** Run your model anywhere, then call a metric such as `fm.auc_gap` or the full
`fm.evaluate` on its outputs, use them as scikit-learn scorers, or point `fairmedfm score` at prediction files. See
[Evaluate your model](evaluate-your-model.md).

**Run the benchmark.** Evaluate built-in foundation models such as CLIP, BiomedCLIP, MedCLIP, DINOv2, SigLIP,
RETFound, SAM and MedSAM on medical imaging datasets with `fairmedfm run`. See [Run experiments](benchmark.md).

![FairMedFM overview: datasets, foundation models, usages and fairness evaluation](assets/overview.png)

## Key findings from the FairMedFM benchmark

- **Bias is pervasive**: all evaluated medical foundation models show fairness disparities across sex, race and age.
- **Utility and fairness trade off differently**: the most accurate model is often not the fairest.
- **The dataset matters most**: disparities on a dataset stay similar regardless of the foundation model.
- **Mitigation has limited effect**: existing unfairness mitigation methods give only marginal improvements.

Read the paper: [FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models](https://arxiv.org/abs/2407.00983).
For capability (rather than fairness) evaluation of medical vision-language models, see the companion benchmark
[MedVLMBench](https://github.com/ubc-tea/MedVLMBench).
