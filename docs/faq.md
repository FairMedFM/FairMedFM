---
title: FairMedFM FAQ - fairness evaluation of classification and segmentation models
description: Answers to common questions about evaluating model fairness with FairMedFM - supported models and tasks, metrics, sensitive attributes, multi-class classification, and the medical foundation model benchmark.
---

# FAQ

## Can I use FairMedFM to evaluate my own model?

Yes. FairMedFM evaluates any binary classifier or segmentation model, from any framework and in any domain,
including non-medical data. Run your model, save per-sample predictions with sensitive attributes, and run
`fairmedfm score`. Your model does not need to be one of the benchmark's foundation models. Labels can be class
names, scores can be probabilities, logits or softmax matrices, and segmentation can start from masks. See
[Evaluate your model](evaluate-your-model.md).

## Does FairMedFM need PyTorch or a GPU?

Not for the fairness metrics: `pip install fairmedfm` depends only on NumPy, pandas and scikit-learn. PyTorch
is needed only for the benchmark runner (`fairmedfm[cls]` or `fairmedfm[seg]`).

## Which fairness metrics does FairMedFM compute?

For binary classification: overall AUC, accuracy, binary cross-entropy and expected calibration error (ECE);
worst-group AUC; AUC, accuracy, BCE and ECE gaps between groups; the equal opportunity gap (difference in true
positive rate); and an equalized odds score, `1 - (TPR gap + TNR gap) / 2`. For segmentation: mean Dice,
worst- and best-group Dice, the Dice gap, the standard deviation and skewness of group Dice, and equity-scaled
Dice. See [Fairness metrics](metrics.md).

## Which sensitive attributes are supported?

Any categorical attribute: sex, age group, race, ethnicity, language, hospital or scanner. Values can be strings
or integers, and an attribute can have two or more groups. Bin continuous attributes such as age before scoring.
`fairmedfm score --sensitive sex age race` evaluates several attributes at once, each separately; group
continuous attributes with `bins`, and add `intersectional=True` (`--intersectional`) for combined groups.

## Does it support multi-class or multi-label classification?

Yes, one class at a time. Pass the class with `pos_label` (`--pos-label`): that class is evaluated against all
others, and with a (samples, classes) score matrix FairMedFM uses that class's column. For multi-label tasks,
evaluate each label separately.

## What input formats does FairMedFM accept?

In Python: lists, NumPy arrays, pandas objects and PyTorch tensors; labels as 0/1, booleans, one-hot rows or
class names; scores as probabilities, logits or softmax matrices; masks as arrays or file paths. On the command
line: CSV, TSV, Parquet, Feather, JSON, JSON Lines and Excel tables, with predictions and patient metadata in one
file or two. See [Evaluate your model](evaluate-your-model.md).

## Why is `eod` higher-is-better while the gaps are lower-is-better?

`eod` is an equalized odds score, `1 - (TPR gap + TNR gap) / 2`, so 1 means equal true positive and true
negative rates across groups. `eo` and the `*-gap` metrics are differences, so 0 means equal performance.

## Which threshold does FairMedFM use?

Accuracy, TPR and TNR in the summary use the threshold that maximizes F1 on all samples, shared by all groups.
The detailed output also reports every metric at threshold 0.5.

## Are the results the same as in the FairMedFM paper?

The package reproduces the paper's metric implementation to within 1e-6, checked in continuous integration.
Benchmark runs in newer environments can differ slightly from the paper because of newer model and library
versions; the paper environment is described in [Reproduce the paper](reproduce.md).

## What is the FairMedFM benchmark?

A fairness benchmark of foundation models for medical imaging. The paper evaluates 20 foundation models on 17
datasets with linear probing, zero-shot classification, CLIP adaptation, parameter-efficient fine-tuning and
promptable segmentation, and compares unfairness mitigation methods. It finds disparities across sex, race and
age in all evaluated models; the repository has since added newer models such as DINOv3, SigLIP2, MedGemma,
RETFound, UNI2-h, SAM2 and SAM3. See [Models](models.md) and [Datasets](datasets.md).

## Is FairMedFM related to MedVLMBench?

Yes. [MedVLMBench](https://github.com/ubc-tea/MedVLMBench) is the companion benchmark for the capability of
medical vision-language models; FairMedFM measures fairness. Together they evaluate both on shared models and
datasets.

## How do I cite FairMedFM?

See [Cite FairMedFM](citation.md).
