---
title: How to measure model fairness in Python - FairMedFM recipes
description: Short recipes for measuring the fairness of a classifier or segmentation model in Python - AUC gap, calibration gap, equal opportunity, equalized odds, continuous and intersectional attributes, multi-class models, Dice disparities and model selection.
---

# How-to recipes

Each recipe needs only `pip install fairmedfm`. `y_true` holds the labels, `y_score` the model's outputs and
`group` the sensitive attribute (sex, age group, race, hospital, scanner, ...) of each sample.

```python
import fairmedfm as fm
```

## How do I compute the AUC gap between groups in Python?

```python
fm.auc_gap(y_true, y_score, sensitive_features=group)          # largest minus smallest group AUC; 0 is fair
fm.worst_group_auc(y_true, y_score, sensitive_features=group)  # AUC of the worst group
```

`y_score` can be positive-class probabilities, logits, or a `predict_proba`/softmax matrix. The AUC gap does not
depend on a decision threshold.

## How do I check whether a model is equally well calibrated for every group?

```python
fm.ece_gap(y_true, y_score, sensitive_features=group)   # gap in expected calibration error (10 bins)
fm.bce_gap(y_true, y_score, sensitive_features=group)   # gap in binary cross-entropy
```

`fm.evaluate(...).by_group` lists each group's ECE, so you can see which group is miscalibrated.

## How do I measure equal opportunity and equalized odds from predicted probabilities?

```python
fm.equal_opportunity_difference(y_true, y_score, sensitive_features=group)  # TPR gap; 0 is fair
fm.equalized_odds_score(y_true, y_score, sensitive_features=group)          # 1 - (TPR gap + TNR gap) / 2; 1 is fair
```

Both binarize the scores at the threshold with the best overall F1, shared by all groups. To use your own
threshold, binarize first and use Fairlearn's rate metrics; see
[FairMedFM, Fairlearn and AIF360](comparison.md#using-fairmedfm-with-fairlearn).

## How do I evaluate fairness across age or another continuous attribute?

Group it with `bins`: cut points or a number of quantile groups.

```python
fm.evaluate(y_true, y_score, {"age": age}, bins={"age": [40, 60]})   # <40, 40-60, >=60
fm.evaluate(y_true, y_score, {"age": age}, bins={"age": 4})          # quartiles
```

## How do I evaluate intersectional groups, such as sex and age together?

```python
report = fm.evaluate(y_true, y_score, df[["sex", "age"]], bins={"age": [40, 60]}, intersectional=True)
report.summary        # rows: sex, age, and "sex & age" (groups such as "F & >=60")
```

The single-metric functions compare combined groups whenever `sensitive_features` has several columns.

## How do I evaluate a multi-class classifier?

Pick the class to evaluate with `pos_label`; it is compared against all other classes (one-vs-rest).

```python
fm.evaluate(labels, softmax, group, pos_label="melanoma")   # labels are class names, softmax is (samples, classes)
```

Columns of a score matrix are taken in sorted class order, as in scikit-learn's `predict_proba`; pass
`score_column` if yours differ.

## How do I measure the fairness of a segmentation model?

From per-sample Dice scores, or directly from masks (arrays, tensors or `.npy`, `.png`, `.nii.gz` files):

```python
fm.dice_gap(dice, sensitive_features=group)              # best minus worst group mean Dice
fm.equity_scaled_dice(dice, sensitive_features=group)    # mean Dice / (1 + std of group means)
fm.evaluate_segmentation(group, pred_masks=pred_paths, true_masks=gt_paths).summary
```

## How do I evaluate predictions and patient metadata stored in separate files?

On the command line, join them on an ID column:

```bash
fairmedfm score predictions.csv --metadata patients.csv --on image_id --sensitive sex age --bins age=40,60
```

In Python, pandas inputs whose indexes contain the same labels in a different order are paired by index, so a
predictions table and a metadata table indexed by the same image IDs line up. Otherwise samples are paired by
position, as in scikit-learn.

## How do I track fairness during training?

The functions accept PyTorch tensors on any device:

```python
probs = torch.softmax(model(x_val), dim=1)
wandb.log({"val_auc_gap": fm.auc_gap(y_val, probs, sensitive_features=sex_val)})
```

## How do I select a model by fairness with scikit-learn?

Every single-metric function works as a scorer with metadata routing (scikit-learn 1.4+):

```python
import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

sklearn.set_config(enable_metadata_routing=True)
scorer = make_scorer(fm.auc_gap, response_method="predict_proba", greater_is_better=False)
cross_validate(model, X, y, scoring=scorer.set_score_request(sensitive_features=True),
               params={"sensitive_features": group})
```
