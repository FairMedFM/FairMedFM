---
title: Evaluate the fairness of your own classification or segmentation model
description: Score any binary classifier or segmentation model for fairness across sex, age, race or other groups. Export per-sample predictions to CSV and run fairmedfm score, or call the Python API.
---

# Evaluate your model

FairMedFM evaluates predictions, not models: run your model in its own environment, save one row per sample,
and score the file. This works for any binary classifier or segmentation model, in medical imaging or any other
domain.

```bash
pip install fairmedfm
```

## Binary classification

Save a CSV with the predicted probability of the positive class, the ground-truth label (0 or 1) and one column
per sensitive attribute. Attribute values can be strings or integers.

```text title="predictions.csv"
prob,label,sex,age
0.91,1,F,60+
0.12,0,M,<60
0.67,1,M,60+
```

```bash
fairmedfm score --task cls --input predictions.csv --sensitive sex age --output fairness.json
```

The command prints the fairness summary for each attribute:

```json
{
  "sex": {
    "overall-auc": 0.91,
    "overall-acc": 0.83,
    "overall-bce": 0.38,
    "overall-ece": 0.04,
    "worst-auc": 0.88,
    "auc-gap": 0.05,
    "acc-gap": 0.03,
    "bce-gap": 0.06,
    "ece-gap": 0.02,
    "eod": 0.96,
    "eo": 0.04
  },
  "age": { "...": "..." }
}
```

`--output` also saves the metrics of every group at threshold 0.5 and at the best-F1 threshold, with sample
counts. Use `--prob-col` and `--label-col` if your columns have other names.

From Python:

```python
from fairmedfm import classification_fairness

result = classification_fairness(prob, label, sex)
result["summary"]          # the fairness summary above
result["overall"]          # metrics on all samples
result["groups"]["F"]      # metrics for one group, with its sample count
```

!!! note "Requirements"
    Every group needs both positive and negative samples, so that AUC, TPR and TNR are defined. Multi-class
    classification is not supported yet: score each class one-vs-rest.

## Segmentation

Save a CSV with the Dice score of each image or volume and its sensitive attributes:

```text title="dice.csv"
dice,sex
0.87,F
0.79,M
0.91,F
```

```bash
fairmedfm score --task seg --input dice.csv --sensitive sex
```

```python
from fairmedfm import segmentation_fairness

result = segmentation_fairness(dice, sex)
result["summary"]    # mean_dice, min_dice, max_dice, delta_dice, skewness_dice, std_dice, es_dice
result["groups"]     # mean Dice and sample count per group
```

Attributes can have any number of groups. Use `--dice-col` if the column has another name.

## Exporting predictions

Any framework works. A typical PyTorch loop for a binary classifier:

```python
import pandas as pd
import torch

rows = []
model.eval()
with torch.no_grad():
    for images, labels, sex in loader:
        prob = torch.softmax(model(images), dim=1)[:, 1]
        rows += [{"prob": p, "label": y, "sex": s} for p, y, s in zip(prob.tolist(), labels.tolist(), sex)]
pd.DataFrame(rows).to_csv("predictions.csv", index=False)
```

For segmentation, compute one Dice score per image or volume (for example with
`torchmetrics.functional.dice` or MONAI's `DiceMetric`) and save it with the sample's attributes.

## Next steps

- [Fairness metrics](metrics.md): what each number means.
- [Command line](cli.md) and [Python API](api.md) references.
