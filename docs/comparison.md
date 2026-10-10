---
title: FairMedFM vs Fairlearn and AIF360 - fairness metrics libraries compared
description: How FairMedFM differs from Fairlearn and IBM AIF360 for evaluating model fairness - score-based metrics such as AUC and calibration gaps, segmentation (Dice) fairness, input formats, mitigation, and how to use them together.
---

# FairMedFM, Fairlearn and AIF360

[Fairlearn](https://fairlearn.org/) and [AIF360](https://github.com/Trusted-AI/AIF360) are general-purpose fairness
toolkits with both metrics and bias mitigation algorithms. FairMedFM is narrower: it measures how a model's
performance differs across groups, and covers two things the others do not provide out of the box, threshold-free
and calibration metrics computed from predicted probabilities, and fairness of segmentation models. It has no
mitigation algorithms. The libraries work well together.

## At a glance

As of Fairlearn 0.14 and AIF360 0.6:

| | FairMedFM | Fairlearn | AIF360 |
| --- | --- | --- | --- |
| Focus | Fairness evaluation of classification and segmentation models; medical imaging benchmark | Fairness assessment and mitigation for scikit-learn-style models | Fairness metrics, explanations and mitigation |
| Main inputs | Labels and probabilities, logits or softmax outputs; Dice scores or masks | Labels and hard predictions (or any metric through `MetricFrame`) | `BinaryLabelDataset` objects |
| Per-group AUC, worst-group AUC, AUC gap | Built in (`auc_gap`, `worst_group_auc`) | `MetricFrame` with `roc_auc_score`; `roc_auc_score_group_min` | Not built in |
| Calibration (ECE) and cross-entropy gaps | Built in (`ece_gap`, `bce_gap`) | With a custom metric in `MetricFrame` | Not built in |
| Equal opportunity, equalized odds | From probabilities, at the best-F1 threshold | From hard predictions | From hard predictions |
| Demographic parity, disparate impact | No | Yes | Yes |
| Segmentation: Dice gap, equity-scaled Dice, Dice from masks | Built in | With a custom metric in `MetricFrame` | No |
| Bias mitigation algorithms | No | Reductions (`ExponentiatedGradient`, `GridSearch`), `ThresholdOptimizer`, `CorrelationRemover` | Pre-, in- and post-processing algorithms |
| Dependencies | NumPy, pandas, scikit-learn | NumPy, pandas, scikit-learn, SciPy, narwhals | NumPy, pandas, scikit-learn, SciPy, matplotlib; some algorithms need extras such as TensorFlow |
| License | Apache-2.0 | MIT | Apache-2.0 |

## When to use which

- **Evaluate a medical imaging or other deep learning model** from its predicted probabilities, including AUC,
  calibration and Dice disparities: FairMedFM. It accepts the outputs as they come (PyTorch tensors, logits,
  softmax matrices, class-name labels, mask files) and reports every group, including groups it cannot evaluate.
- **Mitigate unfairness** while training or post-processing a model, or measure **demographic parity**: Fairlearn
  or AIF360.
- **Compare against the FairMedFM paper**, which reports these metrics for 20 medical imaging foundation models:
  FairMedFM gives the same numbers.

## Using FairMedFM with Fairlearn

FairMedFM's metrics relate exactly to Fairlearn's (checked with Fairlearn 0.14). Fairlearn's rate metrics take
hard predictions, so binarize FairMedFM's input at the threshold it uses, the one with the best overall F1:

```python
import fairmedfm as fm
from fairmedfm.metrics import find_threshold
from fairlearn.metrics import MetricFrame, equal_opportunity_difference, equalized_odds_difference
from sklearn.metrics import roc_auc_score

# Same AUC gap
MetricFrame(metrics=roc_auc_score, y_true=y, y_pred=prob, sensitive_features=group).difference()
fm.auc_gap(y, prob, sensitive_features=group)

y_pred = (prob > find_threshold(prob, y)).astype(int)
# Same TPR gap
equal_opportunity_difference(y, y_pred, sensitive_features=group)
fm.equal_opportunity_difference(y, prob, sensitive_features=group)
# FairMedFM's equalized odds score is 1 minus Fairlearn's mean equalized odds difference
1 - equalized_odds_difference(y, y_pred, sensitive_features=group, agg="mean")
fm.equalized_odds_score(y, prob, sensitive_features=group)
```

Note the conventions: `fm.equalized_odds_score` is a score where 1 is fair, while Fairlearn's
`equalized_odds_difference` is a difference where 0 is fair (and by default takes the larger of the TPR and FPR
gaps rather than their mean).

A typical workflow measures with FairMedFM, mitigates with Fairlearn (for example `ThresholdOptimizer`), and
measures again with FairMedFM to see the effect on AUC, calibration and equal opportunity gaps.

## See also

- [Fairness metrics](metrics.md): exact definitions of every FairMedFM metric.
- [Evaluate your model](evaluate-your-model.md): input formats and examples.
