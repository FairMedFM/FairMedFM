import ipdb
import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# adapted from https://github.com/LalehSeyyed/Underdiagnosis_NatMed/blob/main/CXP/classification/predictions.py
# and https://github.com/MLforHealth/CXR_Fairness/blob/master/cxr_fairness/metrics.py
def find_threshold(tol_output, tol_target):
    # to find this thresold, first we get the precision and recall without this, from there we calculate f1 score,
    # using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation
    # are used to calculate our binary output.

    p, r, t = sklm.precision_recall_curve(tol_target, tol_output)
    # Choose the best threshold based on the highest F1 measure
    f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p) + 1e-8))
    bestthr = t[np.where(f1 == max(f1))]

    return bestthr[0]


def bce_loss(pred_probs, labels):
    bce = nn.BCELoss()

    pred_probs, labels = torch.from_numpy(pred_probs).flatten(
    ).cuda(), torch.from_numpy(labels).flatten().cuda()
    with torch.no_grad():
        loss = bce(pred_probs, labels.float())
    # print(loss)
    return loss.item()


def binary_classification_report(pred, y, threshold=0.5, suffix=""):
    auc = roc_auc_score(y, pred)
    ece = expected_calibration_error(pred, y)
    bce = bce_loss(pred, y)

    tn, fp, fn, tp = confusion_matrix(
        y, (pred > threshold).astype(int)).ravel()
    report = {
        f"auc{suffix}": auc,
        f"acc{suffix}": (tp + tn) / (tn + fp + fn + tp),
        f"bce{suffix}": bce,
        f"ece{suffix}": ece,
        f"tpr{suffix}": tp / (tp + fn),
        f"tnr{suffix}": tn / (tn + fp),
        # "fpr": fp / (fp + tn),
        # "fnr": fn / (fn + tp),
        f"tn{suffix}": tn,
        f"fp{suffix}": fp,
        f"fn{suffix}": fn,
        f"tp{suffix}": tp,
    }

    return report


def expected_calibration_error(pred_probs, labels, num_bins=10, metric_variant="abs", quantile_bins=False):
    """
    Computes the calibration error with a binning estimator over equal sized bins
    See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685.
    Does not currently support sample weights
    https://github.com/MLforHealth/CXR_Fairness/blob/c2a0e884171d6418e28d59dca1ccfb80a3f125fe/cxr_fairness/metrics.py#L1557
    """
    if metric_variant == "abs":
        transform_func = np.abs
    elif (metric_variant == "squared") or (metric_variant == "rmse"):
        transform_func = np.square
    else:
        raise ValueError("provided metric_variant not supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut

    bin_ids = cut_fn(pred_probs, num_bins, labels=False, retbins=False)
    df = pd.DataFrame(
        {"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids})
    ece_df = (
        df.groupby("bin_id")
        .agg(
            pred_probs_mean=("pred_probs", "mean"),
            labels_mean=("labels", "mean"),
            bin_size=("pred_probs", "size"),
        )
        .assign(
            bin_weight=lambda x: x.bin_size / df.shape[0],
            err=lambda x: transform_func(x.pred_probs_mean - x.labels_mean),
        )
    )
    result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
    if metric_variant == "rmse":
        result = np.sqrt(result)
    return result


def evaluate_binary(pred, Y, A):
    threshold_list = [0.5, find_threshold(pred, Y)]
    suffix_list = ["", "@best_f1"]

    overall_metrics = {}
    subgroup_metrics = {}

    for threshold, suffix in zip(threshold_list, suffix_list):
        # overall
        overall_metrics.update(binary_classification_report(
            pred, Y, threshold=threshold, suffix=suffix))

        # subgroup
        idx = np.arange(pred.shape[0])

        for i in range(len(np.unique(A))):
            idx_sub = idx[np.where(A == i)[0]]

            sub_report = binary_classification_report(
                pred[idx_sub], Y[idx_sub], threshold=threshold, suffix=suffix)

            for k, v in sub_report.items():
                if k not in subgroup_metrics.keys():
                    subgroup_metrics[k] = []

                subgroup_metrics[k].append(v)

    return overall_metrics, subgroup_metrics


def organize_results(overall_metrics, subgroup_metrics):
    subgroup_auc = subgroup_metrics["auc"]
    subgroup_acc = subgroup_metrics["acc@best_f1"]
    subgroup_bce = subgroup_metrics["bce"]
    subgroup_ece = subgroup_metrics["ece"]
    subgroup_tpr = subgroup_metrics["tpr@best_f1"]
    subgroup_tnr = subgroup_metrics["tnr@best_f1"]

    result = {
        "overall-auc": overall_metrics["auc"],
        "overall-acc": overall_metrics["acc@best_f1"],
        "overall-bce": overall_metrics["bce"],
        "overall-ece": overall_metrics["ece"],
        "worst-auc": min(subgroup_auc),
        "auc-gap": max(subgroup_auc) - min(subgroup_auc),
        "acc-gap": max(subgroup_acc) - min(subgroup_acc),
        "bce-gap": max(subgroup_bce) - min(subgroup_bce),
        "ece-gap": max(subgroup_ece) - min(subgroup_ece),
        "eod": 1 - ((max(subgroup_tpr) - min(subgroup_tpr)) + (max(subgroup_tnr) - min(subgroup_tnr))) / 2,
        "eo": max(subgroup_tpr) - min(subgroup_tpr),
    }

    return result


def evaluate_seg(dsc_list, sensitive_list):
    dsc_list, sensitive_list = np.array(
        dsc_list), np.array(sensitive_list).squeeze()

    mean_dice = dsc_list.mean()
    # TODO: modify for multi-class sensitive
    class_0_dice = dsc_list[sensitive_list == 0]
    class_1_dice = dsc_list[sensitive_list == 1]

    mean_class_0_dice = class_0_dice.mean()
    mean_class_1_dice = class_1_dice.mean()

    min_dice = min(mean_class_0_dice, mean_class_1_dice)
    max_dice = max(mean_class_0_dice, mean_class_1_dice)
    delta_dice = abs(mean_class_0_dice - mean_class_1_dice)
    sk_dice = (1 - min_dice) / (1 - max_dice)
    std_dice = np.array([mean_class_0_dice, mean_class_1_dice]).std()
    es_dice = mean_dice / (1 + std_dice)

    result = {
        "mean_dice": mean_dice,
        "min_dice": min_dice,
        "max_dice": max_dice,
        "delta_dice": delta_dice,
        "skewness_dice": sk_dice,
        "std_dice": std_dice,
        "es_dice": es_dice
    }

    return result
