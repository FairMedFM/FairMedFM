import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fairmedfm as fm

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text())
CLS, SEG = GOLDEN["cls_three_groups"], GOLDEN["seg_two_groups"]

CLASSIFICATION = [(fm.auc_gap, "auc-gap"), (fm.worst_group_auc, "worst-auc"), (fm.accuracy_gap, "acc-gap"),
                  (fm.bce_gap, "bce-gap"), (fm.ece_gap, "ece-gap"), (fm.equal_opportunity_difference, "eo"),
                  (fm.equalized_odds_score, "eod")]
SEGMENTATION = [(fm.dice_gap, "delta_dice"), (fm.worst_group_dice, "min_dice"), (fm.dice_std, "std_dice"),
                (fm.dice_skewness, "skewness_dice"), (fm.equity_scaled_dice, "es_dice")]


@pytest.mark.parametrize("function,key", CLASSIFICATION)
def test_classification_functions_match_the_paper_values(function, key):
    value = function(CLS["label"], CLS["prob"], sensitive_features=CLS["group"])
    assert isinstance(value, float)
    assert value == pytest.approx(CLS["summary"][key], abs=1e-6)


@pytest.mark.parametrize("function,key", SEGMENTATION)
def test_segmentation_functions_match_the_paper_values(function, key):
    assert function(SEG["dice"], sensitive_features=SEG["group"]) == pytest.approx(SEG["summary"][key], abs=1e-12)


def test_several_attributes_are_compared_as_combined_groups():
    meta = pd.DataFrame({"sex": np.arange(len(CLS["label"])) % 2, "site": CLS["group"]})
    expected = fm.evaluate(CLS["label"], CLS["prob"], meta, intersectional=True).summary.loc["sex & site", "auc-gap"]
    assert fm.auc_gap(CLS["label"], CLS["prob"], sensitive_features=meta) == expected


def test_options_are_passed_through():
    names = np.where(np.array(CLS["label"]) == 1, "sick", "healthy")
    softmax = np.stack([1 - np.array(CLS["prob"]), CLS["prob"]], axis=1)
    value = fm.auc_gap(names, softmax, sensitive_features=CLS["group"], pos_label="sick")
    assert value == pytest.approx(CLS["summary"]["auc-gap"], abs=1e-6)
    age = np.random.default_rng(0).integers(20, 90, len(names))
    assert 0 <= fm.auc_gap(names, softmax, sensitive_features={"age": age}, pos_label="sick", bins={"age": 3}) < 1


def test_dice_from_masks():
    pred = [np.ones((4, 4)), np.zeros((4, 4)), np.ones((4, 4))]
    true = [np.ones((4, 4))] * 3
    assert fm.dice_gap(pred_masks=pred, true_masks=true, sensitive_features=["a", "b", "b"]) == pytest.approx(0.5)


def test_skipped_groups_warn_at_the_callers_line():
    group = np.array(CLS["group"], dtype=object)
    group[np.flatnonzero(np.array(CLS["label"]) == 0)[:3]] = "tiny"
    with pytest.warns(UserWarning, match="tiny") as record:
        fm.auc_gap(CLS["label"], CLS["prob"], sensitive_features=group)
    assert record[0].filename == __file__


def test_scikit_learn_scorer_with_metadata_routing():
    sklearn = pytest.importorskip("sklearn")
    from packaging.version import Version
    if Version(sklearn.__version__) < Version("1.4"):
        pytest.skip("metadata routing for scorers needs scikit-learn 1.4")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_validate

    X, y = make_classification(300, random_state=0)
    group = np.random.default_rng(0).choice(["a", "b"], 300)
    scorer = make_scorer(fm.auc_gap, response_method="predict_proba", greater_is_better=False)
    with sklearn.config_context(enable_metadata_routing=True):
        result = cross_validate(LogisticRegression(), X, y, cv=3, params={"sensitive_features": group},
                                scoring={"gap": scorer.set_score_request(sensitive_features=True)})
    assert len(result["test_gap"]) == 3 and (result["test_gap"] <= 0).all()
