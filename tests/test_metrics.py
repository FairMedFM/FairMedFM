import json
from pathlib import Path

import numpy as np
import pytest

from fairmedfm import metrics

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text())
# The legacy BCE runs in float32 on PyTorch; everything else matches to float64 rounding.
TOLERANCE = 1e-6
# classification_fairness and segmentation_fairness are deprecated but still checked against the paper values.
pytestmark = pytest.mark.filterwarnings("ignore:fairmedfm.*_fairness is deprecated:FutureWarning")


@pytest.mark.parametrize("case", ["cls_two_groups", "cls_three_groups"])
def test_classification_matches_legacy_implementation(case):
    data = GOLDEN[case]
    result = metrics.classification_fairness(data["prob"], data["label"], data["group"])
    assert result["summary"] == pytest.approx(data["summary"], abs=TOLERANCE)
    assert result["overall"] == pytest.approx(data["overall"], abs=TOLERANCE)
    overall, subgroup = metrics.evaluate_binary(data["prob"], data["label"], data["group"])
    assert overall == pytest.approx(data["overall"], abs=TOLERANCE)
    for key, values in data["subgroup"].items():
        assert subgroup[key] == pytest.approx(values, abs=TOLERANCE)
    assert metrics.organize_results(overall, subgroup) == pytest.approx(data["summary"], abs=TOLERANCE)


def test_segmentation_matches_legacy_implementation():
    data = GOLDEN["seg_two_groups"]
    assert metrics.evaluate_seg(data["dice"], data["group"]) == pytest.approx(data["summary"], abs=1e-12)


def test_group_labels_can_be_strings_or_non_contiguous_integers():
    data = GOLDEN["cls_two_groups"]
    expected = metrics.classification_fairness(data["prob"], data["label"], data["group"])
    names = np.where(np.asarray(data["group"]) == 0, "female", "male")
    by_name = metrics.classification_fairness(data["prob"], data["label"], names)
    by_code = metrics.classification_fairness(data["prob"], data["label"], np.asarray(data["group"]) * 5 + 3)
    assert by_name["summary"] == expected["summary"] == by_code["summary"]
    assert list(by_name["groups"]) == ["female", "male"]
    assert by_name["groups"]["female"]["n"] + by_name["groups"]["male"]["n"] == len(data["prob"])


def test_result_is_json_serializable():
    data = GOLDEN["cls_three_groups"]
    json.dumps(metrics.classification_fairness(data["prob"], data["label"], data["group"]), allow_nan=False)
    seg = GOLDEN["seg_two_groups"]
    json.dumps(metrics.segmentation_fairness(seg["dice"], seg["group"]), allow_nan=False)


def test_segmentation_supports_more_than_two_groups():
    dice = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    group = ["a", "a", "b", "b", "c", "c"]
    result = metrics.segmentation_fairness(dice, group)
    summary = result["summary"]
    assert summary["min_dice"] == pytest.approx(0.45)
    assert summary["max_dice"] == pytest.approx(0.85)
    assert summary["delta_dice"] == pytest.approx(0.4)
    assert summary["std_dice"] == pytest.approx(np.std([0.85, 0.65, 0.45]))
    assert summary["es_dice"] == pytest.approx(0.65 / (1 + summary["std_dice"]))
    assert result["groups"]["b"] == {"n": 2, "mean_dice": pytest.approx(0.65)}


def test_skewness_is_none_when_a_group_is_perfect():
    summary = metrics.segmentation_fairness([1.0, 1.0, 0.8, 0.6], [0, 0, 1, 1])["summary"]
    assert summary["skewness_dice"] is None
    assert summary["delta_dice"] == pytest.approx(0.3)
    assert metrics.evaluate_seg([1.0, 1.0, 0.8, 0.6], [0, 0, 1, 1])["skewness_dice"] == float("inf")


def test_trainer_interface_keeps_nan_dice_like_the_original():
    summary = metrics.evaluate_seg([0.9, float("nan"), 0.7, 0.5], [[0], [0], [1], [1]])
    assert np.isnan(summary["mean_dice"])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        metrics.segmentation_fairness([0.9, float("nan"), 0.7, 0.5], [0, 0, 1, 1])


def test_bce_clamps_saturated_probabilities_like_torch():
    assert metrics.binary_cross_entropy([0.0, 1.0], [1, 0]) == pytest.approx(100.0)
    assert metrics.binary_cross_entropy([1.0, 0.0], [1, 0]) == pytest.approx(0.0)


@pytest.mark.parametrize("prob,label,group,message", [
    ([0.2, 0.8, 0.3, 0.7], [0, 1, 0, 0], [0, 0, 1, 1], "group '1' contains only label 0"),
    ([0.2, 0.8, 0.3, 0.7], [0, 1, 2, 1], [0, 0, 1, 1], "only 0 and 1"),
    ([0.2, 0.8, 0.3, 1.7], [0, 1, 0, 1], [0, 0, 1, 1], r"in \[0, 1\]"),
    ([0.2, 0.8, 0.3, 0.7], [0, 1, 0, 1], [0, 0, 0, 0], "at least two groups"),
    ([0.2, 0.8, 0.3], [0, 1, 0, 1], [0, 0, 1, 1], "same length"),
])
def test_invalid_classification_inputs_are_rejected(prob, label, group, message):
    with pytest.raises(ValueError, match=message):
        metrics.classification_fairness(prob, label, group)


def test_top_level_exports():
    import fairmedfm
    assert fairmedfm.classification_fairness is metrics.classification_fairness
    assert fairmedfm.segmentation_fairness is metrics.segmentation_fairness


def test_the_0_1_functions_are_deprecated():
    data = GOLDEN["cls_two_groups"]
    with pytest.warns(FutureWarning, match="use fairmedfm.evaluate"):
        metrics.classification_fairness(data["prob"], data["label"], data["group"])
    with pytest.warns(FutureWarning, match="use fairmedfm.evaluate_segmentation"):
        metrics.segmentation_fairness([0.5, 0.7], ["a", "b"])


def test_trainer_interfaces_are_not_deprecated(recwarn):
    data = GOLDEN["cls_two_groups"]
    metrics.organize_results(*metrics.evaluate_binary(data["prob"], data["label"], data["group"]))
    metrics.evaluate_seg([0.5, 0.7], ["a", "b"])
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]
