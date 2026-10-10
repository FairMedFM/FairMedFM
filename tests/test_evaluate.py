import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fairmedfm as fm
from fairmedfm import metrics

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text())
CLS = GOLDEN["cls_two_groups"]
LABEL, PROB, GROUP = np.array(CLS["label"]), np.array(CLS["prob"]), np.array(CLS["group"])


def summary(report, attribute="sensitive"):
    return report.summary.loc[attribute].drop(["n", "n_groups"]).to_dict()


@pytest.mark.parametrize("case", ["cls_two_groups", "cls_three_groups"])
def test_matches_the_paper_implementation(case):
    data = GOLDEN[case]
    report = fm.evaluate(data["label"], data["prob"], data["group"])
    assert summary(report) == pytest.approx(data["summary"], abs=1e-6)
    with pytest.warns(FutureWarning):
        legacy = metrics.classification_fairness(data["prob"], data["label"], data["group"])
    assert summary(report) == legacy["summary"]


def test_segmentation_matches_the_paper_implementation():
    data = GOLDEN["seg_two_groups"]
    report = fm.evaluate_segmentation(data["group"], dice=data["dice"])
    assert summary(report) == pytest.approx(data["summary"], abs=1e-12)


@pytest.mark.parametrize("labels,pos_label", [
    (np.where(LABEL == 1, "malignant", "benign"), "malignant"),
    (LABEL.astype(bool), None),
    (np.eye(2)[LABEL], None),
    (pd.Series(LABEL.astype(float)), None),
])
def test_label_formats(labels, pos_label):
    expected = summary(fm.evaluate(LABEL, PROB, GROUP))
    assert summary(fm.evaluate(labels, PROB, GROUP, pos_label=pos_label)) == pytest.approx(expected)


def test_string_labels_without_pos_label_are_rejected_with_the_values():
    with pytest.raises(ValueError, match=r"\['benign', 'malignant'\].*pos_label"):
        fm.evaluate(np.where(LABEL == 1, "malignant", "benign"), PROB, GROUP)


def test_multiclass_labels_are_evaluated_one_vs_rest():
    classes = np.where(LABEL == 1, 2, np.arange(len(LABEL)) % 2)
    report = fm.evaluate(classes, PROB, GROUP, pos_label=2)
    assert summary(report) == pytest.approx(summary(fm.evaluate(LABEL, PROB, GROUP)))
    assert report.metadata["one_vs_rest"] is True


def test_score_formats():
    expected = summary(fm.evaluate(LABEL, PROB, GROUP))
    softmax = np.stack([1 - PROB, PROB], axis=1)
    assert summary(fm.evaluate(LABEL, softmax, GROUP)) == pytest.approx(expected)
    assert summary(fm.evaluate(LABEL, softmax[:, ::-1], GROUP, score_column=0)) == pytest.approx(expected)
    logits = np.log(PROB / (1 - PROB))
    report = fm.evaluate(LABEL, logits, GROUP)
    assert summary(report) == pytest.approx(expected, abs=1e-6)
    assert report.metadata["score_transform"] == "sigmoid" and "logits" in report.warnings[0]
    report = fm.evaluate(LABEL, np.log(softmax) + 3, GROUP)
    assert summary(report) == pytest.approx(expected, abs=1e-6)
    assert report.metadata["score_transform"] == "softmax"


def test_probabilities_are_not_reinterpreted_and_can_be_forced_to_logits():
    report = fm.evaluate(LABEL, PROB, GROUP, score_type="logit")
    assert report.metadata["score_transform"] == "sigmoid"
    with pytest.raises(ValueError, match="score_type='probability'"):
        fm.evaluate(LABEL, PROB * 3, GROUP, score_type="probability")


def test_score_columns_follow_sorted_class_order():
    names = np.where(LABEL == 1, "benign", "malignant")  # positive class sorts first
    softmax = np.stack([PROB, 1 - PROB], axis=1)        # columns in sorted order: benign, malignant
    report = fm.evaluate(names, softmax, GROUP, pos_label="benign")
    assert report.metadata["score_column"] == 0
    assert summary(report) == pytest.approx(summary(fm.evaluate(LABEL, PROB, GROUP)))


def test_several_classes_need_a_score_column():
    scores = np.random.default_rng(0).dirichlet([1, 1, 1], len(LABEL))
    with pytest.raises(ValueError, match="score_column"):
        fm.evaluate(LABEL, scores, GROUP)
    assert fm.evaluate(LABEL, scores, GROUP, score_column=2).metadata["score_column"] == 2


def test_torch_tensors_are_accepted():
    torch = pytest.importorskip("torch")
    report = fm.evaluate(torch.tensor(LABEL), torch.tensor(PROB), torch.tensor(GROUP))
    assert summary(report) == pytest.approx(summary(fm.evaluate(LABEL, PROB, GROUP)))


def test_sensitive_feature_formats_and_names():
    frame = pd.DataFrame({"sex": np.where(GROUP == 0, "F", "M"), "site": GROUP})
    for features, names in [(frame, ["sex", "site"]), ({"sex": frame["sex"]}, ["sex"]),
                            (frame["sex"], ["sex"]), (list(frame["sex"]), ["sensitive"]),
                            (frame.to_numpy(), ["sensitive_0", "sensitive_1"])]:
        report = fm.evaluate(LABEL, PROB, features)
        assert list(report.summary.index) == names
    assert list(fm.evaluate(LABEL, PROB, frame).by_group.loc["sex"].index) == ["F", "M"]


def test_missing_attribute_values_leave_out_those_samples():
    sex = pd.Series(np.where(GROUP == 0, "F", "M"), dtype=object)
    sex[:10] = None
    report = fm.evaluate(LABEL, PROB, {"sex": sex})
    assert report.summary.loc["sex", "n"] == len(LABEL) - 10
    assert any("10 samples with a missing value" in w for w in report.warnings)


def test_continuous_attributes_must_be_binned():
    age = np.random.default_rng(0).integers(18, 90, len(LABEL))
    with pytest.raises(ValueError, match=r"bins=\{'age'"):
        fm.evaluate(LABEL, PROB, {"age": age})
    report = fm.evaluate(LABEL, PROB, {"age": age}, bins={"age": [40, 60]})
    assert list(report.by_group.loc["age"].index) == ["<40", "40-60", ">=60"]
    assert report.by_group.loc["age", "n"].sum() == len(LABEL)
    quartiles = fm.evaluate(LABEL, PROB, {"age": age}, bins={"age": 4})
    assert quartiles.summary.loc["age", "n_groups"] == 4


def test_intersectional_groups():
    frame = pd.DataFrame({"sex": np.where(GROUP == 0, "F", "M"), "old": np.arange(len(LABEL)) % 2})
    report = fm.evaluate(LABEL, PROB, frame, intersectional=True)
    assert list(report.summary.index) == ["sex", "old", "sex & old"]
    assert report.summary.loc["sex & old", "n_groups"] == 4


def test_groups_with_one_label_are_skipped_not_fatal():
    group = GROUP.astype(object).copy()
    negatives = np.flatnonzero(LABEL == 0)[:5]
    group[negatives] = "tiny"
    report = fm.evaluate(LABEL, PROB, {"site": group})
    assert report.by_group.loc[("site", "tiny"), "skipped"] == "only label 0"
    assert report.summary.loc["site", "n_groups"] == 2
    assert any("tiny" in w for w in report.warnings)


def test_report_serialization(tmp_path):
    report = fm.evaluate(LABEL, PROB, {"sex": np.where(GROUP == 0, "F", "M")})
    data = json.loads(report.to_json(tmp_path / "out" / "fairness.json"))
    assert data["task"] == "cls"
    assert set(data["attributes"]["sex"]["groups"]) == {"F", "M"}
    assert json.loads((tmp_path / "out" / "fairness.json").read_text()) == data
    assert "auc-gap" in repr(report) and "<table" in report._repr_html_()


def masks():
    true = [np.zeros((8, 8), np.uint8) for _ in range(4)]
    for m in true:
        m[2:6, 2:6] = 255
    pred = [m.astype(float) / 255 for m in true]
    pred[1] = np.zeros((8, 8))            # misses the object: Dice 0
    pred[3][2:6, 2:4] = 0.2               # half the object below threshold: Dice 2/3
    return pred, true


def test_dice_from_mask_arrays():
    pred, true = masks()
    report = fm.evaluate_segmentation(["a", "a", "b", "b"], pred_masks=pred, true_masks=true)
    assert report.by_group["mean_dice"].tolist() == pytest.approx([0.5, (1 + 2 / 3) / 2])
    stacked = fm.evaluate_segmentation(["a", "a", "b", "b"], pred_masks=np.stack(pred), true_masks=np.stack(true))
    assert stacked.summary.equals(report.summary)


def test_multiclass_masks_and_empty_masks():
    true = np.zeros((2, 4, 4), int)
    true[0, :2] = 1
    true[0, 2:] = 2
    pred = true.copy()
    pred[0, 2:] = 0
    report = fm.evaluate_segmentation(["a", "b"], pred_masks=pred, true_masks=true, label=2)
    assert report.by_group["mean_dice"].tolist() == [0.0, 1.0]  # sample b: no class 2 anywhere
    report = fm.evaluate_segmentation(["a", "b"], pred_masks=pred, true_masks=true, label=2, empty_score=0)
    assert report.by_group["mean_dice"].tolist() == [0.0, 0.0]


def test_mask_shape_mismatch_names_the_sample():
    with pytest.raises(ValueError, match="sample 1"):
        fm.evaluate_segmentation(["a", "b"], pred_masks=[np.zeros((4, 4)), np.zeros((4, 5))],
                                 true_masks=[np.zeros((4, 4)), np.zeros((4, 4))])


def test_dice_from_mask_files(tmp_path):
    pred, true = masks()
    paths = {"pred": [], "true": []}
    for i, (p, t) in enumerate(zip(pred, true)):
        np.save(tmp_path / f"pred{i}.npy", p)
        np.savez(tmp_path / f"true{i}.npz", mask=t)
        paths["pred"].append(tmp_path / f"pred{i}.npy")
        paths["true"].append(str(tmp_path / f"true{i}.npz"))
    report = fm.evaluate_segmentation(["a", "a", "b", "b"], pred_masks=paths["pred"], true_masks=paths["true"])
    assert report.by_group["mean_dice"].tolist() == pytest.approx([0.5, (1 + 2 / 3) / 2])


def test_dice_from_image_and_nifti_files(tmp_path):
    image = pytest.importorskip("PIL.Image")
    nibabel = pytest.importorskip("nibabel")
    pred, true = masks()
    png, nii = [], []
    for i, (p, t) in enumerate(zip(pred, true)):
        image.fromarray((p >= 0.5).astype(np.uint8) * 255).convert("RGB").save(tmp_path / f"pred{i}.png")
        nibabel.save(nibabel.Nifti1Image(t[..., None].astype(np.uint8), np.eye(4)), tmp_path / f"true{i}.nii.gz")
        png.append(tmp_path / f"pred{i}.png")
        nii.append(tmp_path / f"true{i}.nii.gz")
    report = fm.evaluate_segmentation(["a", "a", "b", "b"], pred_masks=png, true_masks=nii)
    assert report.by_group["mean_dice"].tolist() == pytest.approx([0.5, (1 + 2 / 3) / 2])


def test_nan_dice_is_left_out_with_a_warning():
    report = fm.evaluate_segmentation(["a", "a", "b", "b"], dice=[0.9, np.nan, 0.7, 0.5])
    assert report.by_group["mean_dice"].tolist() == pytest.approx([0.9, 0.6])
    assert "NaN" in report.warnings[0]


def test_pandas_inputs_with_the_same_ids_in_another_order_are_paired_by_index():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"label": rng.integers(0, 2, 300), "sex": rng.choice(["F", "M"], 300)},
                         index=[f"img{i}" for i in range(300)])
    frame["prob"] = np.clip(0.3 + 0.4 * frame["label"] + rng.normal(0, 0.2, 300), 0, 1)
    expected = fm.evaluate(frame["label"], frame["prob"], frame[["sex"]])
    shuffled = frame.sample(frac=1, random_state=1)
    report = fm.evaluate(shuffled["label"], shuffled["prob"], frame[["sex"]])
    assert summary(report, "sex") == pytest.approx(summary(expected, "sex"))
    assert any("reordered to match the index of y_true" in w for w in report.warnings)
    with pytest.warns(UserWarning, match="paired by index"):
        assert fm.auc_gap(shuffled["label"], shuffled["prob"], sensitive_features=frame["sex"]) == pytest.approx(
            expected.summary.loc["sex", "auc-gap"])


def test_a_default_index_next_to_a_shuffled_one_is_ambiguous():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"label": rng.integers(0, 2, 300), "sex": rng.choice(["F", "M"], 300)})
    frame["prob"] = np.clip(0.3 + 0.4 * frame["label"] + rng.normal(0, 0.2, 300), 0, 1)
    shuffled = frame.sample(frac=1, random_state=1)
    # Model outputs in the shuffled row order, as a new Series with the default index: positional intent.
    outputs = pd.Series(shuffled["prob"].to_numpy())
    with pytest.raises(ValueError, match=r"pass y_score.loc\[y_true.index\]; to pair by position"):
        fm.auc_gap(shuffled["label"], outputs, sensitive_features=shuffled["sex"])
    with pytest.raises(ValueError, match="unclear whether samples should be paired"):
        fm.evaluate(shuffled["label"], shuffled["prob"], frame[["sex"]])
    expected = fm.auc_gap(shuffled["label"].to_numpy(), outputs, sensitive_features=shuffled["sex"].to_numpy())
    assert fm.auc_gap(shuffled["label"], outputs.to_numpy(), sensitive_features=shuffled["sex"]) == expected
    # Same frame, or different labels (IDs vs a fresh index): positional, no error and no note.
    assert fm.evaluate(shuffled["label"], shuffled["prob"], shuffled[["sex"]]).warnings == []
    ids = shuffled.set_axis(np.arange(300) + 10_000)
    report = fm.evaluate(ids["label"], ids["prob"], shuffled[["sex"]].reset_index(drop=True))
    assert summary(report, "sex") == pytest.approx(summary(fm.evaluate(frame["label"], frame["prob"], frame[["sex"]]),
                                                           "sex")) and not report.warnings


def test_segmentation_inputs_are_paired_by_index():
    frame = pd.DataFrame({"dice": [0.9, 0.8, 0.5, 0.4], "sex": ["F", "F", "M", "M"]}, index=[10, 11, 12, 13])
    shuffled = frame.loc[[13, 10, 12, 11]]
    with pytest.warns(UserWarning, match="paired by index"):
        assert fm.dice_gap(shuffled["dice"], sensitive_features=frame["sex"]) == pytest.approx(0.4)
