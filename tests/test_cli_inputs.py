import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fairmedfm as fm
from fairmedfm import cli

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text())
CLS = GOLDEN["cls_two_groups"]
N = len(CLS["label"])
SEX = np.where(np.array(CLS["group"]) == 0, "F", "M")
EXPECTED = fm.evaluate(CLS["label"], CLS["prob"], {"sex": SEX}).summary.loc["sex"].drop(["n", "n_groups"]).to_dict()


def run(capsys, *argv):
    cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


def fail(capsys, *argv):
    with pytest.raises(SystemExit) as info:
        cli.main([str(a) for a in argv])
    assert info.value.code == 2
    return capsys.readouterr().err


def test_columns_are_detected_by_common_names(tmp_path, capsys):
    pd.DataFrame({"y_true": CLS["label"], "y_score": CLS["prob"], "Sex": SEX}).to_csv(tmp_path / "p.csv", index=False)
    out, err = run(capsys, "score", tmp_path / "p.csv")
    assert out["Sex"] == pytest.approx(EXPECTED)
    assert "label column 'y_true'" in err and "sensitive attributes: Sex" in err


def test_predictions_and_metadata_in_separate_files(tmp_path, capsys):
    ids = [f"img{i:04d}" for i in range(N)]
    pd.DataFrame({"image_id": ids, "target": CLS["label"], "prob": CLS["prob"]}).to_csv(tmp_path / "p.csv", index=False)
    meta = pd.DataFrame({"image_id": ids, "sex": SEX, "age": np.arange(N) % 90}).sample(frac=1, random_state=0)
    meta.iloc[:-5].to_json(tmp_path / "meta.jsonl", orient="records", lines=True)
    out, err = run(capsys, "score", tmp_path / "p.csv", "--metadata", tmp_path / "meta.jsonl", "--sensitive", "sex",
                   "age", "--bins", "age=30,60")
    assert set(out) == {"sex", "age"}
    assert "5 prediction rows had no metadata" in err
    meta = meta.rename(columns={"image_id": "file"})
    meta.to_csv(tmp_path / "meta.tsv", sep="\t", index=False)
    out, _ = run(capsys, "score", tmp_path / "p.csv", "--metadata", tmp_path / "meta.tsv", "--on", "image_id=file",
                 "--sensitive", "sex")
    assert out["sex"] == pytest.approx(EXPECTED)


def test_string_labels_and_per_class_probabilities(tmp_path, capsys):
    prob = np.array(CLS["prob"])
    table = pd.DataFrame({"diagnosis": np.where(np.array(CLS["label"]) == 1, "malignant", "benign"),
                          "prob_0": 1 - prob, "prob_1": prob, "sex": SEX})
    table.to_csv(tmp_path / "p.csv", index=False)
    assert "pos-label" in fail(capsys, "score", tmp_path / "p.csv", "--label", "diagnosis")
    out, err = run(capsys, "score", tmp_path / "p.csv", "--label", "diagnosis", "--pos-label", "malignant")
    assert out["sex"] == pytest.approx(EXPECTED)
    assert "'prob_0', 'prob_1'" in err
    table = table.rename(columns={"prob_0": "p_benign", "prob_1": "p_malignant"})[["diagnosis", "p_malignant",
                                                                                  "p_benign", "sex"]]
    table.to_csv(tmp_path / "named.csv", index=False)
    out, err = run(capsys, "score", tmp_path / "named.csv", "--label", "diagnosis", "--pos-label", "malignant",
                   "--score", "p_malignant", "p_benign")
    assert out["sex"] == pytest.approx(EXPECTED)
    assert "score column 'p_malignant'" in err


def test_parquet_input(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    pd.DataFrame({"label": CLS["label"], "logit": np.log(np.array(CLS["prob"]) / (1 - np.array(CLS["prob"]))),
                  "sex": SEX}).to_parquet(tmp_path / "p.parquet")
    out, err = run(capsys, "score", tmp_path / "p.parquet")
    assert out["sex"] == pytest.approx(EXPECTED, abs=1e-6)
    assert "sigmoid" in err


def test_quantile_bins_table_output_and_csv(tmp_path, capsys):
    pd.DataFrame({"label": CLS["label"], "prob": CLS["prob"], "age": np.arange(N)}).to_csv(tmp_path / "p.csv",
                                                                                       index=False)
    assert "bins" in fail(capsys, "score", tmp_path / "p.csv")
    cli.main(["score", str(tmp_path / "p.csv"), "--bins", "age=q3", "--format", "table",
              "--output", str(tmp_path / "groups.csv")])
    assert "auc-gap" in capsys.readouterr().out
    groups = pd.read_csv(tmp_path / "groups.csv")
    assert groups["n"].sum() == N and groups["group"].nunique() == 3


def test_segmentation_from_dice_or_mask_files(tmp_path, capsys):
    data = GOLDEN["seg_two_groups"]
    pd.DataFrame({"DSC": data["dice"], "sex": data["group"]}).to_csv(tmp_path / "dice.csv", index=False)
    out, _ = run(capsys, "score", tmp_path / "dice.csv")
    assert out["sex"] == pytest.approx(fm.metrics.evaluate_seg(data["dice"], data["group"]))

    (tmp_path / "masks").mkdir()
    rows = []
    for i in range(4):
        true = np.zeros((6, 6), np.uint8)
        true[1:5, 1:5] = 1
        pred = true.copy() if i % 2 == 0 else np.zeros_like(true)
        np.save(tmp_path / "masks" / f"p{i}.npy", pred)
        np.save(tmp_path / "masks" / f"t{i}.npy", true)
        rows.append({"pred_mask": f"masks/p{i}.npy", "gt_mask": f"masks/t{i}.npy", "site": "AB"[i // 2]})
    pd.DataFrame(rows).to_csv(tmp_path / "masks.csv", index=False)
    out, err = run(capsys, "score", tmp_path / "masks.csv", "--sensitive", "site")
    assert out["site"]["mean_dice"] == pytest.approx(0.5)
    assert "masks from columns 'pred_mask' and 'gt_mask'" in err


def test_ambiguous_and_missing_columns_are_explained(tmp_path, capsys):
    pd.DataFrame({"label": [0, 1], "target": [0, 1], "prob": [0.1, 0.9], "sex": ["F", "M"]}).to_csv(
        tmp_path / "p.csv", index=False)
    assert "several columns could be the label: ['label', 'target']" in fail(capsys, "score", tmp_path / "p.csv")
    pd.DataFrame({"outcome": [0, 1], "sex": ["F", "M"]}).to_csv(tmp_path / "q.csv", index=False)
    err = fail(capsys, "score", tmp_path / "q.csv")
    assert "no prediction columns found" in err and "outcome" in err


def test_original_options_still_work(tmp_path, capsys):
    pd.DataFrame({"p": CLS["prob"], "y": CLS["label"], "sex": SEX}).to_csv(tmp_path / "p.csv", index=False)
    out, _ = run(capsys, "score", "--task", "cls", "--input", tmp_path / "p.csv", "--sensitive", "sex",
                 "--prob-col", "p", "--label-col", "y")
    assert out["sex"] == pytest.approx(EXPECTED)
