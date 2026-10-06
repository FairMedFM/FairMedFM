import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from fairmedfm import cli, metrics

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "legacy_golden.json").read_text())


def classification_csv(path):
    data = GOLDEN["cls_two_groups"]
    sex = ["F" if g == 0 else "M" for g in data["group"]]
    age = ["<60" if i % 3 else "60+" for i in range(len(sex))]
    pd.DataFrame({"prob": data["prob"], "label": data["label"], "sex": sex, "age": age}).to_csv(path, index=False)
    return data


def test_score_classification_with_several_attributes(tmp_path, capsys):
    data = classification_csv(tmp_path / "pred.csv")
    cli.main(["score", "--task", "cls", "--input", str(tmp_path / "pred.csv"), "--sensitive", "sex", "age",
              "--output", str(tmp_path / "out" / "fairness.json")])
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"sex", "age"}
    assert printed["sex"] == pytest.approx(data["summary"], abs=1e-6)
    saved = json.loads((tmp_path / "out" / "fairness.json").read_text())
    assert saved["task"] == "cls"
    assert set(saved["attributes"]["sex"]["groups"]) == {"F", "M"}


def test_score_segmentation_with_custom_column(tmp_path, capsys):
    data = GOLDEN["seg_two_groups"]
    pd.DataFrame({"dsc": data["dice"], "sex": data["group"]}).to_csv(tmp_path / "dice.csv", index=False)
    cli.main(["score", "--task", "seg", "--input", str(tmp_path / "dice.csv"), "--sensitive", "sex",
              "--dice-col", "dsc"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["sex"] == pytest.approx(metrics.evaluate_seg(data["dice"], data["group"]))


def test_missing_column_is_a_clear_error(tmp_path, capsys):
    classification_csv(tmp_path / "pred.csv")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["score", "--task", "cls", "--input", str(tmp_path / "pred.csv"), "--sensitive", "race"])
    assert exit_info.value.code == 2
    assert "no column(s) race" in capsys.readouterr().err


def test_module_entry_point_reports_version():
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-m", "fairmedfm", "--version"], capture_output=True, text=True,
                            env=env)
    assert result.returncode == 0
    assert result.stdout.startswith("fairmedfm ")
