"""Import, configuration and command-line checks for the benchmark runner (needs fairmedfm-bench installed).

These do not download checkpoints or run models.
"""
import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from fairmedfm_bench import parse_args, run  # noqa: E402

# Models whose dependencies are all installed with fairmedfm-bench.
PYPI_CLASSIFICATION_MODELS = [
    "BiomedCLIP", "PubMedCLIP", "MedCLIP", "PLIP", "CLIP", "DINOv2", "DINOv3", "AIMv2", "RADDINO", "SigLIP",
    "SigLIP2", "MedSigLIP", "MedGemma", "UNI2", "ProvGigaPath", "Virchow2", "RETFound", "MedLVM", "C2L",
    "MedMAE", "MoCoCXR",
]


def test_runner_modules_import():
    import fairmedfm_bench.datasets.utils  # noqa: F401
    import fairmedfm_bench.models.utils  # noqa: F401
    import fairmedfm_bench.trainers.utils  # noqa: F401
    import fairmedfm_bench.utils.tokenizer  # noqa: F401
    import fairmedfm_bench.wrappers.utils  # noqa: F401


@pytest.mark.parametrize("name", PYPI_CLASSIFICATION_MODELS)
def test_classification_model_dependencies_are_installed(name):
    from fairmedfm_bench import models
    # Models with a missing dependency are replaced by a placeholder that raises ImportError when built.
    assert "Missing" not in getattr(models, name).__qualname__, f"{name}: dependencies are missing"


def test_wrappers_are_available():
    from fairmedfm_bench import wrappers
    for name in ("CLIPWrapper", "LinearProbeWrapper", "LoRAWrapper"):
        assert "Missing" not in getattr(wrappers, name).__qualname__


def test_segmentation_wrappers_are_available():
    pytest.importorskip("segment_anything")
    from fairmedfm_bench import wrappers
    assert "Missing" not in wrappers.SAMWrapper.__qualname__


def test_packaged_configs_are_used_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_args.collect_args(["--task", "cls", "--usage", "lp", "--dataset", "HAM10000", "--model", "DINOv3",
                                    "--no_cuda", "--exp_path", str(tmp_path / "output")])
    args = run.create_exerpiment_setting(args)
    assert args.data_setting is not None and args.data_setting["test_meta_path"].startswith("./data/HAM10000")
    assert args.model_setting == json.loads(run.config_path("models", "DINOv3").read_text())


def test_working_directory_configs_take_precedence(tmp_path, monkeypatch):
    (tmp_path / "configs" / "models").mkdir(parents=True)
    (tmp_path / "configs" / "models" / "DINOv3.json").write_text('{"pretrained_path": "local"}')
    monkeypatch.chdir(tmp_path)
    assert run.config_path("models", "DINOv3") == tmp_path / "configs" / "models" / "DINOv3.json"


def test_run_command_help():
    result = subprocess.run([sys.executable, "-m", "fairmedfm", "run", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage: fairmedfm run" in result.stdout
    assert "--sensitive_name" in result.stdout
