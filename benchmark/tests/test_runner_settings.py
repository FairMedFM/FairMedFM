"""Learning-rate schedule, experiment configs and arguments of the benchmark runner (no PyTorch needed)."""
import math
from types import SimpleNamespace

import pytest

from fairmedfm_bench import parse_args, run
from fairmedfm_bench.utils.lr_sched import adjust_learning_rate, scheduled_lr


def schedule(**overrides):
    args = dict(lr=1e-3, blr=1e-3, min_lr=1e-5, warmup_epochs=5, total_epochs=25, fixed_lr=False)
    return SimpleNamespace(**{**args, **overrides})


def test_learning_rate_warms_up_linearly_then_decays_by_half_cosine():
    args = schedule()
    assert scheduled_lr(0, args) == 0
    assert scheduled_lr(2.5, args) == pytest.approx(5e-4)
    assert scheduled_lr(5, args) == pytest.approx(1e-3)
    assert scheduled_lr(15, args) == pytest.approx(1e-5 + (1e-3 - 1e-5) * (1 + math.cos(math.pi / 2)) / 2)
    assert scheduled_lr(25, args) == pytest.approx(1e-5)
    assert scheduled_lr(15, schedule(fixed_lr=True)) == 1e-3


def test_learning_rate_is_set_on_every_parameter_group_with_its_scale():
    optimizer = SimpleNamespace(param_groups=[{"lr": 0}, {"lr": 0, "lr_scale": 0.1}])
    lr = adjust_learning_rate(optimizer, 5, schedule())
    assert [g["lr"] for g in optimizer.param_groups] == [pytest.approx(lr), pytest.approx(lr * 0.1)]


def runner_args(tmp_path, *argv):
    return parse_args.collect_args(["--no_cuda", "--exp_path", str(tmp_path / "output"), *argv])


def test_missing_dataset_config_is_an_error(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no dataset config for FairSeg"):
        run.create_exerpiment_setting(runner_args(tmp_path, "--task", "seg", "--dataset", "FairSeg"))


def test_sensitive_attribute_without_a_test_split_is_an_error(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="TUSC has test splits for sex, not Race"):
        run.create_exerpiment_setting(runner_args(tmp_path, "--dataset", "TUSC", "--sensitive_name", "Race"))


def test_if_wandb_reads_true_and_false(tmp_path):
    assert runner_args(tmp_path, "--if_wandb", "False").if_wandb is False
    assert runner_args(tmp_path, "--if_wandb", "True").if_wandb is True
    with pytest.raises(SystemExit):
        runner_args(tmp_path, "--if_wandb", "maybe")
