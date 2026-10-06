"""Regenerate tests/fixtures/legacy_golden.json from the legacy utils/metrics.py (the code behind the paper results).

Run from the repository root in the FairMedFM training environment with a CUDA GPU (the legacy BCE uses CUDA):
    python tests/fixtures/make_legacy_golden.py
"""
import json, sys, types
import numpy as np
sys.modules["ipdb"] = types.ModuleType("ipdb")
sys.path.insert(0, ".")
from utils import metrics as legacy

def plain(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = [float(x) for x in v]
        else:
            out[k] = float(v)
    return out

cases = {}
rng = np.random.default_rng(0)
for name, k, n in [("cls_two_groups", 2, 600), ("cls_three_groups", 3, 900)]:
    group = rng.integers(0, k, n)
    label = rng.integers(0, 2, n)
    logit = 1.2 * (label - 0.5) + 0.4 * group + rng.normal(0, 1, n)
    prob = (1 / (1 + np.exp(-logit))).astype(np.float32)
    overall, subgroup = legacy.evaluate_binary(prob, label, group)
    cases[name] = {"prob": prob.tolist(), "label": label.tolist(), "group": group.tolist(),
                   "overall": plain(overall), "subgroup": plain(subgroup),
                   "summary": plain(legacy.organize_results(overall, subgroup))}
group = rng.integers(0, 2, 300)
dice = np.clip(rng.normal(0.8 - 0.05 * group, 0.1), 0, 0.99)
cases["seg_two_groups"] = {"dice": dice.tolist(), "group": group.tolist(),
                           "summary": plain(legacy.evaluate_seg(dice.tolist(), group.tolist()))}
json.dump(cases, open("tests/fixtures/legacy_golden.json", "w"))
print({k: v["summary"] for k, v in cases.items()})
