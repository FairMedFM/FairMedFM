# FairMedFM benchmark runner

The code behind the [FairMedFM paper](https://arxiv.org/abs/2407.00983): datasets, 20+ medical imaging foundation
models, linear probing, zero-shot, CLIP adaptation, LoRA and prompted segmentation, with fairness evaluation by the
[`fairmedfm`](https://pypi.org/project/fairmedfm/) package.

To evaluate the fairness of your own model you only need `pip install fairmedfm`. This runner is for reproducing
or extending the benchmark. It is installed from GitHub:

```bash
pip install "fairmedfm-bench @ git+https://github.com/FairMedFM/FairMedFM#subdirectory=benchmark"        # classification
pip install "fairmedfm-bench[seg] @ git+https://github.com/FairMedFM/FairMedFM#subdirectory=benchmark"   # + segmentation
fairmedfm run --help
```

Always use the full GitHub URL: `fairmedfm-bench` is not published on PyPI. See the
[benchmark documentation](https://fairmedfm.github.io/FairMedFM/docs/benchmark/).
