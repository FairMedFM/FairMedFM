"""Run a FairMedFM experiment from a source checkout: python main.py --task cls ...

Equivalent to ``fairmedfm run ...`` after installing the benchmark runner (see benchmark/README.md).
"""
import sys
from pathlib import Path

# Use this checkout's code even if another fairmedfm version is installed.
root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "src"), str(root / "benchmark")]

from fairmedfm_bench.run import main  # noqa: E402

if __name__ == "__main__":
    main()
