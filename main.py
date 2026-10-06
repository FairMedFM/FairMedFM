"""Run a FairMedFM experiment from a source checkout: python main.py --task cls ...

Equivalent to ``fairmedfm run ...`` after ``pip install 'fairmedfm[cls]'`` or ``'fairmedfm[seg]'``.
"""
import sys
from pathlib import Path

# Use this checkout's code even if another fairmedfm version is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fairmedfm.run import main  # noqa: E402

if __name__ == "__main__":
    main()
