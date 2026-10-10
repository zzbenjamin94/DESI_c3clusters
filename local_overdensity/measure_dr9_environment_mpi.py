"""Measure DR9 photo-z-weighted top-hat and truncated-Gaussian environments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_overdensity.environment_runner import run

if __name__ == "__main__":
    run("dr9")
