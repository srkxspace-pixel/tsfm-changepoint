"""P1: zero-shot TSFM change-point detection vs classical CPD."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name):
    """Load configs/<name>.yaml."""
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
