"""Loads config/config.yml (the single place for tunable settings)."""
from functools import cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yml"


@cache
def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_fracs(values):
    """Train fractions as written in the config or on the command line:
    numbers, or "max" (= every non-test county) which becomes None."""
    return [None if str(v) == "max" else float(v) for v in values]
