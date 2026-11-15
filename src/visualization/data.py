"""Loading experiment results for the figures."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.results import RUNS_DIR


def latest_run(experiment, variant=None):
    """Most recent results/runs/<date>_<experiment>[_<variant>].csv."""
    pattern = f"*_{experiment}" + (f"_{variant}" if variant else "") + ".csv"
    files = sorted(Path(RUNS_DIR).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {RUNS_DIR}; run main.py first")
    return files[-1]


def load_comparison():
    """The model comparison runs, with `known_share` (share of all counties
    whose labels are known) and the tuned parameters parsed into columns."""
    df = pd.read_csv(latest_run("model_comparison"))
    df["train_frac"] = df.train_frac.astype(str)
    total = (df.n_known + df.n_test)[df.train_frac == "max"].max()  # all eligible counties
    df["known_share"] = df.n_known / total
    params = df.params.fillna("").map(lambda s: ast.literal_eval(s) if s else {})
    for col in ("k", "weighted", "alpha", "lam", "threshold"):
        df[col] = params.map(lambda p: p.get(col, np.nan))
    return df


def load_previous_year(split):
    df = pd.read_csv(latest_run("previous_year", split))
    df["train_frac"] = df.train_frac.astype(str)
    return df
