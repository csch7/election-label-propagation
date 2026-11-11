"""Hyperparameter + decision-threshold tuning on the validation split.

    best, results = tune(KNNClassifier, W, y, train, val, {"k": [1, 5, 10]})
    scores = evaluate(KNNClassifier, best, W, y, train, val, test)

Tuning fits on train labels only and scores on val. `evaluate` then refits with
train + val labels known and scores on test once. (The graph is denser in
labels at that point than during tuning, which is expected.)
"""
import itertools

import numpy as np
import pandas as pd

from .metrics import balanced_accuracy, report

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def tune(model_cls, W, y, train, val, param_grid, thresholds=THRESHOLDS, metric=balanced_accuracy):
    """Grid-search param_grid (dict name -> list of values) and the decision
    threshold on the validation counties. Returns (best, results): best is a
    dict of params plus 'threshold' and 'score'; results is every combination
    as a DataFrame sorted best-first."""
    val_idx = np.flatnonzero(val)
    y_val = y[val_idx].astype(int)
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid, values))
        p = model_cls(**params).fit(W, y, train).predict_proba(val_idx)
        for t in thresholds:
            rows.append({**params, "threshold": float(t), "score": metric(y_val, (p > t).astype(int))})
    best = max(rows, key=lambda r: r["score"])  # first best, keeps original value types
    results = pd.DataFrame(rows).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    return best, results


def evaluate(model_cls, best, W, y, train, val, test):
    """Refit with train + val labels and report metrics on the test counties."""
    params = {k: v for k, v in best.items() if k not in ("threshold", "score")}
    test_idx = np.flatnonzero(test)
    p = model_cls(**params).fit(W, y, train | val).predict_proba(test_idx)
    return report(y[test_idx].astype(int), (p > best["threshold"]).astype(int))
