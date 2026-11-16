"""Hyperparameter tuning by k-fold cross-validation.

    folds = fold_array(geoids, make_folds("block", known_df))   # -1 = unknown county
    best, results = tune(KNNClassifier, W, y, folds, {"k": [1, 5, 10]})
    scores = evaluate(KNNClassifier, best, W, y, folds >= 0, test)

The known counties (folds >= 0) are the training counties. For each fold the
model is fitted with the other folds' labels and predicts the held-out fold;
parameters are chosen on those pooled out-of-fold predictions. `evaluate` then
fits with all known counties' labels and scores the test counties once. (The
final fit has 5/5 of the known counties' labels rather than the 4/5 used inside
each fold, the usual cross-validation convention.)

The decision threshold is not tuned: a county is called Democratic when its
predicted probability exceeds the share of Democratic counties among the known
counties. That is the best rule for balanced accuracy when probabilities are
roughly calibrated, and tuning it on a handful of Democratic counties was noisy.
"""
import itertools

import numpy as np
import pandas as pd

from .metrics import balanced_accuracy, report


def tune(model_cls, W, y, folds, param_grid, metric=balanced_accuracy, y_eval=None):
    """Cross-validated grid search over param_grid (dict name -> list of values).
    folds: int array over nodes, fold id 0..k-1 for known counties and -1
    elsewhere. Returns (best, results): best is a dict of the chosen params plus
    'threshold' (the prior, see module doc) and 'score'; results is every
    combination as a DataFrame sorted best-first. y_eval: labels to score
    against when they differ from the labels the model is fitted on (default: y)."""
    folds = np.asarray(folds)
    known = folds >= 0
    threshold = float(np.nanmean(y[known]))  # share of Democratic counties among the known counties
    held_idx = np.concatenate([np.flatnonzero(folds == f) for f in range(folds.max() + 1)])
    y_true = (y if y_eval is None else y_eval)[held_idx].astype(int)
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid, values))
        p = np.concatenate([
            model_cls(**params).fit(W, y, known & (folds != f)).predict_proba(np.flatnonzero(folds == f))
            for f in range(folds.max() + 1)])
        rows.append({**params, "threshold": threshold, "score": metric(y_true, (p > threshold).astype(int))})
    best = max(rows, key=lambda r: r["score"])  # first best, keeps original value types
    results = pd.DataFrame(rows).sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    return best, results


def evaluate(model_cls, best, W, y, known, test, y_eval=None):
    """Fit with all known counties' labels and report metrics on the test counties."""
    params = {k: v for k, v in best.items() if k not in ("threshold", "score")}
    test_idx = np.flatnonzero(test)
    p = model_cls(**params).fit(W, y, known).predict_proba(test_idx)
    y_test = (y if y_eval is None else y_eval)[test_idx].astype(int)
    return report(y_test, (p > best["threshold"]).astype(int), p)
