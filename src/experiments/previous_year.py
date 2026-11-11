"""Seed a fraction of counties with the PREVIOUS election's result and predict
the current election for held-out counties.

Instead of knowing the current (2024) result for some counties, we only know
how a sample of them voted last time (2020). Those 2020 results are the seed
labels; the models spread them over the graph and predict 2024 for the test
counties. `fracs` is the share of all counties that are seeded (the training
counties); parameters are tuned by 5-fold cross-validation among them.

Baselines reported alongside: 'persistence' (each test county's own 2020
result, which a real forecaster would have for every county) and, for the
same seeds, the usual setup with the seeds' true 2024 labels.
"""
import numpy as np
import pandas as pd

from ..evaluation.labels import build_labels
from ..evaluation.metrics import report
from ..evaluation.results import run_path
from ..evaluation.splits import eligible_counties, fold_array, known_mask, make_folds, make_split, node_arrays
from ..evaluation.tuning import evaluate, tune
from ..graphs.adjacency import get_adjacency_graph
from ..graphs.demographics import get_demographic_graph
from ..models.knn import KNNClassifier
from ..models.label_propagation import LabelPropagation

DEFAULT_FRACS = (0.05, 0.1, 0.2, 0.4, None)  # share of all counties; None = whole train pool

GRIDS = {
    "knn": (KNNClassifier, {"k": [3, 10, None], "weighted": [True, False]}),
    "lp": (LabelPropagation, {"alpha": [0.5, 0.8, 0.95]}),
}


def run(split_name="stratified", fracs=DEFAULT_FRACS, prev_year=2020, year=2024, seed=0, df=None):
    """One run: draw a train/test split with this random seed, sample the seed
    counties, tune by 5-fold CV and score on test. fracs: share of ALL counties
    that are seeded (None = the whole train pool). Returns a list of
    (graph, method, train fraction, metrics dict)."""
    df = eligible_counties() if df is None else df
    split = make_split(split_name, df, seed)
    y_prev_all, y_now_all = build_labels(prev_year), build_labels(year)
    rows = []
    for gname, (geoids, W) in (("border", get_adjacency_graph()), ("demographic", get_demographic_graph())):
        y_now, train, test = node_arrays(geoids, split, y_now_all)
        y_prev = node_arrays(geoids, split, y_prev_all)[0]
        # persistence baseline: a test county's own previous result
        ti = np.flatnonzero(test & ~np.isnan(y_prev))
        rows.append((gname, "persistence", "-", report(y_now[ti].astype(int), y_prev[ti].astype(int))))
        for frac in fracs:
            seeds = known_mask(train & ~np.isnan(y_prev), geoids, df, frac, seed=seed)
            folds = fold_array(geoids, make_folds(split_name, df.loc[np.asarray(geoids)[seeds]], seed=seed))
            label = "max" if frac is None else frac
            for mname, (cls, grid) in GRIDS.items():
                for source, y_fit in (("prev-year seeds", y_prev), ("current-year seeds", y_now)):
                    best, _ = tune(cls, W, y_fit, folds, grid, y_eval=y_now)
                    rows.append((gname, f"{mname} / {source}", label,
                                 evaluate(cls, best, W, y_fit, seeds, test, y_eval=y_now)))
    return rows


def run_repeated(split_name="stratified", seeds=range(5), **kwargs):
    """Repeat run() over several random seeds. Returns a DataFrame with the mean
    and standard deviation across seeds of each metric, per graph/method/fraction."""
    df = eligible_counties()
    recs = []
    for sd in seeds:
        for graph, method, frac, r in run(split_name, seed=sd, df=df, **kwargs):
            recs.append({"seed": sd, "graph": graph, "method": method, "train_frac": str(frac), **r})
    runs = pd.DataFrame(recs)
    runs.to_csv(run_path("previous_year", split_name), index=False)
    keys = ["graph", "method", "train_frac"]
    metrics = ["accuracy", "balanced_accuracy", "f1_democratic"]
    agg = runs.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    return agg


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for sp in ("stratified", "block"):
        agg = run_repeated(sp, seeds=range(n))
        print(f"\n== {sp} split, {n} seeds: test balanced accuracy (mean +- std) / accuracy")
        for (g, name, frac), r in agg.iterrows():
            print(f"{g:11s} {name:28s} train {frac:5s} "
                  f"bal {r[('balanced_accuracy', 'mean')]:.3f} +- {r[('balanced_accuracy', 'std')]:.3f}  "
                  f"acc {r[('accuracy', 'mean')]:.3f}")
