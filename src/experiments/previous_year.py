"""Seed a fraction of counties with the PREVIOUS election's result and predict
the current election for held-out counties.

Instead of knowing the current (2024) result for some counties, we only know
how a sample of them voted last time (2020). Those 2020 results are the seed
labels; the models spread them over the graph and predict 2024 for the test
counties. `seed_frac` is the share of training counties used as seeds.

Baselines reported alongside: 'persistence' (each test county's own 2020
result, which a real forecaster would have for every county) and, for the
same seeds, the usual setup with the seeds' true 2024 labels.
"""
import numpy as np

from ..evaluation.labels import build_labels
from ..evaluation.metrics import report
from ..evaluation.splits import eligible_counties, load_split, node_arrays, stratified_split
from ..evaluation.tuning import evaluate, tune
from ..graphs.adjacency import get_adjacency_graph
from ..graphs.demographics import get_demographic_graph
from ..models.knn import KNNClassifier
from ..models.label_propagation import LabelPropagation

GRIDS = {
    "knn": (KNNClassifier, {"k": [3, 10, None], "weighted": [True, False]}),
    "lp": (LabelPropagation, {"alpha": [0.5, 0.8, 0.95]}),
}


def pick_seeds(train_mask, geoids, df, seed_frac, seed=0):
    """Random state-stratified sample of seed_frac of the training counties."""
    if seed_frac >= 1:
        return train_mask.copy()
    train_ids = set(np.asarray(geoids)[train_mask])
    sub = df.loc[[g for g in df.index if g in train_ids]]
    chosen = stratified_split(sub, test_frac=seed_frac, seed=seed)
    chosen = set(chosen.index[chosen == "test"])
    return np.array([g in chosen for g in geoids]) & train_mask


def run(split_name="stratified", fracs=(0.05, 0.1, 0.25, 0.5, 1.0), prev_year=2020, year=2024):
    split = load_split(split_name)
    df = eligible_counties()
    y_prev_all, y_now_all = build_labels(prev_year), build_labels(year)
    rows = []
    for gname, (geoids, W) in (("border", get_adjacency_graph()), ("demographic", get_demographic_graph())):
        y_now, train, val, test = node_arrays(geoids, split, y_now_all)
        y_prev = node_arrays(geoids, split, y_prev_all)[0]
        # persistence baseline: a test county's own previous result
        ti = np.flatnonzero(test & ~np.isnan(y_prev))
        rows.append((gname, "persistence", "-", report(y_now[ti].astype(int), y_prev[ti].astype(int))))
        for frac in fracs:
            seeds = pick_seeds(train & ~np.isnan(y_prev), geoids, df, frac)
            none = np.zeros_like(seeds)
            for mname, (cls, grid) in GRIDS.items():
                for source, y_fit in (("prev-year seeds", y_prev), ("current-year seeds", y_now)):
                    best, _ = tune(cls, W, y_fit, seeds, val, grid, y_eval=y_now)
                    rows.append((gname, f"{mname} / {source}", frac,
                                 evaluate(cls, best, W, y_fit, seeds, none, test, y_eval=y_now)))
    return rows


if __name__ == "__main__":
    for sp in ("stratified", "block"):
        print(f"\n== {sp} split: test balanced accuracy / accuracy")
        for g, name, frac, r in run(sp):
            print(f"{g:11s} {name:28s} seeds {frac!s:5s} bal {r['balanced_accuracy']:.3f} acc {r['accuracy']:.3f}")
