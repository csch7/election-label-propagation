"""Run the full comparison: kNN vs label propagation on the border graph and
the demographic graph, predicting county-level 2024 presidential winners.

    python main.py                 # 10 seeds, both split strategies
    python main.py --seeds 3 --splits stratified
    python main.py --previous-year # also run the previous-year seeding experiment

For every (split strategy, random seed, graph, model):
  1. draw a train / validation / test split of the counties,
  2. tune the model's parameters and decision threshold on validation counties,
  3. refit with train + validation labels and score once on the test counties.
Results are saved to results/runs/<date>_model_comparison.csv and summarised
as mean +- std over seeds. Metrics treat Democratic as the positive class.
"""
import argparse

import numpy as np
import pandas as pd

from src.evaluation.metrics import report
from src.evaluation.results import run_path
from src.evaluation.splits import eligible_counties, make_split, node_arrays
from src.evaluation.tuning import evaluate, tune
from src.experiments.previous_year import run_repeated
from src.graphs.adjacency import get_adjacency_graph
from src.graphs.demographics import get_demographic_graph
from src.models.knn import KNNClassifier
from src.models.label_propagation import LabelPropagation

MODELS = {
    "knn": (KNNClassifier, {"k": [1, 3, 5, 10, 20, None], "weighted": [True, False]}),
    "label_propagation": (LabelPropagation, {"alpha": [0.5, 0.8, 0.9, 0.95, 0.99]}),
}


def run_comparison(splits, seeds):
    df = eligible_counties()
    graphs = {"border": get_adjacency_graph(), "demographic": get_demographic_graph()}
    rows = []
    for split_name in splits:
        for seed in seeds:
            split = make_split(split_name, df, seed)
            for gname, (geoids, W) in graphs.items():
                y, train, val, test = node_arrays(geoids, split)
                ti = np.flatnonzero(test)
                base = {"split": split_name, "seed": seed, "graph": gname, "n_test": len(ti)}
                rows.append({**base, "model": "majority", "params": "",
                             **report(y[ti].astype(int), np.zeros(len(ti), dtype=int))})
                for mname, (cls, grid) in MODELS.items():
                    best, _ = tune(cls, W, y, train, val, grid)
                    scores = evaluate(cls, best, W, y, train, val, test)
                    params = {k: v for k, v in best.items() if k != "score"}
                    rows.append({**base, "model": mname, "params": str(params),
                                 "val_balanced_accuracy": best["score"], **scores})
            print(f"finished {split_name} seed {seed}", flush=True)
    return pd.DataFrame(rows)


def summarise(runs):
    metrics = ["accuracy", "balanced_accuracy", "f1_democratic"]
    agg = runs.groupby(["split", "graph", "model"], sort=False)[metrics].agg(["mean", "std"])
    for split_name, g in agg.groupby(level="split", sort=False):
        print(f"\n== {split_name} split, {runs.seed.nunique()} seeds (mean +- std over seeds)")
        print(f"{'graph':12s} {'model':18s} {'bal. acc':>15s} {'accuracy':>15s} {'F1 (Dem)':>15s}")
        for (_, graph, model), r in g.iterrows():
            cell = lambda m: f"{r[(m, 'mean')]:.3f} +- {r[(m, 'std')]:.3f}"
            print(f"{graph:12s} {model:18s} {cell('balanced_accuracy'):>15s} {cell('accuracy'):>15s} {cell('f1_democratic'):>15s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of random seeds (default 10)")
    ap.add_argument("--splits", nargs="+", default=["stratified", "block"], choices=["stratified", "block"])
    ap.add_argument("--previous-year", action="store_true", help="also run the previous-year seeding experiment")
    args = ap.parse_args()

    runs = run_comparison(args.splits, range(args.seeds))
    path = run_path("model_comparison")
    runs.to_csv(path, index=False)
    summarise(runs)
    print(f"\nsaved {path}")

    if args.previous_year:
        for split_name in args.splits:
            run_repeated(split_name, seeds=range(args.seeds))
        print("previous-year runs saved to results/runs/")
