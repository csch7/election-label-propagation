"""Run the full comparison: kNN vs label propagation on the border graph and
the demographic graph, predicting county-level 2024 presidential winners.

    python main.py                 # 10 seeds, both split strategies
    python main.py --seeds 3 --splits stratified
    python main.py --previous-year # also run the previous-year seeding experiment

For every (split strategy, random seed, graph, model):
  1. draw a train / validation / test split of the counties,
  2. choose the known (seed) counties: --seed-frac of the training counties,
  3. tune the model's parameters and decision threshold on validation counties,
  4. score once on the test counties, with only the seed counties' labels known.
Results are saved to results/runs/<date>_model_comparison.csv and summarised
as mean +- std over seeds. Metrics treat Democratic as the positive class.
"""
import argparse

import numpy as np
import pandas as pd

from src.evaluation.metrics import report
from src.evaluation.results import run_path
from src.evaluation.splits import eligible_counties, make_split, node_arrays, pick_seeds
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


def run_comparison(splits, seeds, fracs=(1.0,)):
    df = eligible_counties()
    graphs = {"border": get_adjacency_graph(), "demographic": get_demographic_graph()}
    rows = []
    for split_name in splits:
        for seed in seeds:
            split = make_split(split_name, df, seed)
            for gname, (geoids, W) in graphs.items():
                y, train, val, test = node_arrays(geoids, split)
                ti = np.flatnonzero(test)
                for frac in fracs:
                    # known (seed) counties: a state-stratified sample of the training counties
                    seeds_mask = pick_seeds(train, geoids, df, frac, seed=seed)
                    base = {"split": split_name, "seed": seed, "graph": gname, "seed_frac": frac,
                            "n_seeds": int(seeds_mask.sum()), "n_test": len(ti)}
                    rows.append({**base, "model": "majority", "params": "",
                                 **report(y[ti].astype(int), np.zeros(len(ti), dtype=int))})
                    for mname, (cls, grid) in MODELS.items():
                        best, _ = tune(cls, W, y, seeds_mask, val, grid)
                        scores = evaluate(cls, best, W, y, seeds_mask, np.zeros_like(seeds_mask), test)
                        params = {k: v for k, v in best.items() if k != "score"}
                        rows.append({**base, "model": mname, "params": str(params),
                                     "val_balanced_accuracy": best["score"], **scores})
            print(f"finished {split_name} seed {seed}", flush=True)
    return pd.DataFrame(rows)


def summarise(runs):
    metrics = ["accuracy", "balanced_accuracy", "f1_democratic"]
    agg = runs.groupby(["split", "seed_frac", "graph", "model"], sort=False)[metrics].agg(["mean", "std"])
    for (split_name, frac), g in agg.groupby(level=["split", "seed_frac"], sort=False):
        print(f"\n== {split_name} split, {frac:.0%} of training counties known, "
              f"{runs.seed.nunique()} seeds (mean +- std over seeds)")
        print(f"{'graph':12s} {'model':18s} {'bal. acc':>15s} {'accuracy':>15s} {'F1 (Dem)':>15s}")
        for (_, _, graph, model), r in g.iterrows():
            cell = lambda m: f"{r[(m, 'mean')]:.3f} +- {r[(m, 'std')]:.3f}"
            print(f"{graph:12s} {model:18s} {cell('balanced_accuracy'):>15s} {cell('accuracy'):>15s} {cell('f1_democratic'):>15s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of random seeds (default 10)")
    ap.add_argument("--splits", nargs="+", default=["stratified", "block"], choices=["stratified", "block"])
    ap.add_argument("--seed-frac", type=float, nargs="+", default=None, metavar="F",
                    help="share(s) of the training counties whose labels are known, e.g. 0.1 0.5 1.0 "
                         "(default 1.0; with --previous-year, default 0.05 0.1 0.25 0.5 1.0)")
    ap.add_argument("--previous-year", action="store_true", help="also run the previous-year seeding experiment")
    args = ap.parse_args()

    runs = run_comparison(args.splits, range(args.seeds), args.seed_frac or (1.0,))
    path = run_path("model_comparison")
    runs.to_csv(path, index=False)
    summarise(runs)
    print(f"\nsaved {path}")

    if args.previous_year:
        for split_name in args.splits:
            run_repeated(split_name, seeds=range(args.seeds), **({"fracs": args.seed_frac} if args.seed_frac else {}))
        print("previous-year runs saved to results/runs/")
