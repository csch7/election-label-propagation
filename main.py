"""Run the full comparison: kNN vs label propagation on the border graph and
the demographic graph, predicting county-level 2024 presidential winners.

    python main.py                 # 10 seeds, both split strategies
    python main.py --seeds 3 --splits stratified
    python main.py --train-frac 0.1 0.4 max   # share of all counties whose labels are known
    python main.py --previous-year # also run the previous-year seeding experiment

For every (split strategy, random seed, graph, model):
  1. hold out 20% of the counties as the test set,
  2. choose the known (training) counties: --train-frac of ALL counties, sampled
     from the remaining 80% (default: all of them),
  3. tune the model's parameters and decision threshold by 5-fold cross-validation
     among the known counties,
  4. fit with all known counties' labels and score once on the test counties.
The "knn" model only sees direct graph connections; "knn_full" is the classic kNN
over all counties' similarity (demographic) or geographic distance (border).
Results are saved to results/runs/<date>_model_comparison.csv and summarised
as mean +- std over seeds. Metrics treat Democratic as the positive class.
"""
import argparse

import numpy as np
import pandas as pd

from src.evaluation.metrics import report
from src.evaluation.results import run_path
from src.evaluation.splits import eligible_counties, fold_array, known_mask, make_folds, make_split, node_arrays
from src.evaluation.tuning import evaluate, tune
from src.experiments.previous_year import run_repeated
from src.graphs.adjacency import geographic_similarity, get_adjacency_graph
from src.graphs.demographics import get_demographic_graph, get_similarity_matrix
from src.models.knn import KNNClassifier
from src.models.label_propagation import LabelPropagation

# name -> (model class, parameter grid, which matrix it runs on)
#   "graph": the graph's own edges (direct connections only)
#   "full":  similarity between every pair of counties (demographic: cosine
#            similarity of features; border: geographic proximity), so kNN always
#            finds neighbours even when few counties are labelled
MODELS = {
    "knn": (KNNClassifier, {"k": [1, 3, 5, 10, 20, None], "weighted": [True, False]}, "graph"),
    "knn_full": (KNNClassifier, {"k": [1, 3, 5, 10, 20, 50], "weighted": [True, False]}, "full"),
    "label_propagation": (LabelPropagation, {"alpha": [0.5, 0.8, 0.9, 0.95, 0.99]}, "graph"),
}


def run_comparison(splits, seeds, fracs=(None,)):
    df = eligible_counties()
    border_ids, A = get_adjacency_graph()
    demo_ids, W_demo = get_demographic_graph()
    sim_ids, S_demo = get_similarity_matrix()
    assert (sim_ids == demo_ids).all()
    graphs = {"border": (border_ids, A, geographic_similarity(border_ids)),
              "demographic": (demo_ids, W_demo, S_demo)}
    rows = []
    for split_name in splits:
        for seed in seeds:
            split = make_split(split_name, df, seed)
            for gname, (geoids, W, S_full) in graphs.items():
                y, train, test = node_arrays(geoids, split)
                ti = np.flatnonzero(test)
                for frac in fracs:
                    known = known_mask(train, geoids, df, frac, seed=seed)
                    folds = fold_array(geoids, make_folds(split_name, df.loc[np.asarray(geoids)[known]], seed=seed))
                    base = {"split": split_name, "seed": seed, "graph": gname,
                            "train_frac": "max" if frac is None else frac,
                            "n_known": int(known.sum()), "n_test": len(ti)}
                    rows.append({**base, "model": "majority", "params": "",
                                 **report(y[ti].astype(int), np.zeros(len(ti), dtype=int))})
                    for mname, (cls, grid, source) in MODELS.items():
                        M = W if source == "graph" else S_full
                        best, _ = tune(cls, M, y, folds, grid)
                        scores = evaluate(cls, best, M, y, known, test)
                        params = {k: v for k, v in best.items() if k != "score"}
                        rows.append({**base, "model": mname, "params": str(params),
                                     "cv_balanced_accuracy": best["score"], **scores})
            print(f"finished {split_name} seed {seed}", flush=True)
    return pd.DataFrame(rows)


def summarise(runs):
    metrics = ["accuracy", "balanced_accuracy", "f1_democratic"]
    agg = runs.groupby(["split", "train_frac", "graph", "model"], sort=False)[metrics].agg(["mean", "std"])
    n_known = runs.groupby(["split", "train_frac"], sort=False).n_known.mean()
    for (split_name, frac), g in agg.groupby(level=["split", "train_frac"], sort=False):
        share = "the whole train pool" if frac == "max" else f"{float(frac):.0%} of all counties"
        print(f"\n== {split_name} split, {share} known (~{n_known[(split_name, frac)]:.0f} counties), "
              f"{runs.seed.nunique()} seeds (mean +- std over seeds)")
        print(f"{'graph':12s} {'model':18s} {'bal. acc':>15s} {'accuracy':>15s} {'F1 (Dem)':>15s}")
        for (_, _, graph, model), r in g.iterrows():
            cell = lambda m: f"{r[(m, 'mean')]:.3f} +- {r[(m, 'std')]:.3f}"
            print(f"{graph:12s} {model:18s} {cell('balanced_accuracy'):>15s} {cell('accuracy'):>15s} {cell('f1_democratic'):>15s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of random seeds (default 10)")
    ap.add_argument("--splits", nargs="+", default=["stratified", "block"], choices=["stratified", "block"])
    ap.add_argument("--train-frac", nargs="+", default=None, metavar="F",
                    help="share(s) of ALL counties whose labels are known (the training counties; at most "
                         "0.8 since 20%% are held out for test), e.g. 0.1 0.4 max. 'max' = every non-test "
                         "county (default; with --previous-year the default is 0.05 0.1 0.2 0.4 max)")
    ap.add_argument("--previous-year", action="store_true", help="also run the previous-year seeding experiment")
    args = ap.parse_args()

    fracs = [None if f == "max" else float(f) for f in args.train_frac] if args.train_frac else None
    runs = run_comparison(args.splits, range(args.seeds), fracs or (None,))
    path = run_path("model_comparison")
    runs.to_csv(path, index=False)
    summarise(runs)
    print(f"\nsaved {path}")

    if args.previous_year:
        for split_name in args.splits:
            run_repeated(split_name, seeds=range(args.seeds), **({"fracs": fracs} if fracs else {}))
        print("previous-year runs saved to results/runs/")
