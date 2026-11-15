"""Run the full comparison: kNN vs label propagation on the border graph and
the demographic graph, predicting county-level 2024 presidential winners.

    python main.py                 # 10 seeds, both split strategies
    python main.py --seeds 3 --splits stratified
    python main.py --train-frac 0.1 0.4 max   # share of all counties whose labels are known
    python main.py --previous-year # also run the previous-year seeding experiment

For every (split strategy, random seed, graph, model):
  1. hold out 20% of the counties as the test set (counties whose labels are not
     known, i.e. the unused part of the training pool, are scored too),
  2. choose the known (training) counties: --train-frac of ALL counties, sampled
     from the remaining 80% (default: all of them),
  3. tune the model's parameters and decision threshold by 5-fold cross-validation
     among the known counties,
  4. fit with all known counties' labels and score once on the test counties.
kNN uses all counties' similarity (demographic) or geographic distance (border);
label propagation spreads labels along the graph's edges.
Results are saved to results/runs/<date>_model_comparison.csv and summarised
as mean +- std over seeds. Metrics treat Democratic as the positive class.
"""
import argparse

import numpy as np
import pandas as pd

from src.config import load_config, parse_fracs
from src.evaluation.metrics import report
from src.evaluation.results import run_path
from src.evaluation.splits import (STRATEGIES, eligible_counties, fold_array, known_mask, make_folds,
                                   make_split, node_arrays)
from src.evaluation.tuning import evaluate, tune
from src.experiments.previous_year import run_repeated
from src.graphs.adjacency import geographic_similarity, get_adjacency_graph
from src.graphs.combined import blend_edges, blend_similarity, build_components
from src.graphs.demographics import get_demographic_graph, get_similarity_matrix
from src.models.knn import KNNClassifier
from src.models.label_propagation import LabelPropagation

cfg = load_config()

# name -> (model class, parameter grid, which matrix it runs on); grids are set in config.yml
#   "similarity": similarity between every pair of counties (demographic: cosine
#                 similarity of features; border: geographic proximity), so kNN
#                 always finds neighbours even when few counties are labelled
#   "graph":      the graph's own edges, over which labels are spread
MODELS = {
    "knn": (KNNClassifier, {"k": cfg["knn_k_grid"], "weighted": cfg["knn_weighted_grid"]}, "similarity"),
    "label_propagation": (LabelPropagation, {"alpha": cfg["lp_alpha_grid"]}, "graph"),
}


def load_graphs():
    """name -> (node ids, {"graph": candidates, "similarity": candidates}), where
    candidates() yields (lam, matrix) pairs. lam is None except for the combined
    graph, which yields one blended matrix per lambda in the config (the best is
    picked by cross-validation along with the other parameters)."""
    border_ids, A = get_adjacency_graph()
    demo_ids, W_demo = get_demographic_graph()
    sim_ids, S_demo = get_similarity_matrix()
    assert (sim_ids == demo_ids).all()
    parts = build_components()
    return {
        "border": (border_ids, {"graph": lambda: [(None, A)],
                                "similarity": lambda: [(None, geographic_similarity(border_ids))]}),
        "demographic": (demo_ids, {"graph": lambda: [(None, W_demo)],
                                   "similarity": lambda: [(None, S_demo)]}),
        "combined": (parts["geoids"], {
            "graph": lambda: ((lam, blend_edges(parts, lam)) for lam in cfg["blend_lambdas"]),
            "similarity": lambda: ((lam, blend_similarity(parts, lam)) for lam in cfg["blend_lambdas"])}),
    }


def run_comparison(splits, seeds, fracs=(None,)):
    df = eligible_counties()
    graphs = load_graphs()
    sim_cache = {}  # geographic similarity is slow enough to compute once per graph
    rows = []
    for split_name in splits:
        for seed in seeds:
            split = make_split(split_name, df, seed)
            for gname, (geoids, sources) in graphs.items():
                y, train, held_out = node_arrays(geoids, split)
                for frac in fracs:
                    known = known_mask(train, geoids, df, frac, seed=seed)
                    # scored counties: the held-out test set plus every pool county whose label is not known
                    # (at "max" every pool county is known, so this is just the test set)
                    test = (train | held_out) & ~known
                    ti = np.flatnonzero(test)
                    folds = fold_array(geoids, make_folds(split_name, df.loc[np.asarray(geoids)[known]], seed=seed))
                    base = {"split": split_name, "seed": seed, "graph": gname,
                            "train_frac": "max" if frac is None else frac,
                            "n_known": int(known.sum()), "n_test": len(ti)}
                    rows.append({**base, "model": "majority", "params": "",
                                 **report(y[ti].astype(int), np.zeros(len(ti), dtype=int), np.zeros(len(ti)))})
                    for mname, (cls, grid, source) in MODELS.items():
                        if source == "similarity" and gname == "border":
                            sim_cache.setdefault("border", sources["similarity"]())
                            candidates = sim_cache["border"]
                        else:
                            candidates = sources[source]()
                        best = None
                        for lam, M in candidates:
                            b, _ = tune(cls, M, y, folds, grid)
                            if best is None or b["score"] > best[0]["score"]:
                                best = (b, lam, M)
                        b, lam, M = best
                        scores = evaluate(cls, b, M, y, known, test)
                        params = {k: v for k, v in b.items() if k != "score"}
                        if lam is not None:
                            params["lam"] = lam
                        rows.append({**base, "model": mname, "params": str(params),
                                     "cv_balanced_accuracy": b["score"], **scores})
            print(f"finished {split_name} seed {seed}", flush=True)
    return pd.DataFrame(rows)


METRIC_COLUMNS = [("balanced_accuracy", "bal. acc"), ("accuracy", "accuracy"), ("precision", "precision"),
                  ("recall", "recall"), ("f1_democratic", "F1"), ("auc", "AUC")]


def summarise(runs):
    """Print mean +- std over seeds. Precision, recall, F1 and accuracy use the
    tuned decision threshold; AUC is threshold-free. Democratic is positive."""
    metrics = [m for m, _ in METRIC_COLUMNS]
    agg = runs.groupby(["split", "train_frac", "graph", "model"], sort=False)[metrics].agg(["mean", "std"])
    n_known = runs.groupby(["split", "train_frac"], sort=False).n_known.mean()
    for (split_name, frac), g in agg.groupby(level=["split", "train_frac"], sort=False):
        share = "the whole train pool" if frac == "max" else f"{float(frac):.0%} of all counties"
        print(f"\n== {split_name} split, {share} known (~{n_known[(split_name, frac)]:.0f} counties), "
              f"{runs.seed.nunique()} seeds (mean +- std over seeds)")
        print(f"{'graph':12s} {'model':18s}" + "".join(f"{name:>15s}" for _, name in METRIC_COLUMNS))
        for (_, _, graph, model), r in g.iterrows():
            cells = "".join(f"{r[(m, 'mean')]:>8.3f}+-{r[(m, 'std')]:.2f}" for m in metrics)
            print(f"{graph:12s} {model:18s}{cells}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=cfg["seeds"], help="number of random seeds (config: seeds)")
    ap.add_argument("--splits", nargs="+", default=cfg["split_strategies"], choices=list(STRATEGIES),
                    help="test-split types (config: split_strategies)")
    ap.add_argument("--train-frac", nargs="+", default=None, metavar="F",
                    help="share(s) of ALL counties whose labels are known (the training counties; at most "
                         "1 - test_fraction), e.g. 0.1 0.4 max. 'max' = every non-test county "
                         "(config: train_fractions, or previous_year_train_fractions with --previous-year)")
    ap.add_argument("--previous-year", action="store_true", help="also run the previous-year seeding experiment")
    args = ap.parse_args()

    fracs = parse_fracs(args.train_frac) if args.train_frac else None
    runs = run_comparison(args.splits, range(args.seeds), fracs or parse_fracs(cfg["train_fractions"]))
    path = run_path("model_comparison")
    runs.to_csv(path, index=False)
    summarise(runs)
    print(f"\nsaved {path}")

    if args.previous_year:
        for split_name in args.splits:
            run_repeated(split_name, seeds=range(args.seeds), **({"fracs": fracs} if fracs else {}))
        print("previous-year runs saved to results/runs/")
