"""Make the project's figures from the saved experiment runs.

    python make_figures.py                 # all figures -> results/figures/
    python make_figures.py --only charts   # charts only (no maps)
    python make_figures.py --only maps --seed 3

Charts read the latest results/runs/*_model_comparison.csv (and the previous-year
runs). Maps refit the runs for one seed (default 0) so their county-level
predictions can be drawn; each refit is checked against the saved metrics.
County boundaries come from the US Atlas (downloaded once, cached in results/);
use --dots to skip the download and draw each county as a dot instead.
Needs matplotlib in addition to the packages in requirements.txt.
"""
import argparse

from src.config import load_config
from src.graphs.graph_io import ROOT
from src.visualization import charts, style
from src.visualization.data import load_comparison

OUT = ROOT / "results" / "figures"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["charts", "maps"], help="make only charts or only maps")
    ap.add_argument("--seed", type=int, default=0, help="seed whose runs the maps show (default 0)")
    ap.add_argument("--dots", action="store_true", help="draw counties as dots (no boundary download)")
    args = ap.parse_args()

    style.apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    comp = load_comparison()

    if args.only != "maps":
        charts.learning_curves(comp, OUT / "learning_curves.png")
        charts.metrics_heatmap(comp, OUT / "metrics_heatmap.png")
        charts.tuned_parameters(comp, OUT / "tuned_parameters.png")
        charts.degree_distributions(OUT / "graph_degrees.png")
        if charts.previous_year(OUT / "previous_year.png"):
            print("previous-year chart done")
        print("charts done")

    if args.only != "charts":
        from src.visualization import maps
        from src.visualization.predictions import Reproducer

        canvas = maps.CountyCanvas(None if args.dots else maps.load_atlas())
        repro = Reproducer()
        maps.splits_map(canvas, repro, OUT / "map_splits.png", args.seed)
        maps.errors_grid(canvas, repro, comp, OUT / "map_errors.png", seed=args.seed)
        maps.agreement_map(canvas, repro, comp, OUT / "map_borders_vs_demographics.png", seed=args.seed)
        maps.probability_map(canvas, repro, comp, OUT / "map_predicted_vs_actual.png", seed=args.seed)
        print("maps done")
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
