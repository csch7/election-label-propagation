"""Make the project's figures from the saved experiment runs.

    python make_figures.py                 # 3 figures -> results/figures/
    python make_figures.py --only charts   # charts only (no map)
    python make_figures.py --only maps --seed 3
    python make_figures.py --only maps --train-frac 0.05 max   # maps for other training sizes

Figures (for the split set by report_split in config/config.yml):
  learning_curves.png  balanced accuracy and AUC against the number of known counties
  metrics_heatmap.png  all metrics for every graph and model
  map_errors.png       right and wrong calls on the test counties, for every model and graph
                       (map_errors_5pct.png etc. for lower training sizes, via --train-frac)
Charts read the latest results/runs/*_model_comparison.csv. The map refits the
runs for one seed (default 0); each refit is checked against the saved metrics.
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
    ap.add_argument("--only", choices=["charts", "maps"], help="make only the charts or only the map")
    ap.add_argument("--seed", type=int, default=0, help="seed whose runs the map shows (default 0)")
    ap.add_argument("--train-frac", nargs="+", default=["max"], metavar="F",
                    help="training sizes to map, as in the results (e.g. 0.05 0.1 max); default max")
    ap.add_argument("--dots", action="store_true", help="draw counties as dots (no boundary download)")
    args = ap.parse_args()

    split = load_config()["report_split"]
    style.apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    comp = load_comparison()

    if args.only != "maps":
        charts.learning_curves(comp, OUT / "learning_curves.png", split)
        charts.metrics_heatmap(comp, OUT / "metrics_heatmap.png", split)
        print("charts done")

    if args.only != "charts":
        from src.visualization import maps
        from src.visualization.predictions import Reproducer

        canvas = maps.CountyCanvas(None if args.dots else maps.load_atlas())
        repro = Reproducer()
        for frac in args.train_frac:
            name = "map_errors.png" if frac == "max" else f"map_errors_{round(float(frac) * 100)}pct.png"
            maps.errors_grid(canvas, repro, comp, OUT / name, split=split, seed=args.seed, frac=frac)
        print("map done")
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
