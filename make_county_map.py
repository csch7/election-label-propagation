"""Place beside main.py and run: python make_county_map.py

Uses the repo's November 11 comparison and saved stratified split (seed 0).
Requires the project dependencies and matplotlib. Fetches the US Atlas county
boundaries on first use. Writes the figure to results/county_error_map.png.
"""
import ast
import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd

from src.evaluation.metrics import report
from src.evaluation.splits import load_split, node_arrays
from src.graphs.adjacency import get_adjacency_graph
from src.graphs.demographics import get_demographic_graph
from src.models.label_propagation import LabelPropagation

ROOT = Path(__file__).resolve().parent
ATLAS = "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/counties-albers-10m.json"
COLORS = {
    "Demographics only": "#087d72", "Shared borders only": "#d2742d",
    "Both correct": "#9fb1bb", "Both wrong": "#674385",
    "Outside test set": "#edf0f1",
}


def get_outcomes():
    runs = pd.read_csv(ROOT / "results/runs/2026-11-11_model_comparison.csv")
    selected = runs[(runs["split"] == "stratified") & (runs.seed == 0)
                    & (runs.train_frac == "max") & (runs.model == "label_propagation")]
    split = load_split("stratified")
    correct = {}
    for graph, loader in [("border", get_adjacency_graph), ("demographic", get_demographic_graph)]:
        row, = selected[selected.graph == graph].itertuples()
        params = ast.literal_eval(row.params)
        threshold = params.pop("threshold")
        ids, weights = loader()
        y, train, test = node_arrays(ids, split)
        model = LabelPropagation(**params).fit(weights, y, train)
        predicted = model.predict(np.flatnonzero(test), threshold=threshold)
        actual = y[test].astype(int)
        assert (train.sum(), test.sum()) == (row.n_known, row.n_test)
        for metric, value in report(actual, predicted).items():
            np.testing.assert_allclose(value, getattr(row, metric), atol=1e-8)
        correct[graph] = pd.Series(predicted == actual, index=ids[test])
    border, demo = correct["border"], correct["demographic"]
    assert set(border.index) == set(demo.index)
    demo = demo.reindex(border.index)
    labels = np.select([demo & ~border, border & ~demo, demo & border],
                       ["Demographics only", "Shared borders only", "Both correct"], "Both wrong")
    return pd.Series(labels, index=border.index)


def load_boundaries():
    cache = ROOT / "results/counties-albers-10m.json"
    if cache.exists():
        return json.loads(cache.read_text())
    with urlopen(ATLAS, timeout=60) as response:
        data = response.read()
    atlas = json.loads(data)
    cache.write_bytes(data)
    return atlas


def geometry_patch(geometry, arcs):
    """Join TopoJSON arcs into closed rings, retaining islands and holes."""
    polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
    paths = []
    for polygon in polygons:
        for ring in polygon:
            parts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in ring]
            points = np.vstack([parts[0]] + [part[1:] for part in parts[1:]])
            paths.append(PlotPath(points, closed=True))
    return PathPatch(PlotPath.make_compound_path(*paths))


def draw_map(outcomes, atlas):
    transform = atlas["transform"]
    arcs = [np.cumsum(a, axis=0) * transform["scale"] + transform["translate"]
            for a in atlas["arcs"]]
    counties = atlas["objects"]["counties"]["geometries"]
    missing = set(outcomes.index) - {g["id"] for g in counties}
    if missing:
        raise ValueError(f"Missing county shapes for {sorted(missing)}")
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_axes([0.035, 0.18, 0.93, 0.65], aspect="equal")
    patches = [geometry_patch(g, arcs) for g in counties]
    fills = [COLORS[outcomes.get(g["id"], "Outside test set")] for g in counties]
    ax.add_collection(PatchCollection(patches, facecolors=fills, edgecolors="white", linewidths=0.15))
    states = [geometry_patch(g, arcs) for g in atlas["objects"]["states"]["geometries"]]
    ax.add_collection(PatchCollection(states, facecolors="none", edgecolors="#647786", linewidths=0.42))
    ax.autoscale_view(tight=True)
    ax.margins(0.005)
    ax.invert_yaxis()
    ax.axis("off")

    ink = "#192d3b"
    fig.text(0.055, 0.934, "Where did each model get the winner right?",
             fontsize=25, weight="bold", color=ink)
    fig.text(0.056, 0.890, f"2024 presidential election  •  Same {len(outcomes)} test counties for both models",
             fontsize=13, color="#415764")
    fig.text(0.056, 0.850, "Each color shows which version of label propagation predicted the county winner.",
             fontsize=11, color="#415764")

    counts = outcomes.value_counts()
    for x, (label, color) in zip([0.057, 0.275, 0.50, 0.66, 0.815], COLORS.items()):
        ax_dot = fig.add_axes([x, 0.115, 0.013, 0.021], aspect="equal")
        ax_dot.add_patch(Circle((0.5, 0.5), 0.42, facecolor=color,
                                edgecolor="#89969c", linewidth=0.4))
        ax_dot.axis("off")
        fig.text(x + 0.018, 0.126, label, va="center", fontsize=10.5, color=ink)
        if label != "Outside test set":
            fig.text(x + 0.018, 0.089, f"{counts.get(label, 0)} counties", fontsize=11,
                     color=ink, weight="bold")
    fig.text(0.056, 0.035, "Stratified split, seed 0  •  All training labels used  •  Boundaries: US Atlas / U.S. Census Bureau",
             fontsize=9, color="#506674")
    path = ROOT / "results/county_error_map.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print(outcomes.value_counts().to_string(), f"\nSaved {path}")


if __name__ == "__main__":
    draw_map(get_outcomes(), load_boundaries())
