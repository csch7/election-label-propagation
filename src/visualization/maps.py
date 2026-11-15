"""County maps. Uses county boundaries (US Atlas, downloaded once and cached
in results/) when available, else draws each county as a dot at its centre."""
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath

from ..graphs.adjacency import load_counties
from ..graphs.graph_io import ROOT
from . import style as st
from .predictions import Reproducer, pick_run

ATLAS_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/counties-albers-10m.json"
ATLAS_CACHE = ROOT / "results" / "counties-albers-10m.json"


def load_atlas(download=True):
    """The US Atlas county boundaries, or None if unavailable."""
    if ATLAS_CACHE.exists():
        return json.loads(ATLAS_CACHE.read_text())
    if not download:
        return None
    try:
        with urlopen(ATLAS_URL, timeout=60) as response:
            data = response.read()
        atlas = json.loads(data)
        ATLAS_CACHE.write_bytes(data)
        return atlas
    except Exception as error:  # offline or blocked: fall back to dots
        print(f"Could not fetch county boundaries ({error}); drawing dot maps instead.")
        return None


def _geometry_patch(geometry, arcs):
    """Join TopoJSON arcs into closed rings, retaining islands and holes."""
    polygons = geometry["arcs"] if geometry["type"] == "MultiPolygon" else [geometry["arcs"]]
    paths = []
    for polygon in polygons:
        for ring in polygon:
            parts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in ring]
            points = np.vstack([parts[0]] + [part[1:] for part in parts[1:]])
            paths.append(PlotPath(points, closed=True))
    return PathPatch(PlotPath.make_compound_path(*paths))


def _albers(lon, lat):
    """Albers equal-area conic for the contiguous US; Hawaii is moved next to the Southwest."""
    lon, lat = np.asarray(lon, float).copy(), np.asarray(lat, float).copy()
    hawaii = (lon < -150) & (lat < 25)
    lon[hawaii] += 51
    lat[hawaii] += 5
    p1, p2, p0, l0 = np.radians([29.5, 45.5, 37.5, -96.0])
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(c - 2 * n * np.sin(p0)) / n
    rho = np.sqrt(c - 2 * n * np.sin(np.radians(lat))) / n
    theta = n * (np.radians(lon) - l0)
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


class CountyCanvas:
    """Colours counties on an axes: polygons if an atlas is given, else dots."""

    def __init__(self, atlas=None):
        self.atlas = atlas
        if atlas:
            t = atlas["transform"]
            self.arcs = [np.cumsum(a, axis=0) * t["scale"] + t["translate"] for a in atlas["arcs"]]
            self.shapes = atlas["objects"]["counties"]["geometries"]
            self.patches = [_geometry_patch(g, self.arcs) for g in self.shapes]
            self.ids = [g["id"] for g in self.shapes]
            self.states = [_geometry_patch(g, self.arcs) for g in atlas["objects"]["states"]["geometries"]]
        else:
            c = load_counties()
            c = c[~c.STUSPS.isin(["AK", "PR", "AS", "MP", "VI", "GU"])]  # no election data; keeps the map compact
            self.ids = list(c.index)
            self.x, self.y = _albers(c.longitude, c.latitude)

    def draw(self, ax, colors, default=st.BLANK, size=None):
        """colors: dict GEOID -> colour; every other county gets `default`."""
        if self.atlas:
            fills = [colors.get(i, default) for i in self.ids]
            ax.add_collection(PatchCollection(self.patches, facecolors=fills, edgecolors="white", linewidths=0.15))
            ax.add_collection(PatchCollection(self.states, facecolors="none", edgecolors="#647786", linewidths=0.4))
            ax.autoscale_view(tight=True)
            ax.invert_yaxis()
        else:
            fills = [colors.get(i, default) for i in self.ids]
            order = np.argsort([i in colors for i in self.ids], kind="stable")  # coloured counties on top
            ax.scatter(self.x[order], np.asarray(self.y)[order], s=size or 10, c=[fills[i] for i in order], linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.01)


def _legend(fig, entries, y=0.02, ncol=None):
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec="#89969c", mew=0.5, label=l) for l, c in entries]
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(entries), bbox_to_anchor=(0.5, y), fontsize=10.5)


def splits_map(canvas, repro, out, seed=0):
    """Where the training and test counties are, for each split type."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    size = 7 if canvas.atlas is None else None
    for ax, strategy in zip(axes, ("stratified", "block")):
        split = repro.split(strategy, seed)
        winner = repro.df.label
        colors = {}
        for g, s in split.items():
            dem = winner[g] == 1
            colors[g] = (st.DEM if dem else st.REP) if s == "test" else (st.PALE_DEM if dem else st.PALE_REP)
        canvas.draw(ax, colors, size=size)
        n_test = int((split == "test").sum())
        ax.set_title(f"{st.SPLIT_LABEL[strategy]}: {n_test} test counties", loc="left")
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.01, right=0.99, wspace=0.02)
    _legend(fig, [("Democratic, test", st.DEM), ("Republican, test", st.REP),
                  ("Democratic, known (train)", st.PALE_DEM), ("Republican, known (train)", st.PALE_REP),
                  ("No data", st.BLANK)])
    st.headline(fig, "Which counties are held out?", "2024 winner by county; bold colours are the counties we try to predict (seed 0)")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def errors_grid(canvas, repro, comp, out, split="stratified", seed=0, frac="max"):
    """Right and wrong calls on the test counties for every model and graph."""
    cats = {"ok": ("Correct", st.NEUTRAL), "miss": ("Democratic county called Republican", st.DEM),
            "false": ("Republican county called Democratic", st.REP)}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4))
    size = 5 if canvas.atlas is None else None
    for r, model in enumerate(st.MODELS):
        for c, graph in enumerate(st.GRAPHS):
            ax = axes[r, c]
            res = repro.run(pick_run(comp, graph, model, split, seed, frac))
            kind = np.where(res.actual == res.pred, "ok", np.where(res.actual == 1, "miss", "false"))
            canvas.draw(ax, {g: cats[k][1] for g, k in zip(res.index, kind)}, size=size)
            ax.set_title(f"{st.MODEL_LABEL[model]} · {st.GRAPH_LABEL[graph].replace(' graph', '')}", loc="left", fontsize=12)
            ax.text(0.02, 0.04, f"{(kind == 'ok').mean():.0%} correct\n{(kind == 'miss').sum()} missed · {(kind == 'false').sum()} false",
                    transform=ax.transAxes, fontsize=9.5, color=st.MUTED)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.01, right=0.99, wspace=0.02, hspace=0.12)
    _legend(fig, [(l, c) for l, c in cats.values()] + [("Not in test set", st.BLANK)])
    st.headline(fig, "Where does each model go wrong?", f"{st.SPLIT_LABEL[split]}, seed {seed}, all training labels known; test counties only")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def agreement_map(canvas, repro, comp, out, split="stratified", seed=0, frac="max"):
    """Border graph against demographic graph: which one gets each county right."""
    colors = {"Demographics only": "#087d72", "Borders only": "#d2742d", "Both correct": "#9fb1bb", "Both wrong": "#674385"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    size = 7 if canvas.atlas is None else None
    for ax, model in zip(axes, st.MODELS):
        ok = {g: repro.run(pick_run(comp, g, model, split, seed, frac)).eval("actual == pred") for g in ("border", "demographic")}
        demo = ok["demographic"]
        border = ok["border"].reindex(demo.index)
        label = np.select([demo & ~border, border & ~demo, demo & border], list(colors)[:3], "Both wrong")
        canvas.draw(ax, {g: colors[l] for g, l in zip(demo.index, label)}, size=size)
        counts = pd.Series(label).value_counts()
        ax.set_title(st.MODEL_LABEL[model], loc="left")
        ax.text(0.02, 0.02, "   ".join(f"{k} {counts.get(k, 0)}" for k in colors), transform=ax.transAxes, fontsize=9.5, color=st.MUTED)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.01, right=0.99, wspace=0.02)
    _legend(fig, list(colors.items()) + [("Not in test set", st.BLANK)])
    st.headline(fig, "Borders or demographics?", "Which graph predicted each test county's 2024 winner correctly (same test counties for both)")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def probability_map(canvas, repro, comp, out, graph="combined", model="label_propagation", split="stratified", seed=0, frac="max"):
    """Actual winners next to the model's predicted probability of a Democratic win."""
    res = repro.run(pick_run(comp, graph, model, split, seed, frac))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    size = 7 if canvas.atlas is None else None
    canvas.draw(axes[0], {g: st.DEM if a else st.REP for g, a in res.actual.items()}, size=size)
    axes[0].set_title("Actual 2024 winner", loc="left")
    cmap = plt.get_cmap("RdBu")  # 0 = red, 1 = blue
    shade = res.proba.clip(0, 1) ** 0.5  # square root stretches the low end: most counties have a small probability
    canvas.draw(axes[1], {g: cmap(v) for g, v in shade.items()}, size=size)
    axes[1].set_title("Predicted chance of a Democratic win", loc="left")
    axes[1].text(0.02, 0.02, f"{st.MODEL_LABEL[model]} on the {graph} graph", transform=axes[1].transAxes, fontsize=9.5, color=st.MUTED)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.01, right=0.99, wspace=0.02)
    _legend(fig, [("Democratic", st.DEM), ("Republican", st.REP), ("Not in test set", st.BLANK)])
    st.headline(fig, "Predicted vs actual", f"Test counties only ({st.SPLIT_LABEL[split]}, seed {seed}); red = unlikely Democratic, blue = likely, white = about 25% (square-root scale)")
    fig.savefig(out, dpi=200)
    plt.close(fig)
