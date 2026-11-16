"""County maps. Uses county boundaries (US Atlas, downloaded once and cached
in results/) when available, else draws each county as a dot at its centre."""
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
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


def errors_grid(canvas, repro, comp, out, split="stratified", seed=0, frac="max"):
    """Right and wrong calls on the test counties for every model and graph."""
    cats = {"ok": ("Called correctly", st.CORRECT), "miss": ("Democratic county called Republican", st.DEM),
            "false": ("Republican county called Democratic", st.REP)}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4))
    size = 5 if canvas.atlas is None else None
    for r, model in enumerate(st.MODELS):
        for c, graph in enumerate(st.GRAPHS):
            ax = axes[r, c]
            row = pick_run(comp, graph, model, split, seed, frac)
            res = repro.run(row)
            kind = np.where(res.actual == res.pred, "ok", np.where(res.actual == 1, "miss", "false"))
            colors = {g: st.KNOWN for g in res.attrs["known"]}
            colors.update({g: cats[k][1] for g, k in zip(res.index, kind)})
            canvas.draw(ax, colors, size=size)
            ax.set_title(f"{st.MODEL_LABEL[model]} · {st.GRAPH_LABEL[graph].replace(' graph', '')}", loc="left", fontsize=12)
            ax.text(1.0, 1.0, f"{(kind == 'ok').mean():.0%} correct · {(kind == 'miss').sum()} missed · {(kind == 'false').sum()} false",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=st.MUTED)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.01, right=0.99, wspace=0.02, hspace=0.12)
    _legend(fig, [("Label known (training)", st.KNOWN)] + [(l, c) for l, c in cats.values()] + [("No data", st.BLANK)])
    st.headline(fig, "Where does each model go wrong?",
                f"{st.SPLIT_LABEL[split]}, seed {seed}; labels known for {row.known_share:.0%} of counties; yellow = label known; the other counties are scored (the test set plus counties whose label is not known)")
    fig.savefig(out, dpi=200)
    plt.close(fig)
