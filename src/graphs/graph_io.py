"""Shared paths and save/load helpers for graphs.

Graphs are stored as compressed .npz files holding the node ids and the
undirected edge list (each edge once), so they are small and load fast:

    geoids : (n,) str    county GEOIDs, in node order
    src, dst : (m,) int  node indices of each edge (src < dst)
    weight : (m,) float  edge weights (1.0 for unweighted graphs)
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"


def save_graph(path, geoids, W):
    """Save a symmetric weight/adjacency matrix W with node ids `geoids`."""
    W = np.asarray(W)
    src, dst = np.nonzero(np.triu(W, k=1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        geoids=np.asarray(geoids, dtype=str),
        src=src.astype(np.int32),
        dst=dst.astype(np.int32),
        weight=W[src, dst].astype(np.float32),
    )
    return path


def load_graph(path):
    """Load a graph saved by save_graph. Returns (geoids, W) with W a dense
    symmetric float32 matrix (zero = no edge)."""
    with np.load(path) as f:
        geoids, src, dst, weight = f["geoids"], f["src"], f["dst"], f["weight"]
    W = np.zeros((len(geoids), len(geoids)), dtype=np.float32)
    W[src, dst] = weight
    W[dst, src] = weight
    return geoids, W
