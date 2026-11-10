"""Graph of bordering counties (nodes = county GEOIDs, edge = shared border)."""
import numpy as np
import pandas as pd

from .graph_io import PREPROCESSED_DIR, RAW_DIR, load_graph, save_graph

ADJACENCY_PATH = PREPROCESSED_DIR / "adjacency_graph.npz"


def load_counties(path=RAW_DIR / "US Counties.csv"):
    """All counties, indexed by 5-digit GEOID string."""
    return pd.read_csv(path, dtype={"GEOID": str, "STATEFP": str, "COUNTYFP": str}).set_index("GEOID")


def build_adjacency_graph(counties=None, border_path=RAW_DIR / "Bordering Counties.csv"):
    """Return (geoids, A): geoids is the ordered node list, A a symmetric
    0/1 uint8 adjacency matrix with a zero diagonal.

    The source file lists each border once, so it is symmetrized here.
    Counties with no neighbours (e.g. islands) stay as isolated nodes.
    """
    counties = load_counties() if counties is None else counties
    geoids = counties.index.to_numpy()
    idx = {g: i for i, g in enumerate(geoids)}

    edges = pd.read_csv(border_path, dtype=str)
    edges = edges[edges.county1.isin(idx) & edges.county2.isin(idx)]
    i = edges.county1.map(idx).to_numpy()
    j = edges.county2.map(idx).to_numpy()

    A = np.zeros((len(geoids), len(geoids)), dtype=np.uint8)
    A[i, j] = 1
    A[j, i] = 1
    np.fill_diagonal(A, 0)
    return geoids, A


def get_adjacency_graph(rebuild=False, path=ADJACENCY_PATH):
    """Load the saved border graph, building and saving it first if needed.
    Returns (geoids, A)."""
    if rebuild or not path.exists():
        geoids, A = build_adjacency_graph()
        save_graph(path, geoids, A)
    return load_graph(path)


if __name__ == "__main__":
    geoids, A = build_adjacency_graph()
    save_graph(ADJACENCY_PATH, geoids, A)
    deg = A.sum(1)
    print(f"{len(geoids)} nodes, {int(A.sum() // 2)} edges, "
          f"mean degree {deg.mean():.2f}, isolated {(deg == 0).sum()}")
    print(f"saved to {ADJACENCY_PATH}")
