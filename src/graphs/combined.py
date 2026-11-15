"""Combined graph: a blend of the border graph and the demographic graph.

    blend = lam * border + (1 - lam) * demographic        (lam = 0..1)

lam = 0 is the demographic graph and lam = 1 the border graph. Only counties in
both graphs (the ones with census data) are included, in the demographic graph's
order. The two sources are on different scales, so before blending:
  * graph edges (used by label propagation): each graph is rescaled so its
    average weighted degree is 1, so lam is the share of a county's total edge
    weight that comes from borders;
  * similarities (used by kNN): cosine similarity and geographic proximity are
    each standardised (zero mean, unit variance over all county pairs).
Counties with no borders (islands) still get edges from the demographic part.
"""
import numpy as np

from .adjacency import geographic_similarity, get_adjacency_graph
from .demographics import get_demographic_graph, get_similarity_matrix


def _standardise(S):
    off = ~np.eye(len(S), dtype=bool)
    Z = (S - S[off].mean()) / S[off].std()
    np.fill_diagonal(Z, 0)
    return Z.astype(np.float32)


def build_components():
    """Return a dict with the shared county ids and the four matrices to blend:
    border_edges, demographic_edges (for label propagation) and
    geographic_sim, demographic_sim (for kNN)."""
    ids, W_demo = get_demographic_graph()
    border_ids, A = get_adjacency_graph()
    pos = {g: i for i, g in enumerate(border_ids)}
    keep = np.array([pos[g] for g in ids])
    A = A[np.ix_(keep, keep)].astype(np.float32)
    sim_ids, S = get_similarity_matrix()
    assert (sim_ids == ids).all()
    return {
        "geoids": ids,
        "border_edges": A / A.sum(1).mean(),
        "demographic_edges": W_demo / W_demo.sum(1).mean(),
        "geographic_sim": _standardise(geographic_similarity(ids)),
        "demographic_sim": _standardise(S),
    }


def blend_edges(parts, lam):
    """Weighted graph for label propagation."""
    return lam * parts["border_edges"] + (1 - lam) * parts["demographic_edges"]


def blend_similarity(parts, lam):
    """Similarity matrix for kNN."""
    return lam * parts["geographic_sim"] + (1 - lam) * parts["demographic_sim"]
