"""Graph of demographically similar counties.

Each county is described by its age distribution and race/ethnicity shares
from the Census county population estimates. Counties are linked either when
their cosine similarity exceeds a threshold, or to their k most similar
counties (symmetrized).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .graph_io import PREPROCESSED_DIR, RAW_DIR, ROOT, load_graph, save_graph

CENSUS_ZIP = RAW_DIR / "Census Data.zip"
FEATURES_PATH = PREPROCESSED_DIR / "demographic_features.csv"

# Race/ethnicity columns: sum over male + female.
# NH* = non-Hispanic by race alone (White, Black, Indian, Asian, Pacific), NHTOM = two+ races, H = Hispanic.
RACE_COLS = ["NHWA", "NHBA", "NHIA", "NHAA", "NHNA", "NHTOM", "H"]


def load_config(path=ROOT / "config" / "config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_features(year=None, zip_path=CENSUS_ZIP):
    """Return a DataFrame indexed by GEOID (str) with one row per county:
    age-group shares (AGE_1..AGE_18) and race/ethnicity shares (sum to 1 each).

    year: Census YEAR code; defaults to the most recent in the file.
    """
    cols = ["GEOID", "YEAR", "AGEGRP", "TOT_POP"] + [f"{r}_{s}" for r in RACE_COLS for s in ("MALE", "FEMALE")]
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, usecols=cols)
    year = df.YEAR.max() if year is None else year
    df = df[df.YEAR == year]
    df["GEOID"] = df.GEOID.astype(int).astype(str).str.zfill(5)

    total = df[df.AGEGRP == 0].set_index("GEOID")
    race = pd.DataFrame({r: total[f"{r}_MALE"] + total[f"{r}_FEMALE"] for r in RACE_COLS})
    race = race.div(total.TOT_POP, axis=0)

    age = df[df.AGEGRP > 0].pivot(index="GEOID", columns="AGEGRP", values="TOT_POP")
    age = age.div(age.sum(axis=1), axis=0)
    age.columns = [f"AGE_{c}" for c in age.columns]

    return pd.concat([age, race], axis=1).dropna()


def similarity_matrix(features):
    """Cosine similarity between z-scored feature rows (diagonal zeroed)."""
    X = features.to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    np.fill_diagonal(S, 0)
    return S


def build_demographic_graph(features=None, method="knn", k=None, threshold=None, weighted=True):
    """Return (geoids, W, S): W is a symmetric adjacency matrix, S the full
    similarity matrix.

    method="threshold": edge i-j if S[i, j] >= threshold.
    method="knn":       edge i-j if j is in i's top-k most similar OR vice versa.
    method="hybrid":    union of the two, so every county has at least k edges
                        while very similar counties are all connected.
    weighted: edge weights are the similarity (otherwise 0/1).

    k defaults to `demographic_graph_k` and threshold to `demographic_similarity`
    in config/config.yml. (`nearest_neighbors` is for the kNN classifier only.)
    """
    if method not in ("threshold", "knn", "hybrid"):
        raise ValueError("method must be 'threshold', 'knn' or 'hybrid'")
    cfg = load_config()
    features = build_features() if features is None else features
    S = similarity_matrix(features)
    A = np.zeros(S.shape, dtype=bool)
    if method in ("knn", "hybrid"):
        k = cfg["demographic_graph_k"] if k is None else k
        top = np.argpartition(-S, k, axis=1)[:, :k]
        A[np.arange(len(S))[:, None], top] = True
    if method in ("threshold", "hybrid"):
        threshold = cfg["demographic_similarity"] if threshold is None else threshold
        A |= S >= threshold
    A |= A.T
    np.fill_diagonal(A, False)
    W = np.where(A, np.clip(S, 0, None), 0.0) if weighted else A.astype(np.uint8)
    return features.index.to_numpy(), W, S


def get_similarity_matrix():
    """Full (n, n) cosine similarity between every pair of counties (diagonal 0),
    from the cached feature table. Returns (geoids, S). Used by the "full" kNN
    baseline, which is not restricted to the edges of the demographic graph."""
    if FEATURES_PATH.exists():
        features = pd.read_csv(FEATURES_PATH, dtype={"GEOID": str}).set_index("GEOID")
    else:
        features = build_features()
    return features.index.to_numpy(), similarity_matrix(features).astype(np.float32)


def graph_path(method="knn"):
    return PREPROCESSED_DIR / f"demographic_graph_{method}.npz"


def get_demographic_graph(method="knn", rebuild=False, **kwargs):
    """Load the saved demographic graph for `method`, building and saving it
    first if needed. Returns (geoids, W). Passing k/threshold/weighted via
    kwargs forces a rebuild (and overwrites the saved file)."""
    path = graph_path(method)
    if rebuild or kwargs or not path.exists():
        geoids, W, _ = build_demographic_graph(method=method, **kwargs)
        save_graph(path, geoids, W)
    return load_graph(path)


if __name__ == "__main__":
    features = build_features()
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH)
    print(f"saved features to {FEATURES_PATH}")
    for m in ("knn", "threshold", "hybrid"):
        geoids, W, _ = build_demographic_graph(features, method=m)
        save_graph(graph_path(m), geoids, W)
        deg = (W > 0).sum(1)
        print(f"{m}: {len(geoids)} nodes, {int(deg.sum() // 2)} edges, "
              f"mean degree {deg.mean():.1f}, isolated {(deg == 0).sum()} -> {graph_path(m).name}")
