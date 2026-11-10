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

from .adjacency import DATA_DIR

ROOT = DATA_DIR.parent
CENSUS_ZIP = DATA_DIR / "Census Data.zip"

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


def build_demographic_graph(features=None, method="knn", k=None, threshold=None):
    """Return (geoids, A, S) with A a symmetric 0/1 adjacency matrix.

    method="knn":       edge i-j if j is in i's top-k most similar OR vice versa.
    method="threshold": edge i-j if similarity >= threshold.
    k / threshold default to the values in config/config.yml.
    """
    cfg = load_config()
    features = build_features() if features is None else features
    S = similarity_matrix(features)
    if method == "knn":
        k = cfg["nearest_neighbors"] if k is None else k
        top = np.argpartition(-S, k, axis=1)[:, :k]
        A = np.zeros(S.shape, dtype=np.uint8)
        A[np.arange(len(S))[:, None], top] = 1
        A = np.maximum(A, A.T)
    elif method == "threshold":
        threshold = cfg["demographic_similarity"] if threshold is None else threshold
        A = (S >= threshold).astype(np.uint8)
    else:
        raise ValueError("method must be 'knn' or 'threshold'")
    np.fill_diagonal(A, 0)
    return features.index.to_numpy(), A, S


if __name__ == "__main__":
    for m in ("knn", "threshold"):
        geoids, A, _ = build_demographic_graph(method=m)
        print(f"{m}: {len(geoids)} nodes, {int(A.sum() // 2)} edges, mean degree {A.sum(1).mean():.2f}")
