"""Train/test splits over counties.

A split has two parts: test (a fixed share, touched only at the end) and train,
the pool of counties whose labels are known. Parameters are tuned by k-fold
cross-validation inside the train pool (see make_folds and tuning.py). Both
graphs are used transductively: the full graph is always visible to the
model, and the labels of test counties are hidden. A split is therefore just a
list of train and test GEOIDs, shared by every model so results are comparable.
Counties without a label or without census features are never in the split.

Two strategies:
  stratified_split  random 80/20 that preserves each state's Dem/Rep ratio and
                    demographic profile (the default).
  block_split       test counties form contiguous blobs grown from random seeds,
                    so test counties are far from labelled ones (a harder test
                    that favours the demographic graph over the border graph).
"""
from collections import deque

import numpy as np
import pandas as pd

from ..graphs.adjacency import get_adjacency_graph, load_counties
from ..graphs.demographics import build_features
from ..graphs.graph_io import PREPROCESSED_DIR
from .labels import build_labels


def eligible_counties(features=None, labels=None):
    """Counties with both census features and a label, with state/label/features."""
    features = build_features() if features is None else features
    labels = build_labels() if labels is None else labels
    counties = load_counties()
    idx = features.index.intersection(labels.index).intersection(counties.index)
    df = features.loc[idx].copy()
    df["state"] = counties.loc[idx, "STUSPS"]
    df["label"] = labels.loc[idx]
    return df


def stratified_split(df=None, test_frac=0.2, seed=0):
    """Systematic stratified sample. Counties are sorted by (state, label,
    first principal component of the demographics) and cut into consecutive
    blocks of 1/test_frac; one random county per block becomes a test county.
    Every state, every state's label mix, and the demographic range within each
    are therefore represented in test in the same proportion as in train."""
    df = eligible_counties() if df is None else df
    rng = np.random.default_rng(seed)
    block = round(1 / test_frac)

    X = df.drop(columns=["state", "label"]).to_numpy(float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    pc1 = np.linalg.svd(X, full_matrices=False)[0][:, 0]
    order = df.assign(pc1=pc1).sort_values(["state", "label", "pc1"])

    test = []
    for _, g in order.groupby(["state", "label"], sort=False):
        ids = g.index.to_numpy()
        for s in range(0, len(ids), block):
            chunk = ids[s:s + block]
            # a short final chunk contributes a test county with proportional probability
            if len(chunk) == block or rng.random() < len(chunk) / block:
                test.append(rng.choice(chunk))
    return _as_split(df, test)


def block_split(df=None, test_frac=0.2, block_size=10, seed=0):
    """Grow contiguous test blocks of ~block_size counties by breadth-first
    search over the border graph from random seed counties, within each state,
    until each state has ~test_frac of its counties in test."""
    df = eligible_counties() if df is None else df
    rng = np.random.default_rng(seed)
    geoids, A = get_adjacency_graph()
    pos = {g: i for i, g in enumerate(geoids)}

    test = set()
    for state, g in df.groupby("state"):
        members = set(g.index)
        target = round(test_frac * len(members))
        chosen = set()
        while len(chosen) < target:
            free = sorted(members - chosen)
            seed_c = free[rng.integers(len(free))]
            queue, region = deque([seed_c]), {seed_c}
            cap = min(block_size, target - len(chosen))
            while queue and len(region) < cap:
                c = queue.popleft()
                nbrs = [geoids[j] for j in np.nonzero(A[pos[c]])[0]]
                for n in rng.permutation(nbrs):
                    if n in members and n not in chosen and n not in region and len(region) < cap:
                        region.add(n)
                        queue.append(n)
            chosen |= region
        test |= chosen
    return _as_split(df, sorted(test))


def pick_seeds(train_mask, geoids, df, seed_frac, seed=0):
    """Random state-stratified sample of seed_frac of the training counties."""
    if seed_frac >= 1:
        return train_mask.copy()
    train_ids = set(np.asarray(geoids)[train_mask])
    sub = df.loc[[g for g in df.index if g in train_ids]]
    chosen = stratified_split(sub, test_frac=seed_frac, seed=seed)
    chosen = set(chosen.index[chosen == "test"])
    return np.array([g in chosen for g in geoids]) & train_mask


def known_mask(train_mask, geoids, df, train_frac=None, seed=0):
    """Counties whose labels are known (the 'training' counties): train_frac is
    their share of ALL eligible counties (None = the whole train pool).
    Sampled state-stratified from the train pool."""
    if train_frac is None:
        return train_mask.copy()
    n_train = int(train_mask.sum())
    share_of_pool = train_frac * len(df) / n_train
    if share_of_pool > 1.001:
        raise ValueError(f"train_frac={train_frac} exceeds the train pool "
                         f"({n_train / len(df):.2f} of all counties)")
    return pick_seeds(train_mask, geoids, df, min(share_of_pool, 1.0), seed)


STRATEGIES = {"stratified": stratified_split, "block": block_split}


def make_split(strategy, df, seed=0, test_frac=0.2):
    """Fresh train/test split ('stratified' or 'block'). 'train' is the pool of
    counties whose labels may be used (training + validation)."""
    return STRATEGIES[strategy](df, test_frac=test_frac, seed=seed)


def make_folds(strategy, df, k=5, seed=0):
    """Assign the counties in df to k cross-validation folds (Series of 0..k-1)
    using the same strategy as the test split: stratified folds each mirror the
    pool's state / party / demographic mix; block folds are contiguous regions."""
    fn = STRATEGIES[strategy]
    fold = pd.Series(k - 1, index=df.index, name="fold")
    remaining = df
    for i in range(k - 1):
        held = fn(remaining, test_frac=1 / (k - i), seed=seed + i)
        ids = held.index[held == "test"]
        fold.loc[ids] = i
        remaining = remaining.drop(ids)
    return fold


def fold_array(geoids, folds):
    """Fold index per graph node (-1 for counties outside the folds)."""
    return folds.reindex(np.asarray(geoids)).fillna(-1).to_numpy(int)


def _as_split(df, test_ids):
    split = pd.Series("train", index=df.index, name="split")
    split.loc[list(test_ids)] = "test"
    return split


def node_arrays(geoids, split, labels=None):
    """Align labels and a split to a graph's node order. Returns (y, train_mask,
    test_mask): y is float with nan for counties without a label; the masks are
    boolean arrays over the graph's nodes (counties outside the split are in neither)."""
    labels = build_labels() if labels is None else labels
    geoids = np.asarray(geoids)
    y = labels.reindex(geoids).to_numpy(float)
    s = split.reindex(geoids)
    return y, (s == "train").to_numpy(), (s == "test").to_numpy()


def save_split(split, name):
    path = PREPROCESSED_DIR / f"split_{name}.csv"
    split.rename_axis("GEOID").reset_index().to_csv(path, index=False)
    return path


def load_split(name="stratified"):
    """Return a Series indexed by GEOID with values 'train' / 'test'."""
    return pd.read_csv(PREPROCESSED_DIR / f"split_{name}.csv", dtype={"GEOID": str}).set_index("GEOID")["split"]


def diagnostics(split, df=None):
    """Print how well test matches train, and how isolated test counties are."""
    df = eligible_counties() if df is None else df
    d = df.assign(split=split)
    te, tr = d[d.split == "test"], d[d.split == "train"]
    print(f"train {len(tr)}, test {len(te)} ({len(te) / len(d):.1%})")
    print(f"Dem share: train {tr.label.mean():.3f}, test {te.label.mean():.3f}")
    frac = d.groupby("state").split.apply(lambda s: (s == "test").mean())
    print(f"test fraction by state: min {frac.min():.2f}, max {frac.max():.2f}")
    feats = d.drop(columns=["state", "label", "split"])
    z = (te[feats.columns].mean() - tr[feats.columns].mean()) / feats.std()
    print(f"max |standardized feature mean gap|: {z.abs().max():.3f}")
    geoids, A = get_adjacency_graph()
    pos = {g: i for i, g in enumerate(geoids)}
    train_mask = np.zeros(len(geoids), bool)
    train_mask[[pos[g] for g in tr.index if g in pos]] = True
    tpos = [pos[g] for g in te.index if g in pos]
    deg = A[tpos].sum(1)
    lab = (A[tpos] * train_mask).sum(1)
    print(f"test counties' border neighbours that are labelled train: {lab.sum() / deg.sum():.1%}")


if __name__ == "__main__":
    df = eligible_counties()
    for name in STRATEGIES:
        split = make_split(name, df)
        print(f"\n== {name} -> {save_split(split, name).name}")
        diagnostics(split, df)
