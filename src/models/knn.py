"""k-nearest-neighbour classifier over a similarity matrix.

A test county is predicted from the labels of the k labelled (training)
counties most similar to it, voting weighted by similarity (or equally). The
matrix covers every pair of counties, so a county always has neighbours even
when few counties are labelled:
  * demographic graph: cosine similarity of the demographic features;
  * border graph: geographic proximity of county centres (nearest counties,
    whether or not they share a border).
Only direct similarity is used, unlike label propagation, which also spreads
labels through unlabelled counties over multiple hops.
"""
import numpy as np

from ..graphs.demographics import load_config


class KNNClassifier:
    def __init__(self, k="config", weighted=True):
        """k: number of voters (default `nearest_neighbors` from the config;
        None = every connected training county). Ties at the k-th weight are
        all kept. weighted: votes count by edge weight, otherwise equally."""
        self.k = load_config()["nearest_neighbors"] if k == "config" else k
        self.weighted = weighted

    def fit(self, W, y, train_mask):
        """W: (n, n) weights over nodes; y: (n,) 0/1 labels (nan allowed
        where unknown); train_mask: (n,) bool of labelled training nodes."""
        self.W = np.asarray(W, dtype=float)
        self.train_idx = np.flatnonzero(train_mask)
        self.y_train = np.asarray(y, dtype=float)[self.train_idx]
        self.prior_ = self.y_train.mean()
        return self

    def predict_proba(self, idx):
        """P(Democratic) for node indices idx. Counties with no connected
        training county get the training prior; see `covered`."""
        idx = np.asarray(idx)
        w = self.W[np.ix_(idx, self.train_idx)].copy()
        w[w < 0] = 0
        if self.k is not None and self.k < w.shape[1]:
            kth = -np.partition(-w, self.k - 1, axis=1)[:, self.k - 1:self.k]
            w = np.where(w >= kth, w, 0)
        if not self.weighted:
            w = (w > 0).astype(float)
        total = w.sum(1)
        self.covered = total > 0
        return np.where(self.covered, w @ self.y_train / np.where(total == 0, 1, total), self.prior_)

    def predict(self, idx, threshold=0.5):
        return (self.predict_proba(idx) > threshold).astype(int)
