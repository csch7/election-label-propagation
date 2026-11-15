"""Label spreading (Zhou et al., 2004) over a weighted graph.

Each node holds a score for [Republican, Democratic]. Training nodes start at
their known label, everything else at zero. Repeatedly:

    F <- alpha * S @ F + (1 - alpha) * Y,     S = D^-1/2 W D^-1/2

so each node averages its neighbours' scores (labels travel over many hops)
while being pulled back towards its own initial label with weight 1 - alpha.
A node's predicted probability is its normalised score.
"""
import numpy as np

from ..config import load_config


class LabelPropagation:
    def __init__(self, alpha=None, max_iter=None, tol=None, class_mass_normalization=False):
        """alpha in (0, 1): how far labels spread (higher = further); max_iter and
        tol are the stopping rule. All default to the lp_* values in the config.
        class_mass_normalization rescales each class's scores to its training
        share, which counters the pull towards the majority class (only ~14%
        of counties are Democratic)."""
        cfg = load_config()
        self.alpha = cfg["lp_alpha"] if alpha is None else alpha
        self.max_iter = cfg["lp_max_iter"] if max_iter is None else max_iter
        self.tol = cfg["lp_tol"] if tol is None else tol
        self.class_mass_normalization = class_mass_normalization

    def fit(self, W, y, train_mask):
        """W: (n, n) symmetric non-negative weights; y: (n,) 0/1 labels (nan
        allowed where unknown); train_mask: (n,) bool of labelled nodes."""
        W = np.clip(np.asarray(W, dtype=np.float32), 0, None)
        deg = W.sum(1)
        inv_sqrt = np.where(deg > 0, 1 / np.sqrt(np.where(deg > 0, deg, 1)), 0).astype(np.float32)
        S = W * inv_sqrt[:, None] * inv_sqrt[None, :]

        train_mask = np.asarray(train_mask, dtype=bool)
        y = np.asarray(y, dtype=float)
        Y = np.zeros((len(W), 2), dtype=np.float32)
        Y[train_mask, 0] = y[train_mask] == 0
        Y[train_mask, 1] = y[train_mask] == 1

        F = Y.copy()
        for self.n_iter_ in range(1, self.max_iter + 1):
            F_new = self.alpha * (S @ F) + (1 - self.alpha) * Y
            delta = np.abs(F_new - F).max()
            F = F_new
            if delta < self.tol:
                break

        self.prior_ = float(y[train_mask].mean())
        self.scores_ = F
        return self

    def predict_proba(self, idx):
        """P(Democratic) for node indices idx. Nodes no label can reach (zero
        scores) get the training prior; see `covered`."""
        F = self.scores_[np.asarray(idx)].astype(float)
        if self.class_mass_normalization:
            share = np.array([1 - self.prior_, self.prior_])
            F = F * share / np.maximum(self.scores_.sum(0), 1e-12)
        total = F.sum(1)
        self.covered = total > 1e-12
        return np.where(self.covered, F[:, 1] / np.where(self.covered, total, 1), self.prior_)

    def predict(self, idx, threshold=0.5):
        return (self.predict_proba(idx) > threshold).astype(int)
