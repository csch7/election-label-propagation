"""Reproduce individual runs from results/runs so their county-level
predictions can be mapped. A run is identified by its row in the model
comparison CSV; it is refitted with the same split, known counties and tuned
parameters, and the result is checked against the saved metrics."""
import ast

import numpy as np
import pandas as pd

from main import MODELS, load_graphs
from ..evaluation.metrics import balanced_accuracy
from ..evaluation.splits import eligible_counties, known_mask, make_split, node_arrays


class Reproducer:
    def __init__(self):
        self.df = eligible_counties()
        self.graphs = load_graphs()

    def split(self, strategy, seed):
        return make_split(strategy, self.df, seed)

    def run(self, row):
        """Return a DataFrame indexed by test-county GEOID with the actual
        winner (`actual`, 1 = Democratic), predicted probability (`proba`) and
        tuned-threshold prediction (`pred`)."""
        geoids, sources = self.graphs[row.graph]
        cls, _, source = MODELS[row.model]
        params = ast.literal_eval(row.params)
        threshold, lam = params.pop("threshold"), params.pop("lam", None)
        M = next(m for l, m in sources[source]() if l == lam)
        y, train, held_out = node_arrays(geoids, self.split(row.split, int(row.seed)))
        frac = None if row.train_frac == "max" else float(row.train_frac)
        known = known_mask(train, geoids, self.df, frac, seed=int(row.seed))
        test = (train | held_out) & ~known  # test set plus every pool county whose label is not known
        proba = cls(**params).fit(M, y, known).predict_proba(np.flatnonzero(test))
        actual = y[test].astype(int)
        pred = (proba > threshold).astype(int)
        saved = row.balanced_accuracy
        assert abs(balanced_accuracy(actual, pred) - saved) < 1e-6, "reproduced run does not match the saved metrics"
        out = pd.DataFrame({"actual": actual, "proba": proba, "pred": pred}, index=np.asarray(geoids)[test])
        out.attrs["known"] = list(np.asarray(geoids)[known])  # counties whose labels were known
        return out


def pick_run(comp, graph, model, split="stratified", seed=0, frac="max"):
    sel = comp[(comp.graph == graph) & (comp.model == model) & (comp.split == split)
               & (comp.seed == seed) & (comp.train_frac == frac)]
    assert len(sel) == 1, (graph, model, split, seed, frac, len(sel))
    return sel.iloc[0]
