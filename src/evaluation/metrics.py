"""Classification metrics. y_true / y_pred are 0/1 arrays (1 = Democratic,
the positive class); y_score is the predicted probability of Democratic."""
import numpy as np
import pandas as pd


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true, y_pred):
    """Mean recall over the two classes; not fooled by the Republican majority."""
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in (0, 1) if (y_true == c).any()]))


def _counts(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, fn


def precision(y_true, y_pred):
    """Of the counties predicted Democratic, the share that were (0 if none predicted)."""
    tp, fp, _ = _counts(y_true, y_pred)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall(y_true, y_pred):
    """Of the Democratic counties, the share found."""
    tp, _, fn = _counts(y_true, y_pred)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1_democratic(y_true, y_pred):
    tp, fp, fn = _counts(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def roc_auc(y_true, y_score):
    """Probability a random Democratic county is scored above a random
    Republican one (ties count half). Threshold-free; 0.5 = chance."""
    y_true = np.asarray(y_true)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(y_score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def report(y_true, y_pred, y_score=None):
    """All metrics. AUC uses y_score when given, else the hard predictions."""
    return {"accuracy": accuracy(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "precision": precision(y_true, y_pred),
            "recall": recall(y_true, y_pred),
            "f1_democratic": f1_democratic(y_true, y_pred),
            "auc": roc_auc(y_true, y_pred if y_score is None else y_score)}
