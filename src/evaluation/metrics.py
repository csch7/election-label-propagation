"""Classification metrics. y_true / y_pred are 0/1 arrays (1 = Democratic)."""
import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true, y_pred):
    """Mean recall over the two classes; not fooled by the Republican majority."""
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in (0, 1) if (y_true == c).any()]))


def f1_democratic(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    denom = 2 * tp + np.sum((y_pred == 1) & (y_true == 0)) + np.sum((y_pred == 0) & (y_true == 1))
    return float(2 * tp / denom) if denom else 0.0


def report(y_true, y_pred):
    return {"accuracy": accuracy(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "f1_democratic": f1_democratic(y_true, y_pred)}
