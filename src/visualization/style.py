"""Shared colours, labels and matplotlib defaults for the figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GRAPHS = ["border", "demographic", "combined"]
MODELS = ["knn", "label_propagation"]
GRAPH_COLOR = {"border": "#d2742d", "demographic": "#087d72", "combined": "#674385"}
GRAPH_LABEL = {"border": "Border graph", "demographic": "Demographic graph", "combined": "Combined graph"}
MODEL_LABEL = {"knn": "kNN", "label_propagation": "Label propagation"}
MODEL_STYLE = {"knn": dict(ls="-", marker="o"), "label_propagation": dict(ls="--", marker="s")}
SPLIT_LABEL = {"stratified": "Stratified split", "block": "Block split"}
METRIC_LABEL = {"balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "f1_democratic": "F1 (Democratic)", "auc": "AUC"}

INK = "#192d3b"
MUTED = "#506674"
DEM, REP = "#2c6fbb", "#c4473f"
PALE_DEM, PALE_REP = "#bdd2ea", "#ecc3bf"
NEUTRAL = "#c9d3d8"
BLANK = "#f0f2f3"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#8a98a0", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e3e8ea", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
    })


def headline(fig, title, subtitle=None, y=0.975):
    """Title and subtitle at the top left; the gap is fixed in inches so it works for any figure height."""
    fig.text(0.04, y, title, fontsize=20, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.04, y - 0.5 / fig.get_figheight(), subtitle, fontsize=11.5, color=MUTED, va="top")
