"""Chart figures built from the saved experiment runs (one split type)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import style as st


def _frac_order(df):
    return sorted(df.train_frac.unique(), key=lambda f: 1.0 if f == "max" else float(f))


def learning_curves(comp, out, split):
    """Balanced accuracy and AUC against the share of counties whose labels are known."""
    panels = [("balanced_accuracy", "Balanced accuracy", (0.5, 0.92)), ("auc", "AUC", (0.65, 0.97))]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    d = comp[(comp.split == split) & (comp.model != "majority")]
    for ax, (metric, label, ylim) in zip(axes, panels):
        for graph in st.GRAPHS:
            for model in st.MODELS:
                g = d[(d.graph == graph) & (d.model == model)]
                agg = g.groupby("train_frac").agg(x=("known_share", "mean"), m=(metric, "mean"), s=(metric, "std"))
                agg = agg.loc[_frac_order(g)]
                ax.plot(agg.x * 100, agg.m, color=st.GRAPH_COLOR[graph], lw=2, ms=5, **st.MODEL_STYLE[model])
                ax.fill_between(agg.x * 100, agg.m - agg.s, agg.m + agg.s, color=st.GRAPH_COLOR[graph], alpha=0.10, lw=0)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Counties with known labels (% of all counties)")
        ax.set_ylim(*ylim)
    handles = [Line2D([], [], color=st.GRAPH_COLOR[g], lw=3, label=st.GRAPH_LABEL[g]) for g in st.GRAPHS]
    handles += [Line2D([], [], color="#44525c", label=st.MODEL_LABEL[m], **st.MODEL_STYLE[m]) for m in st.MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.76, bottom=0.2, wspace=0.16, left=0.07, right=0.98)
    st.headline(fig, "More known counties, better predictions",
                f"{st.SPLIT_LABEL[split]}; performance on the scored counties vs training size; mean and standard deviation over {d.seed.nunique()} random seeds")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def metrics_heatmap(comp, out, split, frac="max"):
    """All metrics for every graph and model at one training size."""
    metrics = ["balanced_accuracy", "accuracy", "precision", "recall", "f1_democratic", "auc"]
    short = {"balanced_accuracy": "Bal. acc.", "accuracy": "Accuracy", "precision": "Precision",
             "recall": "Recall", "f1_democratic": "F1", "auc": "AUC"}
    d = comp[(comp.train_frac == frac) & (comp.split == split)]
    pairs = [(g, m) for g in st.GRAPHS for m in st.MODELS]
    labels = [f"{st.GRAPH_LABEL[g].replace(' graph', '')} · {st.MODEL_LABEL[m]}" for g, m in pairs]
    mean, std = np.zeros((len(pairs), len(metrics))), np.zeros((len(pairs), len(metrics)))
    for i, (g, m) in enumerate(pairs):
        sel = d[(d.graph == g) & (d.model == m)]
        mean[i], std[i] = sel[metrics].mean().to_numpy(), sel[metrics].std().to_numpy()
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.imshow(mean, cmap="YlGnBu", vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            dark = mean[i, j] > 0.68
            ax.text(j, i - 0.08, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=12, weight="bold",
                    color="white" if dark else st.INK)
            ax.text(j, i + 0.28, f"±{std[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#e8f1f4" if dark else st.INK, alpha=0.8)
    ax.set_xticks(range(len(metrics)), [short[m] for m in metrics])
    ax.set_yticks(range(len(pairs)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.subplots_adjust(top=0.8, bottom=0.07, left=0.31, right=0.98)
    st.headline(fig, "Every metric, every graph and model",
                f"{st.SPLIT_LABEL[split]}, {d.known_share.mean():.0%} of counties known; Democratic is the positive class; mean ± std over {d.seed.nunique()} seeds")
    fig.savefig(out, dpi=200)
    plt.close(fig)
