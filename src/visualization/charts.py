"""Chart figures built from the saved experiment runs."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from . import style as st
from .data import load_comparison, load_previous_year


def _frac_order(df):
    return sorted(df.train_frac.unique(), key=lambda f: 1.0 if f == "max" else float(f))


def learning_curves(comp, out):
    """Balanced accuracy and AUC against the share of counties whose labels are known."""
    splits = [s for s in st.SPLIT_LABEL if s in set(comp.split)]
    rows = [("balanced_accuracy", "Balanced accuracy"), ("auc", "AUC")]
    fig, axes = plt.subplots(2, len(splits), figsize=(5.6 * len(splits) + 0.8, 8.2), sharex=True, squeeze=False)
    for r, (metric, mlabel) in enumerate(rows):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            d = comp[(comp.split == split) & (comp.model != "majority")]
            for graph in st.GRAPHS:
                for model in st.MODELS:
                    g = d[(d.graph == graph) & (d.model == model)]
                    agg = g.groupby("train_frac").agg(x=("known_share", "mean"), m=(metric, "mean"), s=(metric, "std"))
                    agg = agg.loc[_frac_order(g)]
                    ax.plot(agg.x * 100, agg.m, color=st.GRAPH_COLOR[graph], lw=2, ms=5, **st.MODEL_STYLE[model])
                    ax.fill_between(agg.x * 100, agg.m - agg.s, agg.m + agg.s, color=st.GRAPH_COLOR[graph], alpha=0.10, lw=0)
            if r == 0:
                ax.set_title(st.SPLIT_LABEL[split], loc="left")
            ax.set_ylabel(mlabel if c == 0 else "")
            ax.set_xlabel("Counties with known labels (% of all counties)" if r == 1 else "")
            ax.set_ylim(0.5 if metric == "balanced_accuracy" else 0.65, 0.92 if metric == "balanced_accuracy" else 0.97)
    handles = [Line2D([], [], color=st.GRAPH_COLOR[g], lw=3, label=st.GRAPH_LABEL[g]) for g in st.GRAPHS]
    handles += [Line2D([], [], color="#44525c", label=st.MODEL_LABEL[m], **st.MODEL_STYLE[m]) for m in st.MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.84, bottom=0.12, hspace=0.12, wspace=0.18, left=0.09, right=0.98)
    st.headline(fig, "More known counties, better predictions",
                f"Test performance vs training size; mean and standard deviation over {comp.seed.nunique()} random seeds")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def metrics_heatmap(comp, out, frac="max"):
    """All metrics for every graph/model at one training size."""
    metrics = ["balanced_accuracy", "accuracy", "precision", "recall", "f1_democratic", "auc"]
    splits = [s for s in st.SPLIT_LABEL if s in set(comp.split)]
    d = comp[comp.train_frac == frac]
    labels = [f"{st.GRAPH_LABEL[g].replace(' graph', '')} · {st.MODEL_LABEL[m]}" for g in st.GRAPHS for m in st.MODELS]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.6 * len(splits), 5.4), squeeze=False)
    for ax, split in zip(axes[0], splits):
        mean = np.zeros((6, len(metrics)))
        std = np.zeros_like(mean)
        for i, (g, m) in enumerate((g, m) for g in st.GRAPHS for m in st.MODELS):
            sel = d[(d.split == split) & (d.graph == g) & (d.model == m)]
            mean[i], std[i] = sel[metrics].mean().to_numpy(), sel[metrics].std().to_numpy()
        im = ax.imshow(mean, cmap="YlGnBu", vmin=0.3, vmax=1.0, aspect="auto")
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                dark = mean[i, j] > 0.68
                ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=11, weight="bold",
                        color="white" if dark else st.INK)
                ax.text(j, i + 0.30, f"±{std[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#e8f1f4" if dark else st.INK, alpha=0.8)
        short = {"balanced_accuracy": "Bal. acc.", "accuracy": "Accuracy", "precision": "Precision",
                 "recall": "Recall", "f1_democratic": "F1", "auc": "AUC"}
        ax.set_xticks(range(len(metrics)), [short[m] for m in metrics])
        ax.set_yticks(range(6), labels if ax is axes[0, 0] else [""] * 6)
        ax.set_title(st.SPLIT_LABEL[split], loc="left")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.subplots_adjust(top=0.76, bottom=0.08, left=0.2, right=0.98, wspace=0.06)
    share = d.known_share.mean()
    st.headline(fig, "Every metric, every graph and model",
                f"{share:.0%} of counties known; Democratic is the positive class; mean ± std over {comp.seed.nunique()} seeds")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def tuned_parameters(comp, out):
    """Which parameter values cross-validation picked."""
    d = comp[comp.model != "majority"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))

    ax = axes[0]
    k = d[d.model == "knn"]
    ks = sorted(k.k.dropna().unique())
    bottom = np.zeros(len(ks))
    for g in st.GRAPHS:
        cnt = np.array([((k.k == v) & (k.graph == g)).sum() for v in ks])
        ax.bar(range(len(ks)), cnt, bottom=bottom, color=st.GRAPH_COLOR[g], label=st.GRAPH_LABEL[g].replace(" graph", ""))
        bottom += cnt
    ax.set_xticks(range(len(ks)), [str(int(v)) for v in ks])
    ax.set_title("kNN: neighbours k", loc="left")
    ax.set_ylabel("Runs that chose it")
    ax.legend(fontsize=9)

    ax = axes[1]
    a = d[d.model == "label_propagation"]
    als = sorted(a.alpha.dropna().unique())
    bottom = np.zeros(len(als))
    for g in st.GRAPHS:
        cnt = np.array([((a.alpha == v) & (a.graph == g)).sum() for v in als])
        ax.bar(range(len(als)), cnt, bottom=bottom, color=st.GRAPH_COLOR[g])
        bottom += cnt
    ax.set_xticks(range(len(als)), [f"{v:g}" for v in als])
    ax.set_title("Label propagation: alpha", loc="left")

    ax = axes[2]
    c = d[d.graph == "combined"]
    lams = sorted(c.lam.dropna().unique())
    width = 0.38
    for off, m in zip((-width / 2, width / 2), st.MODELS):
        cnt = [((c.lam == v) & (c.model == m)).sum() for v in lams]
        ax.bar(np.arange(len(lams)) + off, cnt, width, color="#44525c" if m == "knn" else "#9aa7ae", label=st.MODEL_LABEL[m])
    ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
    ax.set_title("Combined graph: border share", loc="left")
    ax.set_xlabel("0 = demographic only, 1 = border only", fontsize=9, color=st.MUTED)
    ax.legend(fontsize=9)

    ax = axes[3]
    bins = np.linspace(0, 1, 21)
    for m, col in zip(st.MODELS, ("#44525c", "#c0a02b")):
        ax.hist(d[d.model == m].threshold, bins=bins, histtype="stepfilled", alpha=0.45, color=col, label=st.MODEL_LABEL[m])
    ax.axvline(0.5, color=st.MUTED, ls=":", lw=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.92, "0.5", color=st.MUTED, fontsize=9)
    ax.set_title("Decision threshold", loc="left")
    ax.set_xlabel("Predict Democratic above this probability", fontsize=9, color=st.MUTED)
    ax.legend(fontsize=9)

    fig.subplots_adjust(top=0.68, bottom=0.17, left=0.045, right=0.99, wspace=0.28)
    st.headline(fig, "What cross-validation chose", "Settings selected by 5-fold cross-validation, across every seed, split and training size", y=0.97)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def previous_year(out, splits=("stratified", "block")):
    """Seeding with the previous election's result, against the persistence baseline."""
    frames = {}
    for s in splits:
        try:
            frames[s] = load_previous_year(s)
        except FileNotFoundError:
            continue
    if not frames:
        return False
    graphs = ["border", "demographic"]
    fig, axes = plt.subplots(len(frames), 2, figsize=(11.4, 4.2 * len(frames) + 1.3), sharey=True, sharex=True, squeeze=False)
    xs = {"0.05": 5, "0.1": 10, "0.2": 20, "0.4": 40, "max": 80}
    colors = {"knn": "#44525c", "lp": "#c0a02b"}
    for r, (split, df) in enumerate(frames.items()):
        for c, graph in enumerate(graphs):
            ax = axes[r, c]
            d = df[df.graph == graph]
            pers = d[d.method == "persistence"].balanced_accuracy.mean()
            ax.axhline(pers, color=st.MUTED, ls=":", lw=1.6)
            if c == 1 and r == 0:
                ax.text(5, pers - 0.035, "Persistence: each county's own 2020 result", color=st.MUTED, fontsize=9)
            for method in ("knn / prev-year seeds", "knn / current-year seeds", "lp / prev-year seeds", "lp / current-year seeds"):
                g = d[d.method == method].groupby("train_frac").balanced_accuracy.agg(["mean", "std"])
                g = g.loc[[f for f in xs if f in g.index]]
                model = method.split(" / ")[0]
                prev = "prev-year" in method
                x = [xs[f] for f in g.index]
                ax.plot(x, g["mean"], color=colors[model], ls="-" if prev else "--", marker="o" if prev else "s", lw=2, ms=4.5)
                if prev:
                    ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=colors[model], alpha=0.12, lw=0)
            ax.set_title(f"{st.GRAPH_LABEL[graph]} · {st.SPLIT_LABEL[split]}", loc="left", fontsize=12)
            ax.set_ylim(0.45, 1.0)
            if c == 0:
                ax.set_ylabel("Balanced accuracy")
            if r == len(frames) - 1:
                ax.set_xlabel("Seed counties (% of all counties)")
    handles = [Line2D([], [], color=colors[m], lw=3, label=st.MODEL_LABEL["knn" if m == "knn" else "label_propagation"]) for m in colors]
    handles += [Line2D([], [], color="#44525c", ls="-", marker="o", label="Seeded with 2020 results"),
                Line2D([], [], color="#44525c", ls="--", marker="s", label="Seeded with 2024 results")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.84 if len(frames) == 2 else 0.80, bottom=0.12 if len(frames) == 2 else 0.17, hspace=0.25, wspace=0.08, left=0.09, right=0.98)
    st.headline(fig, "Seeding with last election's results",
                "Predicting 2024 from 2020 results at the seed counties; 2020 seeds work about as well as 2024 seeds")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def degree_distributions(out):
    """How many connections each county has in each graph."""
    from ..graphs.combined import blend_edges, build_components

    parts = build_components()
    degrees = {"border": (parts["border_edges"] > 0).sum(1),
               "demographic": (parts["demographic_edges"] > 0).sum(1),
               "combined": (blend_edges(parts, 0.5) > 0).sum(1)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, g in zip(axes, st.GRAPHS):
        d = degrees[g]
        ax.hist(d, bins=np.arange(0, d.max() + 2) - 0.5, color=st.GRAPH_COLOR[g])
        ax.set_yscale("log")
        ax.set_title(st.GRAPH_LABEL[g], loc="left")
        ax.set_xlabel("Connections per county")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.text(0.97, 0.94, f"mean {d.mean():.1f}\nmax {d.max()}", transform=ax.transAxes, ha="right", va="top", color=st.MUTED, fontsize=10)
    axes[0].set_ylabel("Counties (log scale)")
    fig.subplots_adjust(top=0.68, bottom=0.17, left=0.07, right=0.99, wspace=0.08)
    st.headline(fig, "How connected is each graph?", "Number of edges per county (3,144 counties with census data; combined uses border share 0.5)", y=0.97)
    fig.savefig(out, dpi=200)
    plt.close(fig)
