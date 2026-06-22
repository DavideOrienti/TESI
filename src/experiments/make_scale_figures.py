"""
Genera le figure di confronto di scala in src/experiments/results/figures/,
leggendo scale_benchmark.csv e derived_metrics.csv (nessun re-run di modelli).

F1: barre raggruppate HR@10/NDCG@10/MRR@10 per modello, un pannello per dataset.
F2: lift su popularity di HR@10 per modello (pure_svd_sparse, itemknn_sparse)
    sulle scale disponibili, con riferimento a y=1 (= popularity).
F3: due pannelli (eval_time_s, peak_rss_mb) vs scala (n_users*n_items, asse y
    log), con una retta di riferimento tracciata proporzionale a n_users*n_items
    (ancorata al valore di itemknn_sparse su small, il modello piu costoso, come
    riferimento per un'ipotetica crescita O(n_users x n_items)).

Salva PNG (300 dpi) e PDF per ciascuna figura.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BENCHMARK_CSV = RESULTS_DIR / "scale_benchmark.csv"
DERIVED_CSV = RESULTS_DIR / "derived_metrics.csv"

DATASETS_ORDER = ["small", "ml-25m-1m", "ml-20m"]
MODELS_ORDER = ["popularity", "pure_svd_sparse", "itemknn_sparse"]
MODEL_COLORS = {"popularity": "#888888", "pure_svd_sparse": "#1f77b4", "itemknn_sparse": "#d62728"}


def save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"salvato -> {png_path}")
    print(f"salvato -> {pdf_path}")


def figure_1(df: pd.DataFrame) -> None:
    datasets = [d for d in DATASETS_ORDER if d in df["dataset"].unique()]
    metrics = ["HR@10", "NDCG@10", "MRR@10"]
    models = [m for m in MODELS_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    axes = axes[0]

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        for i, model in enumerate(models):
            row = sub[sub["model"] == model]
            values = [float(row[metric].iloc[0]) if len(row) else 0.0 for metric in metrics]
            ax.bar(x + i * width - 0.4 + width / 2, values, width, label=model, color=MODEL_COLORS[model])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(f"dataset = {dataset}")
        ax.set_ylabel("valore metrica (test)")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend()
    fig.suptitle("HR@10 / NDCG@10 / MRR@10 per modello e dataset")
    fig.tight_layout()
    save_fig(fig, "F1_metriche_per_modello_dataset")
    plt.close(fig)


def figure_2(derived: pd.DataFrame) -> None:
    datasets = [d for d in DATASETS_ORDER if d in derived["dataset"].unique()]
    models = [m for m in MODELS_ORDER if m != "popularity" and m in derived["model"].unique()]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        values = []
        for dataset in datasets:
            row = derived[(derived["dataset"] == dataset) & (derived["model"] == model)]
            values.append(float(row["lift_HR@10"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=model, color=MODEL_COLORS[model])

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="popularity (riferimento, lift=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("lift HR@10 su popularity")
    ax.set_title("Lift HR@10 su popularity, per modello e scala")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "F2_lift_HR10_su_popularity")
    plt.close(fig)


def figure_3(df: pd.DataFrame) -> None:
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.5))

    panels = [("eval_time_s", ax_time, "eval_time_s (secondi)"), ("peak_rss_mb", ax_mem, "peak_rss_mb (MB)")]

    for col, ax, ylabel in panels:
        for model in MODELS_ORDER:
            sub = df[df["model"] == model].copy()
            if sub.empty:
                continue
            sub["scale"] = sub["n_users"] * sub["n_items"]
            sub = sub.sort_values("scale")
            sub = sub.dropna(subset=[col])
            if sub.empty:
                continue
            ax.plot(sub["scale"], sub[col], marker="o", label=model, color=MODEL_COLORS[model])

        # retta di riferimento O(n_users x n_items), ancorata a itemknn_sparse su small
        anchor = df[(df["model"] == "itemknn_sparse") & (df["dataset"] == "small")]
        if len(anchor) == 1 and pd.notna(anchor[col].iloc[0]):
            anchor = anchor.iloc[0]
            anchor_scale = anchor["n_users"] * anchor["n_items"]
            anchor_val = anchor[col]
            all_scales = (df["n_users"] * df["n_items"]).dropna()
            scale_range = np.array(sorted(all_scales.unique()))
            ref_values = anchor_val * (scale_range / anchor_scale)
            ax.plot(
                scale_range, ref_values, linestyle=":", color="gray",
                label="riferimento O(utenti x item)\n(ancorato a itemknn_sparse/small)"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_users x n_items (scala)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    ax_time.set_title("Tempo di valutazione vs scala")
    ax_mem.set_title("Picco di memoria vs scala")
    ax_time.legend(fontsize=8)
    ax_mem.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "F3_costo_vs_scala")
    plt.close(fig)


def main():
    df = pd.read_csv(BENCHMARK_CSV)
    derived = pd.read_csv(DERIVED_CSV)

    figure_1(df)
    figure_2(derived)
    figure_3(df)


if __name__ == "__main__":
    main()
