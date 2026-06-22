"""
Test di significativita per-utente, su TEST, per ciascun dataset disponibile
in data/processed/<dataset>/baseline_{popularity,pure_svd_sparse,itemknn_sparse}/
test_per_user_metrics.csv, allineando sempre gli utenti su userId.

Per ogni coppia di modelli (svd vs popularity, svd vs itemknn, popularity vs itemknn):
- McNemar (esatto, via scipy.stats.binomtest sulle coppie discordanti) sull'hit/miss
  per-utente di HR@10 (paired binario). statsmodels non e installato nel venv e non
  va aggiunto (fuori perimetro requirements.txt) -> implementazione manuale.
- Wilcoxon signed-rank (scipy.stats.wilcoxon) su NDCG@10 e MRR@10 per-utente
  (paired non parametrico, zero_method="wilcox" -> scarta le differenze nulle).

Scrive src/experiments/results/significance_tests.csv. Nessun re-run di modelli:
solo lettura dei file esistenti.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "significance_tests.csv"

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

MODEL_DIRS = {
    "popularity": "baseline_popularity",
    "pure_svd_sparse": "baseline_pure_svd_sparse",
    "itemknn_sparse": "baseline_itemknn_cf_improved_sparse",
}

PAIRS = [
    ("pure_svd_sparse", "popularity"),
    ("pure_svd_sparse", "itemknn_sparse"),
    ("popularity", "itemknn_sparse"),
]

DATASET_DIR_MAP = {
    "small": "small",
    "ml-25m-1m": "ml-25m-1m",
    "ml-20m": "20m",
}


def load_per_user(dataset_dir: str, model: str) -> pd.DataFrame:
    path = DATA_ROOT / dataset_dir / MODEL_DIRS[model] / "test_per_user_metrics.csv"
    return pd.read_csv(path)[["userId", "ground_truth_movieId", "HR@10", "NDCG@10", "MRR@10"]]


def mcnemar_exact(hit_a: np.ndarray, hit_b: np.ndarray) -> tuple[float, float, int, str]:
    """McNemar esatto sulle coppie discordanti. Ritorna (statistic, p_value, n_discordant, note)."""
    b = int(np.sum((hit_a == 1) & (hit_b == 0)))  # a hit, b miss
    c = int(np.sum((hit_a == 0) & (hit_b == 1)))  # a miss, b hit
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, 0, "nessuna coppia discordante"
    statistic = (b - c) ** 2 / n_discordant
    p_value = binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided").pvalue
    return float(statistic), float(p_value), n_discordant, f"b={b}, c={c}"


def run_pair(dataset: str, df_a: pd.DataFrame, df_b: pd.DataFrame, model_a: str, model_b: str) -> list[dict]:
    merged = df_a.merge(df_b, on="userId", suffixes=("_a", "_b"))
    n_total = len(merged)
    rows = []

    # McNemar su HR@10
    stat, p, n_eff, note = mcnemar_exact(merged["HR@10_a"].to_numpy(), merged["HR@10_b"].to_numpy())
    mean_a, mean_b = merged["HR@10_a"].mean(), merged["HR@10_b"].mean()
    winner = model_a if mean_a > mean_b else (model_b if mean_b > mean_a else "pareggio")
    rows.append({
        "dataset": dataset, "model_a": model_a, "model_b": model_b, "test": "McNemar_HR@10",
        "statistic": stat, "p_value": p, "n_effective": n_eff, "n_total": n_total,
        "mean_a": mean_a, "mean_b": mean_b, "winner": winner,
        "significant_0.05": bool(p < 0.05), "note": note,
    })

    # Wilcoxon su NDCG@10 e MRR@10
    for metric in ["NDCG@10", "MRR@10"]:
        a = merged[f"{metric}_a"].to_numpy()
        b = merged[f"{metric}_b"].to_numpy()
        diffs = a - b
        n_nonzero = int(np.sum(diffs != 0))
        mean_a, mean_b = a.mean(), b.mean()
        winner = model_a if mean_a > mean_b else (model_b if mean_b > mean_a else "pareggio")

        if n_nonzero == 0:
            stat, p, note = float("nan"), 1.0, "nessuna differenza non nulla"
        else:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            note = ""

        rows.append({
            "dataset": dataset, "model_a": model_a, "model_b": model_b, "test": f"Wilcoxon_{metric}",
            "statistic": float(stat), "p_value": float(p), "n_effective": n_nonzero, "n_total": n_total,
            "mean_a": mean_a, "mean_b": mean_b, "winner": winner,
            "significant_0.05": bool(p < 0.05), "note": note,
        })

    return rows


def main():
    all_rows = []
    for dataset, dataset_dir in DATASET_DIR_MAP.items():
        base = DATA_ROOT / dataset_dir
        if not base.exists():
            continue
        try:
            per_user = {model: load_per_user(dataset_dir, model) for model in MODEL_DIRS}
        except FileNotFoundError:
            print(f"[significance_tests] {dataset}: per-utente non disponibile per tutti i modelli, salto.")
            continue

        print(f"[significance_tests] dataset={dataset}")
        for model_a, model_b in PAIRS:
            all_rows.extend(run_pair(dataset, per_user[model_a], per_user[model_b], model_a, model_b))

    result = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsalvato -> {OUTPUT_CSV}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
