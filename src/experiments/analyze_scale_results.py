"""
Legge src/experiments/results/scale_benchmark.csv e scrive
src/experiments/results/derived_metrics.csv con, per ogni riga (dataset, model):
- x_random@K = HR@K / (K/n_items), per K in {5,10,20}: quanto il modello batte
  un ranking uniformemente casuale sullo stesso catalogo;
- lift_<METRIC>@K = valore_modello / valore_popularity (stesso dataset), per
  METRIC in {HR,NDCG,MRR}, K in {5,10,20};
- scale_ratio_<colonna> = valore_dataset_corrente / valore_dataset_precedente
  (stesso modello), per fit_time_s, eval_time_s, total_wall_s, peak_rss_mb e
  per ogni metrica di qualita (HR/NDCG/MRR @5/10/20). "Precedente" segue
  l'ordine canonico small -> ml-25m-1m -> ml-20m; popolato solo quando esiste
  una riga per il dataset precedente con lo stesso modello.
- theoretical_user_item_ratio = (n_users*n_items)_corrente / (n_users*n_items)_precedente,
  stesso criterio di "precedente": riferimento per confrontare con gli
  scale_ratio osservati (se un costo scala come O(n_users x n_items), il suo
  scale_ratio dovrebbe avvicinarsi a questo valore).

Nessun re-run di modelli: solo lettura/scrittura di CSV derivati.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BENCHMARK_CSV = RESULTS_DIR / "scale_benchmark.csv"
DERIVED_CSV = RESULTS_DIR / "derived_metrics.csv"

DATASETS_ORDER = ["small", "ml-25m-1m", "ml-20m"]
TOP_K_LIST = [5, 10, 20]
QUALITY_METRICS = [f"{m}@{k}" for m in ("HR", "NDCG", "MRR") for k in TOP_K_LIST]
COST_COLUMNS = ["fit_time_s", "eval_time_s", "total_wall_s", "peak_rss_mb"]


def previous_dataset(dataset: str, present: set[str]) -> str | None:
    idx = DATASETS_ORDER.index(dataset)
    for cand in reversed(DATASETS_ORDER[:idx]):
        if cand in present:
            return cand
    return None


def main():
    df = pd.read_csv(BENCHMARK_CSV)
    present = set(df["dataset"].unique())

    rows = []
    for _, row in df.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        n_items = row["n_items"]

        out = {"dataset": dataset, "model": model, "n_users": row["n_users"], "n_items": n_items}

        # x_random@K
        for k in TOP_K_LIST:
            random_hr = k / n_items if n_items > 0 else float("nan")
            out[f"x_random@{k}"] = row[f"HR@{k}"] / random_hr if random_hr > 0 else float("nan")

        # lift su popularity, stesso dataset
        pop_row = df[(df["dataset"] == dataset) & (df["model"] == "popularity")]
        if len(pop_row) == 1:
            pop_row = pop_row.iloc[0]
            for metric in QUALITY_METRICS:
                pop_val = pop_row[metric]
                out[f"lift_{metric}"] = row[metric] / pop_val if pop_val else float("nan")
        else:
            for metric in QUALITY_METRICS:
                out[f"lift_{metric}"] = float("nan")

        # scale_ratio rispetto al dataset precedente (stesso modello)
        prev_ds = previous_dataset(dataset, present)
        if prev_ds is not None:
            prev_row = df[(df["dataset"] == prev_ds) & (df["model"] == model)]
            if len(prev_row) == 1:
                prev_row = prev_row.iloc[0]
                for col in COST_COLUMNS + QUALITY_METRICS:
                    prev_val = prev_row[col]
                    cur_val = row[col]
                    if pd.isna(prev_val) or pd.isna(cur_val) or prev_val == 0:
                        out[f"scale_ratio_{col}"] = float("nan")
                    else:
                        out[f"scale_ratio_{col}"] = cur_val / prev_val

                prev_users_items = prev_row["n_users"] * prev_row["n_items"]
                cur_users_items = row["n_users"] * n_items
                out["theoretical_user_item_ratio"] = (
                    cur_users_items / prev_users_items if prev_users_items > 0 else float("nan")
                )
                out["scale_ratio_baseline"] = prev_ds
            else:
                for col in COST_COLUMNS + QUALITY_METRICS:
                    out[f"scale_ratio_{col}"] = float("nan")
                out["theoretical_user_item_ratio"] = float("nan")
                out["scale_ratio_baseline"] = None
        else:
            for col in COST_COLUMNS + QUALITY_METRICS:
                out[f"scale_ratio_{col}"] = float("nan")
            out["theoretical_user_item_ratio"] = float("nan")
            out["scale_ratio_baseline"] = None

        rows.append(out)

    derived = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    derived.to_csv(DERIVED_CSV, index=False)
    print(f"salvato -> {DERIVED_CSV}")
    print(derived.to_string(index=False))


if __name__ == "__main__":
    main()
