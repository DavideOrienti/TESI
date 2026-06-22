"""
Esegue UN solo (dataset, model) IN-PROCESS (via runpy, come se fosse lanciato
da CLI con `python -m <modulo>`), misura wall-clock totale del processo e
picco di memoria (Windows: psutil peak_wset), poi stampa su stdout un singolo
oggetto JSON con i risultati.

Pensato per essere lanciato come SUBPROCESS separato da run_scale_benchmark.py:
un processo per combinazione (dataset, model) => picco di memoria isolato e
pulito, non contaminato da run precedenti nello stesso processo.

Uso: python -m src.experiments._bench_one <dataset> <model>
model in {popularity, pure_svd_sparse, itemknn_sparse}
"""
from __future__ import annotations
import json
import os
import runpy
import sys
import time

RANDOM_STATE = 42  # invariato in tutta la linea sperimentale

MODELS = {
    "popularity": {
        "module": "src.pipeline.Fase_1.08_popularity_baseline",
        "output_subdir": "baseline_popularity",
    },
    "pure_svd_sparse": {
        "module": "src.pipeline.Fase_1.10_matrix_factorization_sparse",
        "output_subdir": "baseline_pure_svd_sparse",
    },
    "itemknn_sparse": {
        "module": "src.pipeline.Fase_1.09b_itemknn_cf_improved_sparse",
        "output_subdir": "baseline_itemknn_cf_improved_sparse",
    },
}

METRIC_KEYS = [f"{m}@{k}" for m in ("HR", "NDCG", "MRR") for k in (5, 10, 20)]


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python -m src.experiments._bench_one <dataset> <model>", file=sys.stderr)
        sys.exit(1)

    dataset, model = sys.argv[1], sys.argv[2]
    if model not in MODELS:
        raise ValueError(f"Modello non riconosciuto: {model!r}. Disponibili: {sorted(MODELS)}")

    os.environ["CINEREC_DATASET"] = dataset

    # import dopo aver impostato CINEREC_DATASET, cosi load_settings() lo rilegge
    from src.utils.io import load_settings
    import psutil
    import pandas as pd

    module_name = MODELS[model]["module"]
    output_subdir = MODELS[model]["output_subdir"]

    t0 = time.perf_counter()
    runpy.run_module(module_name, run_name="__main__")
    total_wall_s = time.perf_counter() - t0

    proc = psutil.Process()
    mem = proc.memory_info()
    peak_rss_mb = getattr(mem, "peak_wset", mem.rss) / (1024 * 1024)

    s = load_settings(dataset=dataset)
    base = s.paths.processed

    train = pd.read_csv(base / "ratings_train.csv")
    n_users = int(train["userId"].nunique())
    n_items = int(train["movieId"].nunique())
    n_train = len(train)
    density = n_train / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0

    summary_path = base / output_subdir / "summary_metrics.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    test_metrics = summary.get("test", {})

    result = {
        "dataset": dataset,
        "model": model,
        "random_state": RANDOM_STATE,
        "n_train": n_train,
        "n_users": n_users,
        "n_items": n_items,
        "density": density,
        "fit_time_s": summary.get("fit_time_s"),
        "eval_time_s": summary.get("eval_time_s"),
        "total_wall_s": round(total_wall_s, 4),
        "peak_rss_mb": round(peak_rss_mb, 2),
    }
    for key in METRIC_KEYS:
        result[key] = test_metrics.get(key)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
