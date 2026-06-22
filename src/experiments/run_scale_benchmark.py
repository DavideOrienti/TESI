"""
Harness di benchmark di scala: per ogni combinazione (dataset, model) lancia
_bench_one.py come SUBPROCESS separato (picco di memoria isolato e pulito per
ogni run), raccoglie il JSON prodotto e scrive un aggregato in
src/experiments/results/scale_benchmark.csv (una riga per dataset/model).

Esecuzione dal più economico al più costoso: dataset small prima, poi
ml-25m-1m, poi ml-20m; per ogni dataset: popularity, poi pure_svd_sparse,
poi itemknn_sparse (il più costoso, in base ai tempi osservati su small).

Uso:
  python -m src.experiments.run_scale_benchmark --datasets small
  python -m src.experiments.run_scale_benchmark --datasets small ml-25m-1m ml-20m
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATASETS_ORDER = ["small", "ml-25m-1m", "ml-20m"]
MODELS_ORDER = ["popularity", "pure_svd_sparse", "itemknn_sparse"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "scale_benchmark.csv"

COLUMNS = [
    "dataset", "model", "random_state",
    "n_train", "n_users", "n_items", "density",
    "fit_time_s", "eval_time_s", "total_wall_s", "peak_rss_mb",
    "HR@5", "HR@10", "HR@20",
    "NDCG@5", "NDCG@10", "NDCG@20",
    "MRR@5", "MRR@10", "MRR@20",
]


def run_one(dataset: str, model: str) -> dict:
    print(f"[run_scale_benchmark] {dataset} / {model} ...")
    proc = subprocess.run(
        [sys.executable, "-m", "src.experiments._bench_one", dataset, model],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"_bench_one fallito per ({dataset}, {model}), exit={proc.returncode}")

    last_line = proc.stdout.strip().splitlines()[-1]
    result = json.loads(last_line)
    print(
        f"  -> fit={result.get('fit_time_s')}s eval={result.get('eval_time_s')}s "
        f"total_wall={result.get('total_wall_s')}s peak_rss={result.get('peak_rss_mb')}MB"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ORDER, choices=DATASETS_ORDER)
    parser.add_argument("--models", nargs="+", default=MODELS_ORDER, choices=MODELS_ORDER)
    args = parser.parse_args()

    datasets = [d for d in DATASETS_ORDER if d in args.datasets]
    models = [m for m in MODELS_ORDER if m in args.models]

    rows = []
    for dataset in datasets:
        for model in models:
            rows.append(run_one(dataset, model))

    df = pd.DataFrame(rows, columns=COLUMNS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    print(f"\nsalvato -> {RESULTS_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
