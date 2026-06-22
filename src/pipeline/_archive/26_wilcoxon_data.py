"""
Script: 26_wilcoxon_data.py
Scopo : Estrae gli score NDCG@10 per-utente di PureSVD e Hybrid SVD+CB
        e li salva in un JSON da incollare nel supervisore.

Esecuzione (dalla root del progetto):
    python -m src.pipeline.Fase_4.26_wilcoxon_data

Output :  data/processed/small/wilcoxon_scores.json
          + stampa diretta a schermo dei valori da copiare
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Adatta questi import alla struttura reale del tuo progetto ─────────────
# Se hai già funzioni per calcolare gli score per-utente, usale.
# Altrimenti questo script li ricalcola da zero usando gli artefatti.

from src.recommenders.scoring import (   # <-- modifica se i nomi differiscono
    puresvd_score,
    hybrid_svd_cb_score,
)

# ── Percorsi ───────────────────────────────────────────────────────────────
BASE       = Path("data")
TEST_PATH  = BASE / "processed" / "small" / "ratings_test.csv"
OUT_PATH   = BASE / "processed" / "small" / "wilcoxon_scores.json"

def ndcg_at_10(recommended: list, relevant_item: int) -> float:
    """NDCG@10 con un solo item rilevante (leave-one-out)."""
    try:
        rank = recommended.index(relevant_item) + 1   # 1-based
    except ValueError:
        return 0.0
    if rank > 10:
        return 0.0
    return 1.0 / np.log2(rank + 1)

def main():
    test_df = pd.read_csv(TEST_PATH)
    print(f"Utenti nel test set: {len(test_df)}")

    svd_scores    = []
    hybrid_scores = []
    user_ids      = []

    for _, row in test_df.iterrows():
        uid        = int(row["userId"])
        relevant   = int(row["movieId"])

        rec_svd    = puresvd_score(uid, k=10)          # lista di movieId
        rec_hybrid = hybrid_svd_cb_score(uid, k=10)    # lista di movieId

        svd_scores.append(ndcg_at_10(rec_svd,    relevant))
        hybrid_scores.append(ndcg_at_10(rec_hybrid, relevant))
        user_ids.append(uid)

    result = {
        "n_users"       : len(user_ids),
        "svd_mean"      : float(np.mean(svd_scores)),
        "hybrid_mean"   : float(np.mean(hybrid_scores)),
        "svd_ndcg10"    : svd_scores,
        "hybrid_ndcg10" : hybrid_scores,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f)

    print("=" * 55)
    print(f"  PureSVD   NDCG@10 medio : {result['svd_mean']:.6f}")
    print(f"  Hybrid    NDCG@10 medio : {result['hybrid_mean']:.6f}")
    print(f"  Salvato in: {OUT_PATH}")
    print("=" * 55)
    print("\nCopia e incolla nel supervisore il contenuto di:")
    print(f"  {OUT_PATH}")

if __name__ == "__main__":
    main()
