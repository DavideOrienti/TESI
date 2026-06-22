"""
Calcola Catalog Coverage@K e Novelty@K per tutti i modelli.

Metriche beyond-accuracy per analizzare popularity bias e diversità:
- Coverage bassa → sistema concentrato su pochi item popolari (filter bubble)
- Novelty alta  → sistema raccomanda item meno noti / più sorprendenti

Modelli valutati: Popularity, PureSVD, Content-Based, Hybrid SVD+CB (γ=0.7)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import compute_catalog_coverage, compute_novelty
from src.recommenders.scoring import (
    build_user_seen_ratings,
    build_user_mean_ratings,
    minmax_normalize_scores,
    score_candidate_content,
)

TOP_K_LIST = [5, 10, 20]
# GAMMA = 0.7 — selezionato su validation NDCG@10 (Fase_3/17_hybrid_svd_content_eval.py)
GAMMA = 0.7


def _build_index_to_movieid(index_df: pd.DataFrame) -> dict[int, int]:
    """Mappa posizione riga → movieId (per decodificare movie_idx/neighbor_idx)."""
    return {idx: int(mid) for idx, mid in enumerate(index_df["movieId"])}


def _build_content_neighbors_dict(
    neighbors_df: pd.DataFrame,
    index_to_movieid: dict[int, int],
) -> dict[int, list[tuple[int, float]]]:
    """Costruisce {movieId: [(neighbor_movieId, sim), ...]} dal CSV con indici interi."""
    neigh: dict[int, list[tuple[int, float]]] = {}
    for movie_idx, g in neighbors_df.groupby("movie_idx"):
        mid = index_to_movieid.get(int(movie_idx))
        if mid is None:
            continue
        rows = []
        for _, row in g.iterrows():
            nb_mid = index_to_movieid.get(int(row["neighbor_idx"]))
            if nb_mid is not None:
                rows.append((nb_mid, float(row["similarity"])))
        rows.sort(key=lambda x: (-x[1], x[0]))
        neigh[mid] = rows
    return neigh


def _popularity_recs(
    test_users: list[int],
    seen_items_map: dict[int, set[int]],
    pop_ranking: list[int],
    top_k: int,
) -> dict[int, list[int]]:
    recs = {}
    for uid in test_users:
        seen = seen_items_map.get(uid, set())
        recs[uid] = [i for i in pop_ranking if i not in seen][:top_k]
    return recs


def _svd_recs(
    test_users: list[int],
    seen_items_map: dict[int, set[int]],
    svd_matrix: pd.DataFrame,
    user_means: dict[int, float],
    top_k: int,
) -> dict[int, list[int]]:
    recs = {}
    for uid in test_users:
        if uid not in svd_matrix.index:
            continue
        seen = seen_items_map.get(uid, set())
        u_mean = float(user_means.get(uid, 0.0))
        scores = {
            int(mid): float(svd_matrix.loc[uid, mid]) + u_mean
            for mid in svd_matrix.columns
            if int(mid) not in seen
        }
        recs[uid] = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return recs


def _content_recs(
    test_users: list[int],
    seen_ratings_map: dict[int, dict[int, float]],
    user_means: dict[int, float],
    candidate_items: list[int],
    content_neighbors_dict: dict[int, list[tuple[int, float]]],
    top_k: int,
) -> dict[int, list[int]]:
    recs = {}
    for uid in test_users:
        seen_ratings = seen_ratings_map.get(uid, {})
        seen_items = set(seen_ratings)
        u_mean = float(user_means.get(uid, 0.0))
        unseen = [m for m in candidate_items if m not in seen_items]
        scores = {
            mid: score_candidate_content(mid, seen_ratings, u_mean, content_neighbors_dict)
            for mid in unseen
        }
        recs[uid] = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return recs


def _hybrid_recs(
    test_users: list[int],
    seen_ratings_map: dict[int, dict[int, float]],
    user_means: dict[int, float],
    candidate_items: list[int],
    svd_matrix: pd.DataFrame,
    content_neighbors_dict: dict[int, list[tuple[int, float]]],
    gamma: float,
    top_k: int,
) -> dict[int, list[int]]:
    recs = {}
    for uid in test_users:
        if uid not in svd_matrix.index:
            continue
        seen_ratings = seen_ratings_map.get(uid, {})
        seen_items = set(seen_ratings)
        u_mean = float(user_means.get(uid, 0.0))
        unseen = [m for m in candidate_items if m not in seen_items]

        svd_raw = {
            mid: float(svd_matrix.loc[uid, mid]) + u_mean
            for mid in svd_matrix.columns
            if int(mid) not in seen_items
        }
        cb_raw = {
            mid: score_candidate_content(mid, seen_ratings, u_mean, content_neighbors_dict)
            for mid in unseen
        }

        svd_norm = minmax_normalize_scores(svd_raw)
        cb_norm = minmax_normalize_scores(cb_raw)

        all_items = set(svd_norm) | set(cb_norm)
        hybrid_scores = {
            i: gamma * svd_norm.get(i, 0.0) + (1.0 - gamma) * cb_norm.get(i, 0.0)
            for i in all_items
        }
        recs[uid] = sorted(hybrid_scores, key=hybrid_scores.__getitem__, reverse=True)[:top_k]
    return recs


def main():
    s = load_settings()
    base = s.paths.processed

    # ── Dati ────────────────────────────────────────────────────────────────
    train = pd.read_csv(base / "ratings_train.csv")
    test = pd.read_csv(base / "ratings_test.csv")

    catalog_size = train["movieId"].nunique()
    item_popularity: dict[int, int] = train.groupby("movieId").size().to_dict()
    total_ratings = len(train)
    test_users = test["userId"].unique().tolist()

    seen_ratings_map = build_user_seen_ratings(train)
    seen_items_map = {uid: set(r.keys()) for uid, r in seen_ratings_map.items()}
    user_means = build_user_mean_ratings(train)

    print(f"[24] dataset={s.dataset}")
    print(f"[24] catalogo: {catalog_size} item | train ratings: {total_ratings} | test utenti: {len(test_users)}")

    # ── Artefatti SVD e Content ──────────────────────────────────────────────
    svd_matrix = pd.read_csv(base / "baseline_pure_svd" / "predicted_scores_matrix.csv", index_col=0)
    svd_matrix.columns = [int(c) for c in svd_matrix.columns]
    svd_matrix.index = [int(i) for i in svd_matrix.index]

    index_df = pd.read_csv(base / "movie_embeddings_index_v2.csv")
    index_to_movieid = _build_index_to_movieid(index_df)

    neighbors_df = pd.read_csv(base / "content_top_neighbors_v2.csv")
    content_neighbors_dict = _build_content_neighbors_dict(neighbors_df, index_to_movieid)

    candidate_items = sorted(
        set(train["movieId"].unique()).intersection(set(index_df["movieId"].unique()))
    )

    # Popularity ranking (Bayesian average, come nel resto della pipeline)
    c_global = train["rating"].mean()
    min_votes = 10
    item_stats = train.groupby("movieId").agg(
        n=("rating", "count"), mean=("rating", "mean")
    ).reset_index()
    item_stats["pop_score"] = (
        item_stats["n"] / (item_stats["n"] + min_votes) * item_stats["mean"]
        + min_votes / (item_stats["n"] + min_votes) * c_global
    )
    pop_ranking: list[int] = item_stats.sort_values("pop_score", ascending=False)["movieId"].tolist()

    max_k = max(TOP_K_LIST)

    # ── Genera raccomandazioni ───────────────────────────────────────────────
    print("[24] Popularity...")
    pop_recs = _popularity_recs(test_users, seen_items_map, pop_ranking, max_k)

    print("[24] PureSVD...")
    svd_recs = _svd_recs(test_users, seen_items_map, svd_matrix, user_means, max_k)

    print("[24] Content-Based...")
    cb_recs = _content_recs(
        test_users, seen_ratings_map, user_means, candidate_items, content_neighbors_dict, max_k
    )

    print("[24] Hybrid SVD+CB (gamma=0.7)...")
    hybrid_recs = _hybrid_recs(
        test_users, seen_ratings_map, user_means, candidate_items,
        svd_matrix, content_neighbors_dict, GAMMA, max_k
    )

    # ── Calcola metriche ────────────────────────────────────────────────────
    models = {
        "Popularity":   pop_recs,
        "PureSVD":      svd_recs,
        "Content-Based": cb_recs,
        "Hybrid_SVD_CB": hybrid_recs,
    }

    results: dict[str, dict] = {}
    for name, recs in models.items():
        results[name] = {}
        for k in TOP_K_LIST:
            results[name][f"coverage@{k}"] = round(
                compute_catalog_coverage(recs, catalog_size, k), 4)
            results[name][f"novelty@{k}"] = round(
                compute_novelty(recs, item_popularity, total_ratings, k), 4)
            results[name][f"n_users@{k}"] = len(recs)

    # ── Stampa tabella ───────────────────────────────────────────────────────
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"{'Modello':<20} {'Cov@5':>7} {'Cov@10':>7} {'Cov@20':>7}  "
          f"{'Nov@5':>7} {'Nov@10':>7} {'Nov@20':>7}")
    print(sep)
    for model, m in results.items():
        print(
            f"{model:<20} "
            f"{m['coverage@5']:>7.4f} {m['coverage@10']:>7.4f} {m['coverage@20']:>7.4f}  "
            f"{m['novelty@5']:>7.4f} {m['novelty@10']:>7.4f} {m['novelty@20']:>7.4f}"
        )
    print(sep)

    # ── Salva ────────────────────────────────────────────────────────────────
    out_path = base / "coverage_novelty_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "catalog_size": catalog_size,
                "total_ratings": total_ratings,
                "n_test_users": len(test_users),
                "top_k_list": TOP_K_LIST,
                "gamma_hybrid": GAMMA,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[24] salvato: {out_path}")


if __name__ == "__main__":
    main()
