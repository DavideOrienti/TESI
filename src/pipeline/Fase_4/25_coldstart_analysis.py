"""
Analisi cold-start: stratifica utenti del test set per quartile
di attività nel training e calcola HR@10 e NDCG@10 per quartile
su 4 modelli: Popularity, PureSVD, Content-Based, Hybrid SVD+CB.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k


def _build_index_to_movieid(index_df: pd.DataFrame) -> dict[int, int]:
    return {idx: int(mid) for idx, mid in enumerate(index_df["movieId"])}


def _build_neighbors(neighbors_df: pd.DataFrame,
                     index_to_movieid: dict[int, int]) -> dict[int, list[tuple[int, float]]]:
    """Costruisce {movieId: [(neighbor_movieId, sim), ...]} dal formato movie_idx/neighbor_idx."""
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


QUARTILES = ["Q1 (cold)", "Q2", "Q3", "Q4 (warm)"]


def _empty_quartile_dict():
    return {q: [] for q in QUARTILES}


def _summarize(hits: dict, ndcgs: dict) -> dict:
    return {
        q: {
            "HR@10":   round(float(np.mean(hits[q])),  4) if hits[q]  else 0.0,
            "NDCG@10": round(float(np.mean(ndcgs[q])), 4) if ndcgs[q] else 0.0,
            "n_users": len(hits[q]),
        }
        for q in QUARTILES
    }


def main():
    s = load_settings()
    base = s.paths.processed
    deploy = Path("data/deploy_artifacts")

    # ── Dati ────────────────────────────────────────────────
    ratings_train = pd.read_csv(base / "ratings_train.csv")
    ratings_test  = pd.read_csv(base / "ratings_test.csv")

    # Quartili basati sul numero di rating nel training
    user_train_counts = ratings_train.groupby("userId").size()
    q1 = user_train_counts.quantile(0.25)
    q2 = user_train_counts.quantile(0.50)
    q3 = user_train_counts.quantile(0.75)

    def get_quartile(n):
        if n <= q1:   return "Q1 (cold)"
        elif n <= q2: return "Q2"
        elif n <= q3: return "Q3"
        else:         return "Q4 (warm)"

    user_quartile = user_train_counts.apply(get_quartile)

    print(f"[25] dataset={s.dataset}")
    print(f"Soglie: Q1<={q1:.0f}, Q2<={q2:.0f}, Q3<={q3:.0f}")
    print("Distribuzione quartili:")
    print(user_quartile.value_counts().sort_index())

    seen_items  = ratings_train.groupby("userId")["movieId"].apply(set).to_dict()
    test_users  = ratings_test["userId"].unique().tolist()
    test_items  = ratings_test.set_index("userId")["movieId"].to_dict()

    K = 10
    results = {}

    # ── POPULARITY ──────────────────────────────────────────
    print("\nComputing Popularity...")
    pop_sorted = (
        ratings_train.groupby("movieId").size()
        .sort_values(ascending=False).index.tolist()
    )

    hits, ndcgs = _empty_quartile_dict(), _empty_quartile_dict()
    for uid in test_users:
        if uid not in user_quartile.index:
            continue
        q = user_quartile[uid]
        seen = seen_items.get(uid, set())
        true_item = test_items.get(uid)
        if true_item is None:
            continue
        recs = [i for i in pop_sorted if i not in seen][:K]
        hits[q].append(1.0 if true_item in recs else 0.0)
        rank = recs.index(true_item) + 1 if true_item in recs else None
        ndcgs[q].append(1.0 / np.log2(rank + 1) if rank else 0.0)
    results["Popularity"] = _summarize(hits, ndcgs)

    # ── PURE SVD ────────────────────────────────────────────
    print("Computing PureSVD...")
    svd_matrix = pd.read_parquet(deploy / "svd_matrix.parquet")
    svd_matrix.columns = [int(c) for c in svd_matrix.columns]
    svd_matrix.index   = [int(i) for i in svd_matrix.index]

    hits, ndcgs = _empty_quartile_dict(), _empty_quartile_dict()
    for uid in test_users:
        if uid not in user_quartile.index or uid not in svd_matrix.index:
            continue
        q = user_quartile[uid]
        seen = seen_items.get(uid, set())
        true_item = test_items.get(uid)
        if true_item is None:
            continue
        scores = {k: v for k, v in svd_matrix.loc[uid].to_dict().items() if k not in seen}
        recs = sorted(scores, key=scores.__getitem__, reverse=True)[:K]
        hits[q].append(1.0 if true_item in recs else 0.0)
        rank = recs.index(true_item) + 1 if true_item in recs else None
        ndcgs[q].append(1.0 / np.log2(rank + 1) if rank else 0.0)
    results["PureSVD"] = _summarize(hits, ndcgs)

    # ── CONTENT-BASED ───────────────────────────────────────
    print("Computing Content-Based...")
    index_df      = pd.read_csv(deploy / "movie_embeddings_index_v2.csv")
    neighbors_df  = pd.read_csv(deploy / "content_top_neighbors_v2.csv")
    idx_to_mid    = _build_index_to_movieid(index_df)
    neighbors     = _build_neighbors(neighbors_df, idx_to_mid)

    user_ratings_map = (
        ratings_train
        .groupby("userId")
        .apply(lambda g: g.set_index("movieId")["rating"].to_dict())
        .to_dict()
    )

    hits, ndcgs = _empty_quartile_dict(), _empty_quartile_dict()
    for uid in test_users:
        if uid not in user_quartile.index:
            continue
        q = user_quartile[uid]
        seen = seen_items.get(uid, set())
        true_item = test_items.get(uid)
        if true_item is None:
            continue
        user_ratings = user_ratings_map.get(uid, {})
        scores: dict[int, float] = {}
        for seed_id, seed_rating in user_ratings.items():
            if seed_rating < 3.5:
                continue
            for nb_id, sim in neighbors.get(seed_id, []):
                if nb_id not in seen:
                    scores[nb_id] = scores.get(nb_id, 0.0) + sim * seed_rating
        if not scores:
            hits[q].append(0.0)
            ndcgs[q].append(0.0)
            continue
        recs = sorted(scores, key=scores.__getitem__, reverse=True)[:K]
        hits[q].append(1.0 if true_item in recs else 0.0)
        rank = recs.index(true_item) + 1 if true_item in recs else None
        ndcgs[q].append(1.0 / np.log2(rank + 1) if rank else 0.0)
    results["Content-Based"] = _summarize(hits, ndcgs)

    # ── HYBRID SVD+CB ───────────────────────────────────────
    print("Computing Hybrid SVD+CB (gamma=0.7)...")
    GAMMA = 0.7

    hits, ndcgs = _empty_quartile_dict(), _empty_quartile_dict()
    for uid in test_users:
        if uid not in user_quartile.index or uid not in svd_matrix.index:
            continue
        q = user_quartile[uid]
        seen = seen_items.get(uid, set())
        true_item = test_items.get(uid)
        if true_item is None:
            continue

        # SVD normalizzato
        svd_s = {k: v for k, v in svd_matrix.loc[uid].to_dict().items() if k not in seen}
        sv_min, sv_max = min(svd_s.values()), max(svd_s.values())
        sv_rng = sv_max - sv_min if sv_max > sv_min else 1e-12
        svd_n = {k: (v - sv_min) / sv_rng for k, v in svd_s.items()}

        # CB normalizzato
        user_ratings = user_ratings_map.get(uid, {})
        cb_s: dict[int, float] = {}
        for seed_id, seed_rating in user_ratings.items():
            if seed_rating < 3.5:
                continue
            for nb_id, sim in neighbors.get(seed_id, []):
                if nb_id not in seen:
                    cb_s[nb_id] = cb_s.get(nb_id, 0.0) + sim * seed_rating
        cb_min = min(cb_s.values()) if cb_s else 0.0
        cb_max = max(cb_s.values()) if cb_s else 1.0
        cb_rng = cb_max - cb_min if cb_max > cb_min else 1e-12
        cb_n = {k: (v - cb_min) / cb_rng for k, v in cb_s.items()}

        all_items = set(svd_n) | set(cb_n)
        hybrid = {i: GAMMA * svd_n.get(i, 0.0) + (1 - GAMMA) * cb_n.get(i, 0.0) for i in all_items}
        recs = sorted(hybrid, key=hybrid.__getitem__, reverse=True)[:K]
        hits[q].append(1.0 if true_item in recs else 0.0)
        rank = recs.index(true_item) + 1 if true_item in recs else None
        ndcgs[q].append(1.0 / np.log2(rank + 1) if rank else 0.0)
    results["Hybrid_SVD_CB"] = _summarize(hits, ndcgs)

    # ── STAMPA RISULTATI ────────────────────────────────────
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"{'Modello':<18} {'Q1 cold':>10} {'Q2':>10} {'Q3':>10} {'Q4 warm':>10}")
    print(f"{'':18} {'HR@10':>10} {'HR@10':>10} {'HR@10':>10} {'HR@10':>10}")
    print(sep)
    for model, qdata in results.items():
        print(f"{model:<18} "
              f"{qdata['Q1 (cold)']['HR@10']:>10.4f} "
              f"{qdata['Q2']['HR@10']:>10.4f} "
              f"{qdata['Q3']['HR@10']:>10.4f} "
              f"{qdata['Q4 (warm)']['HR@10']:>10.4f}")

    print(f"\n{sep}")
    print(f"{'Modello':<18} {'Q1 cold':>10} {'Q2':>10} {'Q3':>10} {'Q4 warm':>10}")
    print(f"{'':18} {'NDCG@10':>10} {'NDCG@10':>10} {'NDCG@10':>10} {'NDCG@10':>10}")
    print(sep)
    for model, qdata in results.items():
        print(f"{model:<18} "
              f"{qdata['Q1 (cold)']['NDCG@10']:>10.4f} "
              f"{qdata['Q2']['NDCG@10']:>10.4f} "
              f"{qdata['Q3']['NDCG@10']:>10.4f} "
              f"{qdata['Q4 (warm)']['NDCG@10']:>10.4f}")

    print(f"\nN utenti per quartile:")
    for q in QUARTILES:
        n = results["PureSVD"][q]["n_users"]
        print(f"  {q}: {n} utenti")

    # ── Salva JSON ──────────────────────────────────────────
    out = base / "coldstart_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"K": K, "gamma_hybrid": GAMMA,
                               "q1": q1, "q2": q2, "q3": q3},
                   "results": results}, f, indent=2)
    print(f"\n[25] salvato: {out}")


if __name__ == "__main__":
    main()
