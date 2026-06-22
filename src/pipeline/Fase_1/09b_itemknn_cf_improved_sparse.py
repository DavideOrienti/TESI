"""
Variante sparse/a blocchi di 09b_itemknn_cf_improved.py.

Stessa aritmetica (coseno su rating centrati, filtro min_common_users,
shrinkage, USE_ONLY_POSITIVE_SIM, top-50 vicini per item), ma:
- matrice item x user costruita come scipy.sparse.csr_matrix (nessun
  pivot_table, nessuna matrice densa item x user);
- similarità item-item calcolata A BLOCCHI: mai materializzata la matrice
  item x item completa. Per ogni blocco B di item si calcola
  B_norm @ all_norm.T (denso solo sul blocco, block x n_items) per il
  coseno, e B_mask @ all_mask.T (matmul sparso delle maschere di non-zero)
  per i common_counts. Picco di memoria O(block x n_items).

Lo scoring/valutazione (score_user_item_improved, recommend_top_k_for_user,
compute_rating_errors_knn, evaluate_split) sono identici a 09b dense: una
volta costruito neighbors_dict nello stesso formato (via build_cf_neighbors_dict,
importato invariato da src.recommenders.collaborative), tutto il resto della
pipeline di valutazione non dipende da come la similarità è stata calcolata.

Output in baseline_itemknn_cf_improved_sparse/, per non sovrascrivere
baseline_itemknn_cf_improved/ che Fase_3/15 e Fase_3/16 leggono.
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.utils.io import load_settings
from src.recommenders.collaborative import build_cf_neighbors_dict, recommend_itemknn_top_k
from src.recommenders.popularity import build_popularity_ranking
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k, compute_rmse, compute_mae

# =========================
# CONFIG
# =========================
TOP_K_LIST = [5, 10, 20]

TOP_NEIGHBORS = 50
MIN_ITEM_INTERACTIONS = 5
MIN_COMMON_USERS = 3
SIM_SHRINKAGE = 10.0
MIN_POSITIVE_NEIGHBORS_FOR_SCORE = 1
USE_ONLY_POSITIVE_SIM = True

POPULARITY_MIN_VOTES = 10

BLOCK_SIZE = 200  # righe item processate per blocco; picco memoria O(BLOCK_SIZE x n_items)


def print_stats(df: pd.DataFrame, label: str) -> None:
    n_ratings = len(df)
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()
    density = n_ratings / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0

    print(f"\n=== {label} ===")
    print(f"ratings: {n_ratings}")
    print(f"users: {n_users}")
    print(f"items: {n_items}")
    print(f"density: {density:.6f}")


def load_data(train_file, val_file, test_file) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    return train, val, test


def build_user_means(train: pd.DataFrame) -> dict[int, float]:
    return train.groupby("userId")["rating"].mean().to_dict()


def get_valid_items(train: pd.DataFrame) -> list[int]:
    counts = train["movieId"].value_counts()
    return counts[counts >= MIN_ITEM_INTERACTIONS].index.tolist()


def build_centered_item_user_sparse(
    train: pd.DataFrame,
    valid_items: list[int],
) -> tuple[csr_matrix, dict[int, int], dict[int, int], np.ndarray]:
    """
    Costruisce la matrice item x user centrata come CSR, ristretta a valid_items.
    Equivalente sparse di build_centered_matrix (09b dense) + la selezione
    `cols = [c for c in user_item_centered.columns if c in valid_items]`:
    stesso rating_centered = rating - user_mean, stesse righe (item, ordine
    movieId ascendente) e stesse colonne (TUTTI gli utenti del train, ordine
    userId ascendente).
    """
    df = train.copy()
    user_mean = df.groupby("userId")["rating"].mean().rename("user_mean")
    df = df.merge(user_mean, on="userId", how="left")
    df["rating_centered"] = df["rating"] - df["user_mean"]

    valid_set = {int(m) for m in valid_items}
    df = df[df["movieId"].isin(valid_set)]

    user_ids = np.sort(train["userId"].unique())
    item_ids = np.sort(np.array(sorted(valid_set), dtype=np.int64))

    user_index = {int(u): i for i, u in enumerate(user_ids)}
    item_index = {int(m): i for i, m in enumerate(item_ids)}

    rows = df["movieId"].map(item_index).to_numpy()
    cols = df["userId"].map(user_index).to_numpy()
    data = df["rating_centered"].to_numpy(dtype=np.float64)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(item_ids), len(user_ids)))
    matrix.eliminate_zeros()  # rating == user_mean esatto => "nessuna interazione", come X != 0 nel denso

    return matrix, item_index, user_index, item_ids


def extract_top_neighbors_block(
    sim_block: np.ndarray,
    block_item_ids: np.ndarray,
    movie_ids_by_col: np.ndarray,
) -> list[dict]:
    """
    Estrae i top-TOP_NEIGHBORS vicini per ogni riga del blocco.
    Tie-break: similarità decrescente, poi movieId crescente (stesso esito
    di extract_top_neighbors dense per valori non in parità; per i pari
    sceglie deterministicamente l'ordine di indice originale).
    """
    rows = []
    for local_r in range(sim_block.shape[0]):
        item_id = int(block_item_ids[local_r])
        sims_row = sim_block[local_r]

        if USE_ONLY_POSITIVE_SIM:
            keep = sims_row > 0
        else:
            keep = sims_row != 0

        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            continue

        sims_vals = sims_row[idx]
        neighbor_ids = movie_ids_by_col[idx]

        order = np.lexsort((neighbor_ids, -sims_vals))
        top_order = order[:TOP_NEIGHBORS]

        for o in top_order:
            rows.append({
                "movieId": item_id,
                "neighbor_movieId": int(neighbor_ids[o]),
                "similarity": float(sims_vals[o]),
            })
    return rows


def build_item_similarity_blocks(
    item_user_sparse: csr_matrix,
    item_ids: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> pd.DataFrame:
    """
    Similarità item-item a blocchi, mai materializzando la matrice item x item
    completa. Replica l'aritmetica di build_item_similarity_improved (09b dense):
    coseno sui rating centrati, filtro min_common_users, shrinkage, diagonale a 0.
    """
    n_items = item_user_sparse.shape[0]

    item_user_norm = normalize(item_user_sparse, norm="l2", axis=1)

    item_user_mask = item_user_sparse.copy()
    item_user_mask.data = np.ones_like(item_user_mask.data)

    norm_t = item_user_norm.T.tocsr()
    mask_t = item_user_mask.T.tocsr()

    all_rows: list[dict] = []

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)

        cos_block = (item_user_norm[start:end] @ norm_t).toarray()
        common_block = (item_user_mask[start:end] @ mask_t).toarray()

        sim_block = cos_block.copy()
        sim_block[common_block < MIN_COMMON_USERS] = 0.0
        sim_block = sim_block * (common_block / (common_block + SIM_SHRINKAGE))

        for local_r in range(end - start):
            sim_block[local_r, start + local_r] = 0.0

        block_item_ids = item_ids[start:end]
        all_rows.extend(extract_top_neighbors_block(sim_block, block_item_ids, item_ids))

        print(f"[09b-sparse] processed items {end}/{n_items}")

    return pd.DataFrame(all_rows)


def get_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def score_user_item_improved(
    user_id: int,
    target_item: int,
    user_seen_ratings: dict[int, float],
    user_mean: float,
    neighbors_dict: dict[int, list[tuple[int, float]]]
) -> tuple[float, int]:
    neighbors = neighbors_dict.get(target_item, [])

    num = 0.0
    den = 0.0
    used_neighbors = 0

    for neigh_item, sim in neighbors:
        if neigh_item in user_seen_ratings:
            r_uj = user_seen_ratings[neigh_item]
            centered = r_uj - user_mean
            num += sim * centered
            den += abs(sim)
            used_neighbors += 1

    if den == 0.0 or used_neighbors < MIN_POSITIVE_NEIGHBORS_FOR_SCORE:
        return 0.0, used_neighbors

    pred = user_mean + (num / den)
    return pred, used_neighbors


def recommend_top_k_for_user(
    user_id: int,
    candidate_items: list[int],
    user_seen_map: dict[int, dict[int, float]],
    user_means: dict[int, float],
    neighbors_dict: dict[int, list[tuple[int, float]]],
    popularity_ranking: list[int],
    k: int
) -> list[int]:
    seen_ratings = user_seen_map.get(user_id, {})
    user_mean = float(user_means.get(user_id, 0.0))

    return recommend_itemknn_top_k(
        candidate_items=candidate_items,
        seen_ratings=seen_ratings,
        user_mean_rating=user_mean,
        cf_neighbors_dict=neighbors_dict,
        popularity_ranking=popularity_ranking,
        k=k,
        min_neighbors_for_personalized=MIN_POSITIVE_NEIGHBORS_FOR_SCORE,
    )


def compute_rating_errors_knn(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    neighbors_dict: dict[int, list[tuple[int, float]]],
) -> tuple[float, float, int]:
    """RMSE e MAE sul set di valutazione usando score_user_item_improved."""
    user_seen_map = get_user_seen_ratings(train)
    user_means = build_user_means(train)
    predictions = []
    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        true_rating = float(row["rating"])
        seen_ratings = user_seen_map.get(user_id, {})
        user_mean = float(user_means.get(user_id, 0.0))
        pred, used = score_user_item_improved(
            user_id=user_id,
            target_item=movie_id,
            user_seen_ratings=seen_ratings,
            user_mean=user_mean,
            neighbors_dict=neighbors_dict,
        )
        if used >= MIN_POSITIVE_NEIGHBORS_FOR_SCORE:
            pred = max(0.5, min(5.0, pred))
            predictions.append((true_rating, pred))
    return compute_rmse(predictions), compute_mae(predictions), len(predictions)


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    neighbors_dict: dict[int, list[tuple[int, float]]],
    popularity_ranking: list[int],
    top_k_list: list[int]
) -> tuple[dict, pd.DataFrame]:
    user_seen_map = get_user_seen_ratings(train)
    user_means = build_user_means(train)
    max_k = max(top_k_list)

    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})
    metrics.update({f"P@{k}": [] for k in top_k_list})
    metrics.update({f"R@{k}": [] for k in top_k_list})

    rows = []

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])

        recs = recommend_top_k_for_user(
            user_id=user_id,
            candidate_items=candidate_items,
            user_seen_map=user_seen_map,
            user_means=user_means,
            neighbors_dict=neighbors_dict,
            popularity_ranking=popularity_ranking,
            k=max_k
        )

        result_row = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
            "num_recommendations": len(recs)
        }

        for k in top_k_list:
            hr   = hit_rate_at_k(recs, ground_truth, k)
            ndcg = ndcg_at_k_single(recs, ground_truth, k)
            mrr  = mrr_at_k_single(recs, ground_truth, k)
            p    = precision_at_k(recs, ground_truth, k)
            r    = recall_at_k(recs, ground_truth, k)

            metrics[f"HR@{k}"].append(hr)
            metrics[f"NDCG@{k}"].append(ndcg)
            metrics[f"MRR@{k}"].append(mrr)
            metrics[f"P@{k}"].append(p)
            metrics[f"R@{k}"].append(r)

            result_row[f"HR@{k}"]   = hr
            result_row[f"NDCG@{k}"] = ndcg
            result_row[f"MRR@{k}"]  = mrr
            result_row[f"P@{k}"]    = p
            result_row[f"R@{k}"]    = r

        rows.append(result_row)

    summary = {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metrics.items()
    }
    summary["n_users_evaluated"] = int(len(eval_df))

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    s = load_settings()
    base = s.paths.processed

    train_file = base / "ratings_train.csv"
    val_file   = base / "ratings_val.csv"
    test_file  = base / "ratings_test.csv"
    output_dir = base / "baseline_itemknn_cf_improved_sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[09b-sparse] dataset={s.dataset}")
    print(f"[09b-sparse] base path={base}")

    for p in [train_file, val_file, test_file]:
        if not p.exists():
            raise FileNotFoundError(f"File non trovato: {p}")

    train, val, test = load_data(train_file, val_file, test_file)

    print_stats(train, "TRAIN")
    print_stats(val, "VAL")
    print_stats(test, "TEST")

    popularity_df = build_popularity_ranking(train, POPULARITY_MIN_VOTES)
    popularity_ranking = popularity_df["movieId"].tolist()

    valid_items = get_valid_items(train)
    print(f"\nvalid items for similarity: {len(valid_items)}")

    fit_start = time.perf_counter()

    item_user_sparse, item_index, user_index, item_ids = build_centered_item_user_sparse(train, valid_items)
    print(f"item-user sparse matrix shape: {item_user_sparse.shape}, nnz={item_user_sparse.nnz}")

    top_neighbors_df = build_item_similarity_blocks(item_user_sparse, item_ids, BLOCK_SIZE)
    neighbors_dict = build_cf_neighbors_dict(top_neighbors_df)

    fit_time_s = time.perf_counter() - fit_start
    print(f"fit_time_s: {fit_time_s:.2f}")

    sim_path = output_dir / "item_similarity_top_neighbors.csv"
    top_neighbors_df.to_csv(sim_path, index=False)
    popularity_df.to_csv(output_dir / "fallback_popularity_ranking.csv", index=False)

    print(f"saved neighbors -> {sim_path}")

    print("\n=== SAMPLE NEIGHBORS ===")
    print(top_neighbors_df.head(10).to_string(index=False))

    candidate_items = sorted(train["movieId"].unique().tolist())

    eval_start = time.perf_counter()

    val_summary, val_per_user = evaluate_split(
        train=train,
        eval_df=val,
        candidate_items=candidate_items,
        neighbors_dict=neighbors_dict,
        popularity_ranking=popularity_ranking,
        top_k_list=TOP_K_LIST
    )

    test_summary, test_per_user = evaluate_split(
        train=train,
        eval_df=test,
        candidate_items=candidate_items,
        neighbors_dict=neighbors_dict,
        popularity_ranking=popularity_ranking,
        top_k_list=TOP_K_LIST
    )

    val_rmse, val_mae, val_n = compute_rating_errors_knn(train, val, neighbors_dict)
    test_rmse, test_mae, test_n = compute_rating_errors_knn(train, test, neighbors_dict)

    eval_time_s = time.perf_counter() - eval_start
    print(f"eval_time_s: {eval_time_s:.2f}")

    val_summary["RMSE"] = round(val_rmse, 4)
    val_summary["MAE"] = round(val_mae, 4)
    val_summary["n_rating_predictions"] = val_n
    test_summary["RMSE"] = round(test_rmse, 4)
    test_summary["MAE"] = round(test_mae, 4)
    test_summary["n_rating_predictions"] = test_n

    val_per_user_path = output_dir / "val_per_user_metrics.csv"
    test_per_user_path = output_dir / "test_per_user_metrics.csv"
    summary_path = output_dir / "summary_metrics.json"

    val_per_user.to_csv(val_per_user_path, index=False)
    test_per_user.to_csv(test_per_user_path, index=False)

    summary = {
        "config": {
            "dataset": s.dataset,
            "top_k_list": TOP_K_LIST,
            "top_neighbors": TOP_NEIGHBORS,
            "min_item_interactions": MIN_ITEM_INTERACTIONS,
            "min_common_users": MIN_COMMON_USERS,
            "sim_shrinkage": SIM_SHRINKAGE,
            "popularity_min_votes": POPULARITY_MIN_VOTES,
            "block_size": BLOCK_SIZE,
            "model": "item_knn_cf_improved_sparse"
        },
        "fit_time_s": round(fit_time_s, 4),
        "eval_time_s": round(eval_time_s, 4),
        "validation": val_summary,
        "test": test_summary
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VALIDATION METRICS ===")
    for k in TOP_K_LIST:
        print(
            f"HR@{k}: {val_summary[f'HR@{k}']:.4f} | "
            f"NDCG@{k}: {val_summary[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {val_summary[f'MRR@{k}']:.4f}"
        )

    print("\n=== TEST METRICS ===")
    for k in TOP_K_LIST:
        print(
            f"HR@{k}: {test_summary[f'HR@{k}']:.4f} | "
            f"NDCG@{k}: {test_summary[f'NDCG@{k}']:.4f} | "
            f"MRR@{k}: {test_summary[f'MRR@{k}']:.4f}"
        )

    print(f"\n=== RATING PREDICTION (VAL) ===")
    print(f"RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | N: {val_n}")
    print(f"\n=== RATING PREDICTION (TEST) ===")
    print(f"RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | N: {test_n}")

    print(f"\nsaved val per-user metrics -> {val_per_user_path}")
    print(f"saved test per-user metrics -> {test_per_user_path}")
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
