"""
Variante sparse di 10_matrix_factorization.py.

Stessa formulazione matematica (rating centrati per utente, TruncatedSVD,
N_FACTORS=50, RANDOM_STATE=42), ma:
- costruzione della matrice utente x item via scipy.sparse.csr_matrix
  invece di pivot_table denso (gli zeri impliciti della sparse coincidono
  col fill_value=0.0 del pivot denso: la matrice è matematicamente identica);
- nessuna ricostruzione densa n_users x n_items materializzata in memoria
  o su disco; lo scoring per utente è on-the-fly:
  score(candidati) = user_factors[u] @ item_factors[:, col_candidati] + user_mean.

Output in baseline_pure_svd_sparse/ (fattori, non la matrice ricostruita),
per non sovrascrivere baseline_pure_svd/ che backend e Fase_3 leggono.
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k, compute_rmse, compute_mae
from src.recommenders.popularity import build_popularity_ranking, recommend_popular

# =========================
# CONFIG
# =========================
TOP_K_LIST = [5, 10, 20]

N_FACTORS = 50
RANDOM_STATE = 42  # seed globale — vedere src/config/random_state.py
USE_USER_CENTERING = True
POPULARITY_MIN_VOTES = 10


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


def build_sparse_user_item_matrix(
    train: pd.DataFrame,
) -> tuple[csr_matrix, dict[int, int], dict[int, int], dict[int, float]]:
    """
    Costruisce la matrice utente x item centrata come CSR sparse, senza passare
    da pivot_table. Equivalente matematico di build_user_item_matrix (10 dense):
    stesso rating_model = rating - user_mean, stesso fill implicito a 0.0.
    """
    df = train.copy()

    if USE_USER_CENTERING:
        user_mean = df.groupby("userId")["rating"].mean().rename("user_mean")
        df = df.merge(user_mean, on="userId", how="left")
        df["rating_model"] = df["rating"] - df["user_mean"]
        user_means = user_mean.to_dict()
    else:
        df["rating_model"] = df["rating"]
        user_means = {int(u): 0.0 for u in df["userId"].unique()}

    user_ids = np.sort(df["userId"].unique())
    item_ids = np.sort(df["movieId"].unique())
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    item_index = {int(m): i for i, m in enumerate(item_ids)}

    rows = df["userId"].map(user_index).to_numpy()
    cols = df["movieId"].map(item_index).to_numpy()
    data = df["rating_model"].to_numpy(dtype=np.float64)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    return matrix, user_index, item_index, user_means


def fit_mf_sparse(matrix: csr_matrix, n_factors: int, random_state: int):
    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_  # shape (n_factors, n_items)

    explained = float(svd.explained_variance_ratio_.sum())
    return svd, user_factors, item_factors, explained


def get_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("userId")["movieId"].apply(lambda x: set(map(int, x))).to_dict()


def build_movie_ids_by_col(item_index: dict[int, int]) -> np.ndarray:
    """movieId ordinato per indice di colonna (posizione coincide con la colonna in item_factors)."""
    movie_ids_by_col = np.empty(len(item_index), dtype=np.int64)
    for movie_id, col in item_index.items():
        movie_ids_by_col[col] = movie_id
    return movie_ids_by_col


def recommend_svd_top_k_sparse(
    user_id: int,
    user_index: dict[int, int],
    item_index: dict[int, int],
    movie_ids_by_col: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items: set[int],
    popularity_ranking: list[int],
    k: int,
    score_offset: float = 0.0,
) -> list[int]:
    """
    Equivalente locale di recommend_svd_top_k (src/recommenders/collaborative.py),
    ma con scoring on-the-fly su fattori sparsi invece che lookup su matrice densa
    ricostruita. Stessa logica di ranking/fallback, nessuna riga del contratto
    backend toccata.
    """
    if k < 0:
        raise ValueError("k must be >= 0.")

    if user_id not in user_index:
        return recommend_popular(popularity_ranking or [], seen_items, k)

    u = user_index[user_id]

    seen_cols = {item_index[m] for m in seen_items if m in item_index}
    mask = np.ones(len(movie_ids_by_col), dtype=bool)
    if seen_cols:
        mask[list(seen_cols)] = False
    candidate_cols = np.nonzero(mask)[0]

    if candidate_cols.size == 0:
        return recommend_popular(popularity_ranking or [], seen_items, k)

    raw_scores = user_factors[u] @ item_factors[:, candidate_cols]
    candidate_movie_ids = movie_ids_by_col[candidate_cols]

    scores = {
        int(movie_id): float(score) + float(score_offset)
        for movie_id, score in zip(candidate_movie_ids, raw_scores)
    }

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    recommendations = [movie_id for movie_id, _ in ranked[:k]]

    if len(recommendations) < k:
        used = set(seen_items).union(recommendations)
        recommendations.extend(recommend_popular(popularity_ranking or [], used, k - len(recommendations)))

    return recommendations[:k]


def recommend_top_k_for_user(
    user_id: int,
    user_index: dict[int, int],
    item_index: dict[int, int],
    movie_ids_by_col: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items: set[int],
    user_mean: float,
    popularity_ranking: list[int],
    k: int
) -> list[int]:
    return recommend_svd_top_k_sparse(
        user_id=user_id,
        user_index=user_index,
        item_index=item_index,
        movie_ids_by_col=movie_ids_by_col,
        user_factors=user_factors,
        item_factors=item_factors,
        seen_items=seen_items,
        popularity_ranking=popularity_ranking,
        k=k,
        score_offset=user_mean if USE_USER_CENTERING else 0.0,
    )


def compute_rating_errors(
    eval_df: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_means: dict[int, float],
) -> tuple[float, float, int]:
    """RMSE e MAE sul set di valutazione, scoring on-the-fly per coppia (user, item)."""
    predictions = []
    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        movie_id = int(row["movieId"])
        true_rating = float(row["rating"])
        if user_id in user_index and movie_id in item_index:
            u = user_index[user_id]
            i = item_index[movie_id]
            pred = float(user_factors[u] @ item_factors[:, i]) + float(user_means.get(user_id, 0.0))
            pred = max(0.5, min(5.0, pred))
            predictions.append((true_rating, pred))
    return compute_rmse(predictions), compute_mae(predictions), len(predictions)


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    movie_ids_by_col: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_means: dict[int, float],
    popularity_ranking: list[int],
    top_k_list: list[int]
) -> tuple[dict, pd.DataFrame]:
    seen_map = get_seen_items(train)
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
        seen_items = seen_map.get(user_id, set())
        user_mean = float(user_means.get(user_id, 0.0))

        recs = recommend_top_k_for_user(
            user_id=user_id,
            user_index=user_index,
            item_index=item_index,
            movie_ids_by_col=movie_ids_by_col,
            user_factors=user_factors,
            item_factors=item_factors,
            seen_items=seen_items,
            user_mean=user_mean,
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
    output_dir = base / "baseline_pure_svd_sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[10-sparse] dataset={s.dataset}")
    print(f"[10-sparse] base path={base}")

    for p in [train_file, val_file, test_file]:
        if not p.exists():
            raise FileNotFoundError(f"File non trovato: {p}")

    train, val, test = load_data(train_file, val_file, test_file)

    print_stats(train, "TRAIN")
    print_stats(val, "VAL")
    print_stats(test, "TEST")

    popularity_df = build_popularity_ranking(train, POPULARITY_MIN_VOTES)
    popularity_ranking = popularity_df["movieId"].tolist()
    popularity_df.to_csv(output_dir / "fallback_popularity_ranking.csv", index=False)

    fit_start = time.perf_counter()

    sparse_matrix, user_index, item_index, user_means = build_sparse_user_item_matrix(train)
    print(f"\nsparse user-item matrix shape: {sparse_matrix.shape}, nnz={sparse_matrix.nnz}")

    max_rank = min(sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1)
    n_factors = min(N_FACTORS, max_rank)
    print(f"n_factors used: {n_factors}")

    _, user_factors, item_factors, explained = fit_mf_sparse(
        matrix=sparse_matrix,
        n_factors=n_factors,
        random_state=RANDOM_STATE
    )

    fit_time_s = time.perf_counter() - fit_start
    print(f"explained_variance_ratio_sum: {explained:.4f}")
    print(f"fit_time_s: {fit_time_s:.2f}")

    movie_ids_by_col = build_movie_ids_by_col(item_index)

    # Salva i fattori, MAI la ricostruzione densa n_users x n_items.
    np.save(output_dir / "user_factors.npy", user_factors)
    np.save(output_dir / "item_factors.npy", item_factors)
    pd.DataFrame(
        {"userId": list(user_index.keys()), "row": list(user_index.values())}
    ).to_csv(output_dir / "user_index.csv", index=False)
    pd.DataFrame(
        {"movieId": list(item_index.keys()), "col": list(item_index.values())}
    ).to_csv(output_dir / "item_index.csv", index=False)
    pd.Series(user_means, name="user_mean").rename_axis("userId").reset_index().to_csv(
        output_dir / "user_means.csv", index=False
    )
    print(f"saved factors -> {output_dir}")

    eval_start = time.perf_counter()

    val_summary, val_per_user = evaluate_split(
        train=train,
        eval_df=val,
        user_index=user_index,
        item_index=item_index,
        movie_ids_by_col=movie_ids_by_col,
        user_factors=user_factors,
        item_factors=item_factors,
        user_means=user_means,
        popularity_ranking=popularity_ranking,
        top_k_list=TOP_K_LIST
    )

    test_summary, test_per_user = evaluate_split(
        train=train,
        eval_df=test,
        user_index=user_index,
        item_index=item_index,
        movie_ids_by_col=movie_ids_by_col,
        user_factors=user_factors,
        item_factors=item_factors,
        user_means=user_means,
        popularity_ranking=popularity_ranking,
        top_k_list=TOP_K_LIST
    )

    val_rmse, val_mae, val_n = compute_rating_errors(val, user_index, item_index, user_factors, item_factors, user_means)
    test_rmse, test_mae, test_n = compute_rating_errors(test, user_index, item_index, user_factors, item_factors, user_means)

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
            "n_factors": n_factors,
            "random_state": RANDOM_STATE,
            "use_user_centering": USE_USER_CENTERING,
            "model": "pure_svd_truncated_sparse"
        },
        "train": {
            "explained_variance_ratio_sum": explained
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
