from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k

# =========================
# CONFIG
# =========================
TOP_K_LIST = [5, 10, 20]

N_FACTORS = 50
RANDOM_STATE = 42
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


def build_popularity_ranking(train: pd.DataFrame, min_votes: int) -> pd.DataFrame:
    item_stats = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        )
        .reset_index()
    )

    c_global = train["rating"].mean()

    item_stats["pop_score"] = (
        (item_stats["rating_count"] / (item_stats["rating_count"] + min_votes)) * item_stats["rating_mean"]
        + (min_votes / (item_stats["rating_count"] + min_votes)) * c_global
    )

    item_stats = item_stats.sort_values(
        ["pop_score", "rating_count", "movieId"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return item_stats


def build_user_item_matrix(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    df = train.copy()

    if USE_USER_CENTERING:
        user_mean = df.groupby("userId")["rating"].mean().rename("user_mean")
        df = df.merge(user_mean, on="userId", how="left")
        df["rating_model"] = df["rating"] - df["user_mean"]
        user_means = user_mean.to_dict()
    else:
        df["rating_model"] = df["rating"]
        user_means = {int(u): 0.0 for u in df["userId"].unique()}

    user_item = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating_model",
        fill_value=0.0
    )

    return user_item, user_means


def fit_mf(user_item: pd.DataFrame, n_factors: int, random_state: int):
    X = user_item.values

    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    user_factors = svd.fit_transform(X)
    item_factors = svd.components_.T  # items x factors

    reconstructed = user_factors @ svd.components_

    pred_df = pd.DataFrame(
        reconstructed,
        index=user_item.index,
        columns=user_item.columns
    )

    explained = float(svd.explained_variance_ratio_.sum())
    return svd, pred_df, explained


def get_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("userId")["movieId"].apply(lambda x: set(map(int, x))).to_dict()


def recommend_top_k_for_user(
    user_id: int,
    pred_df: pd.DataFrame,
    seen_items: set[int],
    user_mean: float,
    popularity_ranking: list[int],
    k: int
) -> list[int]:
    if user_id not in pred_df.index:
        # fallback totale
        recs = []
        for movie_id in popularity_ranking:
            if movie_id not in seen_items:
                recs.append(movie_id)
            if len(recs) >= k:
                return recs
        return recs

    user_scores = pred_df.loc[user_id].copy()

    if USE_USER_CENTERING:
        user_scores = user_scores + user_mean

    user_scores = user_scores.drop(labels=list(seen_items), errors="ignore")
    user_scores = user_scores.sort_values(ascending=False)

    recs = list(map(int, user_scores.head(k).index.tolist()))

    if len(recs) < k:
        used = set(recs).union(seen_items)
        for movie_id in popularity_ranking:
            if movie_id not in used:
                recs.append(movie_id)
            if len(recs) >= k:
                break

    return recs[:k]


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    pred_df: pd.DataFrame,
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
            pred_df=pred_df,
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
    output_dir = base / "baseline_pure_svd"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[10] dataset={s.dataset}")
    print(f"[10] base path={base}")

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

    user_item, user_means = build_user_item_matrix(train)
    print(f"\nuser-item matrix shape: {user_item.shape}")

    max_rank = min(user_item.shape[0] - 1, user_item.shape[1] - 1)
    n_factors = min(N_FACTORS, max_rank)
    print(f"n_factors used: {n_factors}")

    svd, pred_df, explained = fit_mf(
        user_item=user_item,
        n_factors=n_factors,
        random_state=RANDOM_STATE
    )

    print(f"explained_variance_ratio_sum: {explained:.4f}")

    pred_path = output_dir / "predicted_scores_matrix.csv"
    pred_df.to_csv(pred_path)
    print(f"saved predicted scores -> {pred_path}")

    val_summary, val_per_user = evaluate_split(
        train=train,
        eval_df=val,
        pred_df=pred_df,
        user_means=user_means,
        popularity_ranking=popularity_ranking,
        top_k_list=TOP_K_LIST
    )

    test_summary, test_per_user = evaluate_split(
        train=train,
        eval_df=test,
        pred_df=pred_df,
        user_means=user_means,
        popularity_ranking=popularity_ranking,
        top_k_list=TOP_K_LIST
    )

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
            "model": "pure_svd_truncated"
        },
        "train": {
            "explained_variance_ratio_sum": explained
        },
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

    print(f"\nsaved val per-user metrics -> {val_per_user_path}")
    print(f"saved test per-user metrics -> {test_per_user_path}")
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()