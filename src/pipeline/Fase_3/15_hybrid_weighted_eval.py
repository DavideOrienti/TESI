from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single


TOP_K_LIST = [5, 10, 20]
ALPHAS = [0.2, 0.4, 0.5, 0.6, 0.8]

TRAIN_NAME = "ratings_train.csv"
VAL_NAME = "ratings_val.csv"
TEST_NAME = "ratings_test.csv"

# CF improved artifacts
CF_NEIGHBORS_NAME = "baseline_itemknn_cf_improved/item_similarity_top_neighbors.csv"

# Content artifacts
CONTENT_INDEX_NAME = "movie_embeddings_index_v2.csv"
CONTENT_NEIGHBORS_NAME = "content_top_neighbors_v2.csv"

OUTPUT_DIR_NAME = "hybrid_weighted_v1"


def load_data(base):
    train = pd.read_csv(base / TRAIN_NAME)
    val = pd.read_csv(base / VAL_NAME)
    test = pd.read_csv(base / TEST_NAME)

    cf_neighbors_df = pd.read_csv(base / CF_NEIGHBORS_NAME)
    content_index_df = pd.read_csv(base / CONTENT_INDEX_NAME)
    content_neighbors_df = pd.read_csv(base / CONTENT_NEIGHBORS_NAME)

    return train, val, test, cf_neighbors_df, content_index_df, content_neighbors_df


def build_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def build_user_mean_ratings(train: pd.DataFrame) -> dict[int, float]:
    return {
        int(user_id): float(g["rating"].mean())
        for user_id, g in train.groupby("userId")
    }


def build_cf_neighbors_dict(cf_neighbors_df: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    neigh = {}
    for movie_id, g in cf_neighbors_df.groupby("movieId"):
        rows = [
            (int(row["neighbor_movieId"]), float(row["similarity"]))
            for _, row in g.iterrows()
        ]
        rows.sort(key=lambda x: (-x[1], x[0]))
        neigh[int(movie_id)] = rows
    return neigh


def build_index_to_movieid(index_df: pd.DataFrame) -> dict[int, int]:
    return {
        int(idx): int(movie_id)
        for idx, movie_id in enumerate(index_df["movieId"].tolist())
    }


def build_content_neighbors_dict(
    content_neighbors_df: pd.DataFrame,
    index_to_movieid: dict[int, int]
) -> dict[int, list[tuple[int, float]]]:
    """
    candidate_movieId -> [(neighbor_movieId, sim), ...]
    """
    neigh = {}
    for movie_idx, g in content_neighbors_df.groupby("movie_idx"):
        candidate_movie_id = index_to_movieid.get(int(movie_idx))
        if candidate_movie_id is None:
            continue

        rows = []
        for _, row in g.iterrows():
            neighbor_idx = int(row["neighbor_idx"])
            neighbor_movie_id = index_to_movieid.get(neighbor_idx)
            if neighbor_movie_id is None:
                continue

            rows.append((neighbor_movie_id, float(row["similarity"])))

        rows.sort(key=lambda x: (-x[1], x[0]))
        neigh[candidate_movie_id] = rows

    return neigh


def minmax_normalize_scores(score_dict: dict[int, float]) -> dict[int, float]:
    if not score_dict:
        return {}

    values = np.array(list(score_dict.values()), dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())

    if vmax - vmin < 1e-12:
        return {k: 0.0 for k in score_dict}

    return {
        k: float((v - vmin) / (vmax - vmin))
        for k, v in score_dict.items()
    }


# =========================
# CF SCORE
# =========================
def score_candidate_cf(
    candidate_movie_id: int,
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    cf_neighbors_dict: dict[int, list[tuple[int, float]]]
) -> float:
    neighbors = cf_neighbors_dict.get(candidate_movie_id, [])

    num = 0.0
    den = 0.0

    for neighbor_movie_id, sim in neighbors:
        if neighbor_movie_id not in seen_ratings:
            continue

        rating = seen_ratings[neighbor_movie_id]
        centered = rating - user_mean_rating

        num += sim * centered
        den += abs(sim)

    if den == 0.0:
        return user_mean_rating

    return user_mean_rating + (num / den)


# =========================
# CONTENT SCORE
# =========================
def score_candidate_content(
    candidate_movie_id: int,
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    content_neighbors_dict: dict[int, list[tuple[int, float]]]
) -> float:
    neighbors = content_neighbors_dict.get(candidate_movie_id, [])

    num = 0.0
    den = 0.0

    for neighbor_movie_id, sim in neighbors:
        if neighbor_movie_id not in seen_ratings:
            continue

        if sim <= 0:
            continue

        rating = seen_ratings[neighbor_movie_id]
        centered = rating - user_mean_rating

        num += sim * centered
        den += abs(sim)

    if den == 0.0:
        return user_mean_rating

    return user_mean_rating + (num / den)


def build_user_model_scores(
    user_id: int,
    candidate_items: list[int],
    user_seen_map: dict[int, dict[int, float]],
    user_mean_map: dict[int, float],
    cf_neighbors_dict: dict[int, list[tuple[int, float]]],
    content_neighbors_dict: dict[int, list[tuple[int, float]]]
) -> tuple[dict[int, float], dict[int, float]]:
    seen_ratings = user_seen_map.get(user_id, {})
    seen_items = set(seen_ratings.keys())
    user_mean_rating = user_mean_map.get(user_id, 0.0)

    cf_scores = {}
    content_scores = {}

    for movie_id in candidate_items:
        if movie_id in seen_items:
            continue

        cf_scores[movie_id] = score_candidate_cf(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            cf_neighbors_dict=cf_neighbors_dict
        )

        content_scores[movie_id] = score_candidate_content(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            content_neighbors_dict=content_neighbors_dict
        )

    return cf_scores, content_scores


def recommend_top_k_hybrid(
    cf_scores: dict[int, float],
    content_scores: dict[int, float],
    alpha: float,
    k: int
) -> list[int]:
    cf_norm = minmax_normalize_scores(cf_scores)
    content_norm = minmax_normalize_scores(content_scores)

    all_items = set(cf_norm.keys()).union(content_norm.keys())

    final_scores = []
    for movie_id in all_items:
        scf = cf_norm.get(movie_id, 0.0)
        scontent = content_norm.get(movie_id, 0.0)
        sfinal = alpha * scf + (1.0 - alpha) * scontent
        final_scores.append((movie_id, sfinal))

    final_scores.sort(key=lambda x: (-x[1], x[0]))
    return [movie_id for movie_id, _ in final_scores[:k]]


def evaluate_split_for_alpha(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    cf_neighbors_dict: dict[int, list[tuple[int, float]]],
    content_neighbors_dict: dict[int, list[tuple[int, float]]],
    alpha: float,
    top_k_list: list[int]
):
    user_seen_map = build_user_seen_ratings(train)
    user_mean_map = build_user_mean_ratings(train)
    max_k = max(top_k_list)

    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})

    rows = []

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])

        cf_scores, content_scores = build_user_model_scores(
            user_id=user_id,
            candidate_items=candidate_items,
            user_seen_map=user_seen_map,
            user_mean_map=user_mean_map,
            cf_neighbors_dict=cf_neighbors_dict,
            content_neighbors_dict=content_neighbors_dict
        )

        recs = recommend_top_k_hybrid(
            cf_scores=cf_scores,
            content_scores=content_scores,
            alpha=alpha,
            k=max_k
        )

        result_row = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
            "alpha": alpha,
            "num_recommendations": len(recs),
        }

        for k in top_k_list:
            hr = hit_rate_at_k(recs, ground_truth, k)
            ndcg = ndcg_at_k_single(recs, ground_truth, k)
            mrr = mrr_at_k_single(recs, ground_truth, k)

            metrics[f"HR@{k}"].append(hr)
            metrics[f"NDCG@{k}"].append(ndcg)
            metrics[f"MRR@{k}"].append(mrr)

            result_row[f"HR@{k}"] = hr
            result_row[f"NDCG@{k}"] = ndcg
            result_row[f"MRR@{k}"] = mrr

        rows.append(result_row)

    summary = {
        metric: float(np.mean(values)) if values else 0.0
        for metric, values in metrics.items()
    }
    summary["n_users_evaluated"] = int(len(eval_df))
    summary["alpha"] = float(alpha)

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    s = load_settings()
    base = s.paths.processed
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, cf_neighbors_df, content_index_df, content_neighbors_df = load_data(base)

    print(f"[15] dataset={s.dataset}")
    print(f"train: {train.shape}")
    print(f"val: {val.shape}")
    print(f"test: {test.shape}")
    print(f"cf_neighbors: {cf_neighbors_df.shape}")
    print(f"content_index: {content_index_df.shape}")
    print(f"content_neighbors: {content_neighbors_df.shape}")

    cf_neighbors_dict = build_cf_neighbors_dict(cf_neighbors_df)

    index_to_movieid = build_index_to_movieid(content_index_df)
    content_neighbors_dict = build_content_neighbors_dict(content_neighbors_df, index_to_movieid)

    candidate_items = sorted(
        set(train["movieId"].unique()).intersection(set(content_index_df["movieId"].unique()))
    )
    print(f"candidate items: {len(candidate_items)}")

    all_results = []
    best_alpha = None
    best_val_ndcg10 = -1.0

    for alpha in ALPHAS:
        print(f"\n=== ALPHA = {alpha:.2f} ===")

        val_summary, val_per_user = evaluate_split_for_alpha(
            train=train,
            eval_df=val,
            candidate_items=candidate_items,
            cf_neighbors_dict=cf_neighbors_dict,
            content_neighbors_dict=content_neighbors_dict,
            alpha=alpha,
            top_k_list=TOP_K_LIST
        )

        test_summary, test_per_user = evaluate_split_for_alpha(
            train=train,
            eval_df=test,
            candidate_items=candidate_items,
            cf_neighbors_dict=cf_neighbors_dict,
            content_neighbors_dict=content_neighbors_dict,
            alpha=alpha,
            top_k_list=TOP_K_LIST
        )

        print(
            f"VAL  HR@10={val_summary['HR@10']:.4f} | "
            f"NDCG@10={val_summary['NDCG@10']:.4f} | "
            f"MRR@10={val_summary['MRR@10']:.4f}"
        )
        print(
            f"TEST HR@10={test_summary['HR@10']:.4f} | "
            f"NDCG@10={test_summary['NDCG@10']:.4f} | "
            f"MRR@10={test_summary['MRR@10']:.4f}"
        )

        val_per_user.to_csv(output_dir / f"val_per_user_alpha_{alpha:.2f}.csv", index=False)
        test_per_user.to_csv(output_dir / f"test_per_user_alpha_{alpha:.2f}.csv", index=False)

        all_results.append({
            "alpha": alpha,
            **{f"val_{k}": v for k, v in val_summary.items()},
            **{f"test_{k}": v for k, v in test_summary.items()},
        })

        if val_summary["NDCG@10"] > best_val_ndcg10:
            best_val_ndcg10 = val_summary["NDCG@10"]
            best_alpha = alpha

    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / "alpha_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    summary = {
        "config": {
            "dataset": s.dataset,
            "alphas": ALPHAS,
            "top_k_list": TOP_K_LIST,
            "model": "hybrid_weighted_v1",
            "cf_source": CF_NEIGHBORS_NAME,
            "content_source": CONTENT_NEIGHBORS_NAME,
            "normalization": "per-user minmax"
        },
        "best_alpha_by_val_ndcg10": best_alpha,
        "results_table_file": str(results_csv)
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST CONFIG ===")
    print(f"best alpha (by VAL NDCG@10): {best_alpha}")
    print(f"saved table -> {results_csv}")
    print(f"saved summary -> {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()