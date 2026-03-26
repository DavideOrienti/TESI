from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single


TOP_K_LIST = [5, 10, 20]

TRAIN_NAME = "ratings_train.csv"
VAL_NAME = "ratings_val.csv"
TEST_NAME = "ratings_test.csv"

INDEX_NAME = "movie_embeddings_index_v2.csv"
NEIGHBORS_NAME = "content_top_neighbors_v2.csv"

OUTPUT_DIR_NAME = "baseline_content_based_v2"


def load_data(base):
    train = pd.read_csv(base / TRAIN_NAME)
    val = pd.read_csv(base / VAL_NAME)
    test = pd.read_csv(base / TEST_NAME)

    index_df = pd.read_csv(base / INDEX_NAME)
    neighbors_df = pd.read_csv(base / NEIGHBORS_NAME)

    return train, val, test, index_df, neighbors_df


def build_movieid_to_index(index_df: pd.DataFrame) -> dict[int, int]:
    return {
        int(movie_id): int(idx)
        for idx, movie_id in enumerate(index_df["movieId"].tolist())
    }


def build_index_to_movieid(index_df: pd.DataFrame) -> dict[int, int]:
    return {
        int(idx): int(movie_id)
        for idx, movie_id in enumerate(index_df["movieId"].tolist())
    }


def build_neighbors_dict(neighbors_df: pd.DataFrame, index_to_movieid: dict[int, int]) -> dict[int, list[tuple[int, float]]]:
    """
    Converte il file top-neighbors basato su indici in:
    candidate_movieId -> [(neighbor_movieId, similarity), ...]
    """
    mapping: dict[int, list[tuple[int, float]]] = {}

    for movie_idx, g in neighbors_df.groupby("movie_idx"):
        candidate_movie_id = index_to_movieid.get(int(movie_idx))
        if candidate_movie_id is None:
            continue

        neighs = []
        for _, row in g.iterrows():
            neighbor_idx = int(row["neighbor_idx"])
            neighbor_movie_id = index_to_movieid.get(neighbor_idx)
            if neighbor_movie_id is None:
                continue

            sim = float(row["similarity"])
            neighs.append((neighbor_movie_id, sim))

        neighs.sort(key=lambda x: (-x[1], x[0]))
        mapping[candidate_movie_id] = neighs

    return mapping


def get_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def get_user_mean_ratings(train: pd.DataFrame) -> dict[int, float]:
    return {
        int(user_id): float(g["rating"].mean())
        for user_id, g in train.groupby("userId")
    }


def score_candidate(
    candidate_movie_id: int,
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    neighbors_dict: dict[int, list[tuple[int, float]]]
) -> tuple[float, int]:
    """
    Score content-based:
    - usa i top neighbors del candidato
    - aggrega i film già visti dall'utente
    - usa rating centrati
    - riporta lo score nello spazio originale dei rating
    """
    neighbors = neighbors_dict.get(candidate_movie_id, [])

    num = 0.0
    den = 0.0
    used_neighbors = 0

    for neighbor_movie_id, sim in neighbors:
        if neighbor_movie_id not in seen_ratings:
            continue

        if sim <= 0:
            continue

        rating = seen_ratings[neighbor_movie_id]
        adjusted_rating = rating - user_mean_rating

        num += sim * adjusted_rating
        den += abs(sim)
        used_neighbors += 1

    if den == 0.0:
        return user_mean_rating, used_neighbors

    score = user_mean_rating + (num / den)
    return score, used_neighbors


def recommend_top_k_for_user(
    user_id: int,
    candidate_items: list[int],
    user_seen_map: dict[int, dict[int, float]],
    user_mean_map: dict[int, float],
    neighbors_dict: dict[int, list[tuple[int, float]]],
    k: int
) -> tuple[list[int], int]:
    seen_ratings = user_seen_map.get(user_id, {})
    seen_items = set(seen_ratings.keys())
    user_mean_rating = user_mean_map.get(user_id, 0.0)

    scored = []
    with_neighbor_support = 0

    for movie_id in candidate_items:
        if movie_id in seen_items:
            continue

        score, used_neighbors = score_candidate(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            neighbors_dict=neighbors_dict
        )

        if used_neighbors > 0:
            with_neighbor_support += 1

        scored.append((movie_id, score, used_neighbors))

    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    top_items = [movie_id for movie_id, _, _ in scored[:k]]
    return top_items, with_neighbor_support


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    neighbors_dict: dict[int, list[tuple[int, float]]],
    top_k_list: list[int]
):
    user_seen_map = get_user_seen_ratings(train)
    user_mean_map = get_user_mean_ratings(train)
    max_k = max(top_k_list)

    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})

    rows = []
    coverage_support_counts = []
    recommendation_lengths = []

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])

        recs, with_neighbor_support = recommend_top_k_for_user(
            user_id=user_id,
            candidate_items=candidate_items,
            user_seen_map=user_seen_map,
            user_mean_map=user_mean_map,
            neighbors_dict=neighbors_dict,
            k=max_k
        )

        recommendation_lengths.append(len(recs))
        coverage_support_counts.append(with_neighbor_support)

        result_row = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
            "num_recommendations": len(recs),
            "num_candidates_with_neighbor_support": with_neighbor_support
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
    summary["avg_num_recommendations"] = float(np.mean(recommendation_lengths)) if recommendation_lengths else 0.0
    summary["avg_candidates_with_neighbor_support"] = float(np.mean(coverage_support_counts)) if coverage_support_counts else 0.0
    summary["users_with_less_than_5_recs"] = int(sum(x < 5 for x in recommendation_lengths))
    summary["users_with_less_than_10_recs"] = int(sum(x < 10 for x in recommendation_lengths))
    summary["users_with_less_than_20_recs"] = int(sum(x < 20 for x in recommendation_lengths))

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    s = load_settings()
    base = s.paths.processed
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, index_df, neighbors_df = load_data(base)

    print(f"[14] dataset={s.dataset}")
    print("train:", train.shape)
    print("val:", val.shape)
    print("test:", test.shape)
    print("index:", index_df.shape)
    print("neighbors:", neighbors_df.shape)

    movieid_to_index = build_movieid_to_index(index_df)
    index_to_movieid = build_index_to_movieid(index_df)
    neighbors_dict = build_neighbors_dict(neighbors_df, index_to_movieid)

    candidate_items = sorted(
        set(train["movieId"].unique()).intersection(set(index_df["movieId"].unique()))
    )

    print("candidate items:", len(candidate_items))
    print("items with stored neighbors:", len(neighbors_dict))

    val_summary, val_per_user = evaluate_split(
        train=train,
        eval_df=val,
        candidate_items=candidate_items,
        neighbors_dict=neighbors_dict,
        top_k_list=TOP_K_LIST
    )

    test_summary, test_per_user = evaluate_split(
        train=train,
        eval_df=test,
        candidate_items=candidate_items,
        neighbors_dict=neighbors_dict,
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
            "model": "content_based_top_neighbors_v2",
            "features": ["genres", "director", "actors_top5", "tags", "overview_en"],
            "title_used_in_similarity": False,
            "user_rating_centering": True,
            "neighbors_source": NEIGHBORS_NAME
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

    print("\n=== COVERAGE AUDIT ===")
    print(f"VAL avg recommendations: {val_summary['avg_num_recommendations']:.2f}")
    print(f"VAL avg candidates with neighbor support: {val_summary['avg_candidates_with_neighbor_support']:.2f}")
    print(f"VAL users with <5 recs: {val_summary['users_with_less_than_5_recs']}")
    print(f"VAL users with <10 recs: {val_summary['users_with_less_than_10_recs']}")
    print(f"VAL users with <20 recs: {val_summary['users_with_less_than_20_recs']}")

    print(f"TEST avg recommendations: {test_summary['avg_num_recommendations']:.2f}")
    print(f"TEST avg candidates with neighbor support: {test_summary['avg_candidates_with_neighbor_support']:.2f}")
    print(f"TEST users with <5 recs: {test_summary['users_with_less_than_5_recs']}")
    print(f"TEST users with <10 recs: {test_summary['users_with_less_than_10_recs']}")
    print(f"TEST users with <20 recs: {test_summary['users_with_less_than_20_recs']}")

    print(f"\nSaved val per-user metrics -> {val_per_user_path}")
    print(f"Saved test per-user metrics -> {test_per_user_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()