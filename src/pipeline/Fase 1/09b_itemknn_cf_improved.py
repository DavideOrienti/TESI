from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single

# =========================
# CONFIG
# =========================
DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

TRAIN_FILE = BASE / "ratings_train.csv"
VAL_FILE = BASE / "ratings_val.csv"
TEST_FILE = BASE / "ratings_test.csv"

OUTPUT_DIR = BASE / "baseline_itemknn_cf_improved"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_LIST = [5, 10, 20]

TOP_NEIGHBORS = 50
MIN_ITEM_INTERACTIONS = 5
MIN_COMMON_USERS = 3
SIM_SHRINKAGE = 10.0
MIN_POSITIVE_NEIGHBORS_FOR_SCORE = 1
USE_ONLY_POSITIVE_SIM = True

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)
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


def get_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("userId")["movieId"].apply(lambda x: set(map(int, x))).to_dict()


def build_user_means(train: pd.DataFrame) -> dict[int, float]:
    return train.groupby("userId")["rating"].mean().to_dict()


def build_centered_matrix(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    user_mean = df.groupby("userId")["rating"].mean().rename("user_mean")
    df = df.merge(user_mean, on="userId", how="left")
    df["rating_centered"] = df["rating"] - df["user_mean"]

    user_item = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating_centered",
        fill_value=0.0
    )
    return user_item


def get_valid_items(train: pd.DataFrame) -> list[int]:
    counts = train["movieId"].value_counts()
    return counts[counts >= MIN_ITEM_INTERACTIONS].index.tolist()


def build_item_similarity_improved(
    user_item_centered: pd.DataFrame,
    valid_items: list[int],
) -> pd.DataFrame:
    """
    Similarità item-item con:
    - cosine su rating centrati
    - filtro min_common_users
    - shrinkage
    """
    cols = [c for c in user_item_centered.columns if c in valid_items]
    item_matrix = user_item_centered[cols].T  # item x user

    item_ids = item_matrix.index.to_numpy()
    X = item_matrix.values

    sim_raw = cosine_similarity(X)
    nonzero_mask = (X != 0).astype(np.int8)
    common_counts = nonzero_mask @ nonzero_mask.T

    sim = sim_raw.copy()

    # Filtro common users
    sim[common_counts < MIN_COMMON_USERS] = 0.0

    # Shrinkage
    sim = sim * (common_counts / (common_counts + SIM_SHRINKAGE))

    np.fill_diagonal(sim, 0.0)

    sim_df = pd.DataFrame(sim, index=item_ids, columns=item_ids)
    return sim_df


def extract_top_neighbors(sim_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for item_id in sim_df.index:
        sims = sim_df.loc[item_id]

        if USE_ONLY_POSITIVE_SIM:
            sims = sims[sims > 0]
        else:
            sims = sims[sims != 0]

        top = sims.sort_values(ascending=False).head(TOP_NEIGHBORS)

        for neigh_id, value in top.items():
            rows.append({
                "movieId": int(item_id),
                "neighbor_movieId": int(neigh_id),
                "similarity": float(value)
            })

    return pd.DataFrame(rows)


def build_neighbors_dict(top_neighbors_df: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    neigh = {}
    for movie_id, g in top_neighbors_df.groupby("movieId"):
        neigh[int(movie_id)] = [
            (int(row["neighbor_movieId"]), float(row["similarity"]))
            for _, row in g.iterrows()
        ]
    return neigh


def get_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def popularity_recommendations_for_user(
    popularity_ranking: list[int],
    seen_items: set[int],
    k: int
) -> list[int]:
    recs = []
    for movie_id in popularity_ranking:
        if movie_id not in seen_items:
            recs.append(movie_id)
        if len(recs) >= k:
            break
    return recs


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
    seen_items = set(seen_ratings.keys())
    user_mean = float(user_means.get(user_id, 0.0))

    scored = []

    for item_id in candidate_items:
        if item_id in seen_items:
            continue

        score, used_neighbors = score_user_item_improved(
            user_id=user_id,
            target_item=item_id,
            user_seen_ratings=seen_ratings,
            user_mean=user_mean,
            neighbors_dict=neighbors_dict
        )

        if used_neighbors > 0:
            scored.append((item_id, score, used_neighbors))

    # ranking personalizzato
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    personalized = [item_id for item_id, _, _ in scored]

    # fallback popularity per completare la top-k
    final_recs = []
    used = set()

    for item_id in personalized:
        if item_id not in used:
            final_recs.append(item_id)
            used.add(item_id)
        if len(final_recs) >= k:
            return final_recs

    pop_fallback = popularity_recommendations_for_user(
        popularity_ranking=popularity_ranking,
        seen_items=seen_items.union(used),
        k=k - len(final_recs)
    )

    final_recs.extend(pop_fallback)
    return final_recs[:k]


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

    per_user_df = pd.DataFrame(rows)
    return summary, per_user_df


def main():
    for p in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"File non trovato: {p}")

    train, val, test = load_data()

    print_stats(train, "TRAIN")
    print_stats(val, "VAL")
    print_stats(test, "TEST")

    popularity_df = build_popularity_ranking(train, POPULARITY_MIN_VOTES)
    popularity_ranking = popularity_df["movieId"].tolist()

    valid_items = get_valid_items(train)
    print(f"\nvalid items for similarity: {len(valid_items)}")

    user_item_centered = build_centered_matrix(train)
    print(f"user-item centered matrix shape: {user_item_centered.shape}")

    sim_df = build_item_similarity_improved(user_item_centered, valid_items)
    top_neighbors_df = extract_top_neighbors(sim_df)
    neighbors_dict = build_neighbors_dict(top_neighbors_df)

    sim_path = OUTPUT_DIR / "item_similarity_top_neighbors.csv"
    top_neighbors_df.to_csv(sim_path, index=False)
    popularity_df.to_csv(OUTPUT_DIR / "fallback_popularity_ranking.csv", index=False)

    print(f"saved neighbors -> {sim_path}")

    print("\n=== SAMPLE NEIGHBORS ===")
    print(top_neighbors_df.head(10).to_string(index=False))

    candidate_items = sorted(train["movieId"].unique().tolist())

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

    val_per_user_path = OUTPUT_DIR / "val_per_user_metrics.csv"
    test_per_user_path = OUTPUT_DIR / "test_per_user_metrics.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.json"

    val_per_user.to_csv(val_per_user_path, index=False)
    test_per_user.to_csv(test_per_user_path, index=False)

    summary = {
        "config": {
            "dataset": DATASET,
            "top_k_list": TOP_K_LIST,
            "top_neighbors": TOP_NEIGHBORS,
            "min_item_interactions": MIN_ITEM_INTERACTIONS,
            "min_common_users": MIN_COMMON_USERS,
            "sim_shrinkage": SIM_SHRINKAGE,
            "popularity_min_votes": POPULARITY_MIN_VOTES,
            "model": "item_knn_cf_improved"
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