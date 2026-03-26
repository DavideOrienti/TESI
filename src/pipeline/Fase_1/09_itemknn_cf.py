from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
TOP_K_LIST = [5, 10, 20]
TOP_NEIGHBORS = 50
MIN_ITEM_INTERACTIONS = 2
USE_RATING_CENTERING = True


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


def load_data(train_file, val_file, test_file):
    return (
        pd.read_csv(train_file),
        pd.read_csv(val_file),
        pd.read_csv(test_file)
    )


# =========================
# MATRICE USER-ITEM
# =========================
def build_user_item_matrix(train: pd.DataFrame):
    df = train.copy()

    if USE_RATING_CENTERING:
        user_mean = df.groupby("userId")["rating"].mean().rename("user_mean")
        df = df.merge(user_mean, on="userId", how="left")
        df["rating_for_sim"] = df["rating"] - df["user_mean"]
    else:
        df["user_mean"] = 0.0
        df["rating_for_sim"] = df["rating"]

    user_item = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating_for_sim",
        fill_value=0.0
    )

    # salva anche le medie utente per lo scoring
    user_mean_map = df.groupby("userId")["rating"].mean().to_dict()

    return user_item, user_mean_map


def filter_items_for_similarity(train: pd.DataFrame):
    counts = train["movieId"].value_counts()
    return counts[counts >= MIN_ITEM_INTERACTIONS].index.tolist()


# =========================
# SIMILARITÀ
# =========================
def build_item_similarity(user_item, valid_items):
    cols = [c for c in user_item.columns if c in valid_items]
    item_matrix = user_item[cols].T

    sim = cosine_similarity(item_matrix.values)
    sim_df = pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)

    np.fill_diagonal(sim_df.values, 0.0)

    rows = []
    for item_id in sim_df.index:
        sims = sim_df.loc[item_id]
        top = sims.sort_values(ascending=False).head(TOP_NEIGHBORS)

        for neigh_id, value in top.items():
            if value > 0:
                rows.append({
                    "movieId": int(item_id),
                    "neighbor_movieId": int(neigh_id),
                    "similarity": float(value)
                })

    return sim_df, pd.DataFrame(rows)


def build_neighbors_dict(top_neighbors_df):
    neigh = {}
    for movie_id, g in top_neighbors_df.groupby("movieId"):
        neigh[int(movie_id)] = [
            (int(row["neighbor_movieId"]), float(row["similarity"]))
            for _, row in g.iterrows()
        ]
    return neigh


# =========================
# USER DATA
# =========================
def get_user_seen_ratings(train):
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def get_candidate_items(train):
    return sorted(train["movieId"].unique().tolist())


# =========================
# SCORING (FIX IMPORTANTE)
# =========================
def score_user_item(user_id, user_seen_ratings, user_mean_map, target_item, neighbors_dict):
    neighbors = neighbors_dict.get(target_item, [])
    user_mean = user_mean_map.get(user_id, 0.0)

    num = 0.0
    den = 0.0

    for neigh_item, sim in neighbors:
        if neigh_item in user_seen_ratings:
            rating = user_seen_ratings[neigh_item]

            if USE_RATING_CENTERING:
                rating = rating - user_mean  # ✅ FIX

            num += sim * rating
            den += abs(sim)

    if den == 0.0:
        return 0.0

    score = num / den

    # opzionale: riporta nello spazio originale
    if USE_RATING_CENTERING:
        score += user_mean

    return score


# =========================
# RECOMMENDATION
# =========================
def recommend_top_k_for_user(user_id, candidate_items, user_seen_map, user_mean_map, neighbors_dict, k):
    seen_ratings = user_seen_map.get(user_id, {})
    seen_items = set(seen_ratings.keys())

    scored = []
    for item_id in candidate_items:
        if item_id in seen_items:
            continue

        score = score_user_item(
            user_id,
            seen_ratings,
            user_mean_map,
            item_id,
            neighbors_dict
        )

        if score > 0:
            scored.append((item_id, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [item_id for item_id, _ in scored[:k]]


# =========================
# EVALUATION
# =========================
def evaluate_split(train, eval_df, candidate_items, neighbors_dict, user_mean_map, top_k_list):
    user_seen_map = get_user_seen_ratings(train)
    max_k = max(top_k_list)

    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})

    rows = []

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])

        recs = recommend_top_k_for_user(
            user_id,
            candidate_items,
            user_seen_map,
            user_mean_map,
            neighbors_dict,
            max_k
        )

        result_row = {"userId": user_id}

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

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    summary["n_users_evaluated"] = len(eval_df)

    return summary, pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    s = load_settings()
    base = s.paths.processed

    train_file = base / "ratings_train.csv"
    val_file = base / "ratings_val.csv"
    test_file = base / "ratings_test.csv"

    output_dir = base / "baseline_itemknn_cf"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_data(train_file, val_file, test_file)

    print_stats(train, "TRAIN")

    valid_items = filter_items_for_similarity(train)
    user_item, user_mean_map = build_user_item_matrix(train)

    sim_df, top_neighbors_df = build_item_similarity(user_item, valid_items)
    neighbors_dict = build_neighbors_dict(top_neighbors_df)
    candidate_items = get_candidate_items(train)

    val_summary, _ = evaluate_split(train, val, candidate_items, neighbors_dict, user_mean_map, TOP_K_LIST)
    test_summary, _ = evaluate_split(train, test, candidate_items, neighbors_dict, user_mean_map, TOP_K_LIST)

    print("\n=== TEST METRICS ===")
    for k in TOP_K_LIST:
        print(f"HR@{k}: {test_summary[f'HR@{k}']:.4f}")


if __name__ == "__main__":
    main()