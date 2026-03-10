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

OUTPUT_DIR = BASE / "baseline_itemknn_cf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_LIST = [5, 10, 20]
TOP_NEIGHBORS = 50         # quanti vicini tenere per item
MIN_ITEM_INTERACTIONS = 2  # soglia minima nel train per tenere un film nella similarità
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)
    return train, val, test


def build_user_item_matrix(train: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce matrice user-item.
    Opzionalmente centra i rating per utente.
    """
    df = train.copy()

    if USE_RATING_CENTERING:
        user_mean = df.groupby("userId")["rating"].mean().rename("user_mean")
        df = df.merge(user_mean, on="userId", how="left")
        df["rating_for_sim"] = df["rating"] - df["user_mean"]
    else:
        df["rating_for_sim"] = df["rating"]

    user_item = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating_for_sim",
        fill_value=0.0
    )

    return user_item


def filter_items_for_similarity(train: pd.DataFrame) -> list[int]:
    """
    Tiene solo film con almeno MIN_ITEM_INTERACTIONS nel train
    per evitare similarità inutili su item troppo rari.
    """
    counts = train["movieId"].value_counts()
    valid_items = counts[counts >= MIN_ITEM_INTERACTIONS].index.tolist()
    return valid_items


def build_item_similarity(user_item: pd.DataFrame, valid_items: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Costruisce similarità item-item.
    Ritorna:
      - sim_df completa
      - top_neighbors_df con solo i top vicini per item
    """
    # limita agli item validi
    cols = [c for c in user_item.columns if c in valid_items]
    item_matrix = user_item[cols].T  # item x user

    sim = cosine_similarity(item_matrix.values)
    sim_df = pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)

    # azzera diagonale
    np.fill_diagonal(sim_df.values, 0.0)

    # tieni solo top neighbors per item
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

    top_neighbors_df = pd.DataFrame(rows)
    return sim_df, top_neighbors_df


def get_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    """
    userId -> {movieId: rating}
    """
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def get_candidate_items(train: pd.DataFrame) -> list[int]:
    return sorted(train["movieId"].unique().tolist())


def build_neighbors_dict(top_neighbors_df: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    """
    movieId -> [(neighbor_movieId, sim), ...]
    """
    neigh = {}
    for movie_id, g in top_neighbors_df.groupby("movieId"):
        neigh[int(movie_id)] = [
            (int(row["neighbor_movieId"]), float(row["similarity"]))
            for _, row in g.iterrows()
        ]
    return neigh


def score_user_item(
    user_seen_ratings: dict[int, float],
    target_item: int,
    neighbors_dict: dict[int, list[tuple[int, float]]]
) -> float:
    """
    Score item-based CF:
    usa i vicini del target_item e i rating dell'utente sui vicini.
    """
    neighbors = neighbors_dict.get(target_item, [])
    num = 0.0
    den = 0.0

    for neigh_item, sim in neighbors:
        if neigh_item in user_seen_ratings:
            rating = user_seen_ratings[neigh_item]
            num += sim * rating
            den += abs(sim)

    if den == 0.0:
        return 0.0

    return num / den


def recommend_top_k_for_user(
    user_id: int,
    candidate_items: list[int],
    user_seen_map: dict[int, dict[int, float]],
    neighbors_dict: dict[int, list[tuple[int, float]]],
    k: int
) -> list[int]:
    seen_ratings = user_seen_map.get(user_id, {})
    seen_items = set(seen_ratings.keys())

    scored = []
    for item_id in candidate_items:
        if item_id in seen_items:
            continue
        score = score_user_item(seen_ratings, item_id, neighbors_dict)
        if score > 0:
            scored.append((item_id, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [item_id for item_id, _ in scored[:k]]


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    candidate_items: list[int],
    neighbors_dict: dict[int, list[tuple[int, float]]],
    top_k_list: list[int]
) -> tuple[dict, pd.DataFrame]:
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
            user_id=user_id,
            candidate_items=candidate_items,
            user_seen_map=user_seen_map,
            neighbors_dict=neighbors_dict,
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

    valid_items = filter_items_for_similarity(train)
    print(f"\nvalid items for similarity: {len(valid_items)}")

    user_item = build_user_item_matrix(train)
    print(f"user-item matrix shape: {user_item.shape}")

    sim_df, top_neighbors_df = build_item_similarity(user_item, valid_items)
    neighbors_dict = build_neighbors_dict(top_neighbors_df)
    candidate_items = get_candidate_items(train)

    sim_path = OUTPUT_DIR / "item_similarity_top_neighbors.csv"
    top_neighbors_df.to_csv(sim_path, index=False)
    print(f"saved neighbors -> {sim_path}")

    print("\n=== SAMPLE NEIGHBORS ===")
    print(top_neighbors_df.head(10).to_string(index=False))

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
            "use_rating_centering": USE_RATING_CENTERING,
            "model": "item_knn_cf"
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