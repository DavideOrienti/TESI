from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single

# =========================
# CONFIG
# =========================
DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

TRAIN_FILE = BASE / "ratings_train.csv"
VAL_FILE = BASE / "ratings_val.csv"
TEST_FILE = BASE / "ratings_test.csv"

OUTPUT_DIR = BASE / "baseline_popularity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_LIST = [5, 10, 20]
MIN_VOTES = 10   # m nella formula IMDb


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)
    return train, val, test


def build_popularity_ranking(train: pd.DataFrame, min_votes: int) -> pd.DataFrame:
    """
    Costruisce ranking globale di popolarità sul train usando weighted rating stile IMDb.
    """
    item_stats = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            rating_mean=("rating", "mean")
        )
        .reset_index()
    )

    c_global = train["rating"].mean()

    # Weighted rating
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
    """
    Mappa userId -> set(movieId) già visti nel train.
    """
    seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    return seen


def recommend_top_k_for_user(
    ranking_movie_ids: list[int],
    seen_items: set[int],
    k: int
) -> list[int]:
    recs = []
    for movie_id in ranking_movie_ids:
        if movie_id not in seen_items:
            recs.append(movie_id)
        if len(recs) >= k:
            break
    return recs


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    top_k_list: list[int]
) -> tuple[dict, pd.DataFrame]:
    """
    Valuta il ranking popularity su val o test.
    Assunzione: eval_df contiene una riga per utente (vero nel tuo split leave-last-out).
    """
    seen_map = get_seen_items(train)
    ranking_movie_ids = ranking_df["movieId"].tolist()

    rows = []
    metrics = {
        f"HR@{k}": [] for k in top_k_list
    }
    metrics.update({
        f"NDCG@{k}": [] for k in top_k_list
    })
    metrics.update({
        f"MRR@{k}": [] for k in top_k_list
    })

    for _, row in eval_df.iterrows():
        user_id = int(row["userId"])
        ground_truth = int(row["movieId"])
        seen_items = seen_map.get(user_id, set())

        max_k = max(top_k_list)
        recs = recommend_top_k_for_user(ranking_movie_ids, seen_items, max_k)

        user_result = {
            "userId": user_id,
            "ground_truth_movieId": ground_truth,
        }

        for k in top_k_list:
            hr = hit_rate_at_k(recs, ground_truth, k)
            ndcg = ndcg_at_k_single(recs, ground_truth, k)
            mrr = mrr_at_k_single(recs, ground_truth, k)

            metrics[f"HR@{k}"].append(hr)
            metrics[f"NDCG@{k}"].append(ndcg)
            metrics[f"MRR@{k}"].append(mrr)

            user_result[f"HR@{k}"] = hr
            user_result[f"NDCG@{k}"] = ndcg
            user_result[f"MRR@{k}"] = mrr

        rows.append(user_result)

    summary = {}
    for metric_name, values in metrics.items():
        summary[metric_name] = float(sum(values) / len(values)) if values else 0.0

    summary["n_users_evaluated"] = int(len(eval_df))
    per_user_df = pd.DataFrame(rows)

    return summary, per_user_df


def main():
    for path in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    train, val, test = load_data()

    print("=== INPUT DATA ===")
    print(f"train: {len(train)} ratings | users={train['userId'].nunique()} | items={train['movieId'].nunique()}")
    print(f"val:   {len(val)} ratings | users={val['userId'].nunique()} | items={val['movieId'].nunique()}")
    print(f"test:  {len(test)} ratings | users={test['userId'].nunique()} | items={test['movieId'].nunique()}")

    ranking_df = build_popularity_ranking(train, MIN_VOTES)

    ranking_path = OUTPUT_DIR / "popularity_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    print("\n=== POPULARITY RANKING ===")
    print(ranking_df.head(10).to_string(index=False))
    print(f"\nsaved ranking -> {ranking_path}")

    val_summary, val_per_user = evaluate_split(train, val, ranking_df, TOP_K_LIST)
    test_summary, test_per_user = evaluate_split(train, test, ranking_df, TOP_K_LIST)

    val_per_user_path = OUTPUT_DIR / "val_per_user_metrics.csv"
    test_per_user_path = OUTPUT_DIR / "test_per_user_metrics.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.json"

    val_per_user.to_csv(val_per_user_path, index=False)
    test_per_user.to_csv(test_per_user_path, index=False)

    summary = {
        "config": {
            "dataset": DATASET,
            "top_k_list": TOP_K_LIST,
            "min_votes": MIN_VOTES,
            "scoring": "imdb_weighted_rating"
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

    print(f"\nsaved per-user val metrics -> {val_per_user_path}")
    print(f"saved per-user test metrics -> {test_per_user_path}")
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()