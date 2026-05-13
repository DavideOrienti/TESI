from __future__ import annotations
import json
import pandas as pd

from src.utils.io import load_settings
from src.utils.eval import hit_rate_at_k, ndcg_at_k_single, mrr_at_k_single, precision_at_k, recall_at_k
from src.recommenders.popularity import build_popularity_ranking, recommend_popular


# =========================
# CONFIG (parziale)
# =========================
TOP_K_LIST = [5, 10, 20]
MIN_VOTES = 10   # m nella formula IMDb


def load_data(train_file, val_file, test_file) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    return train, val, test


def get_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("userId")["movieId"].apply(set).to_dict()


def recommend_top_k_for_user(
    ranking_movie_ids: list[int],
    seen_items: set[int],
    k: int
) -> list[int]:
    return recommend_popular(ranking_movie_ids, seen_items, k)


def evaluate_split(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    top_k_list: list[int]
) -> tuple[dict, pd.DataFrame]:

    seen_map = get_seen_items(train)
    ranking_movie_ids = ranking_df["movieId"].tolist()

    rows = []
    metrics = {f"HR@{k}": [] for k in top_k_list}
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"MRR@{k}": [] for k in top_k_list})
    metrics.update({f"P@{k}": [] for k in top_k_list})
    metrics.update({f"R@{k}": [] for k in top_k_list})

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

            user_result[f"HR@{k}"]   = hr
            user_result[f"NDCG@{k}"] = ndcg
            user_result[f"MRR@{k}"]  = mrr
            user_result[f"P@{k}"]    = p
            user_result[f"R@{k}"]    = r

        rows.append(user_result)

    summary = {
        metric: float(sum(values) / len(values)) if values else 0.0
        for metric, values in metrics.items()
    }

    summary["n_users_evaluated"] = int(len(eval_df))
    per_user_df = pd.DataFrame(rows)

    return summary, per_user_df


def main():
    s = load_settings()

    base = s.paths.processed

    train_file = base / "ratings_train.csv"
    val_file = base / "ratings_val.csv"
    test_file = base / "ratings_test.csv"

    output_dir = base / "baseline_popularity"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [train_file, val_file, test_file]:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    print(f"[08] dataset={s.dataset}")
    print(f"[08] base path={base}")

    train, val, test = load_data(train_file, val_file, test_file)

    print("\n=== INPUT DATA ===")
    print(f"train: {len(train)} ratings | users={train['userId'].nunique()} | items={train['movieId'].nunique()}")
    print(f"val:   {len(val)} ratings | users={val['userId'].nunique()} | items={val['movieId'].nunique()}")
    print(f"test:  {len(test)} ratings | users={test['userId'].nunique()} | items={test['movieId'].nunique()}")

    ranking_df = build_popularity_ranking(train, MIN_VOTES)

    ranking_path = output_dir / "popularity_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    print("\n=== POPULARITY RANKING ===")
    print(ranking_df.head(10).to_string(index=False))
    print(f"\nsaved ranking -> {ranking_path}")

    val_summary, val_per_user = evaluate_split(train, val, ranking_df, TOP_K_LIST)
    test_summary, test_per_user = evaluate_split(train, test, ranking_df, TOP_K_LIST)

    val_per_user_path = output_dir / "val_per_user_metrics.csv"
    test_per_user_path = output_dir / "test_per_user_metrics.csv"
    summary_path = output_dir / "summary_metrics.json"

    val_per_user.to_csv(val_per_user_path, index=False)
    test_per_user.to_csv(test_per_user_path, index=False)

    summary = {
        "config": {
            "dataset": s.dataset,
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
