from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings


TRAIN_NAME = "ratings_train.csv"

MODEL_FILES = {
    "content": "baseline_content_based_v2/test_per_user_metrics.csv",
    "cf_improved": "baseline_itemknn_cf_improved/test_per_user_metrics.csv",
    "hybrid": "hybrid_weighted_v1/test_per_user_alpha_0.40.csv",
    "hybrid_pop": "hybrid_with_popularity_v1/test_per_user_alpha_0.40_beta_0.60.csv",
}

OUTPUT_NAME = "user_segment_analysis_test.csv"


def build_user_segments(train: pd.DataFrame) -> pd.DataFrame:
    counts = (
        train.groupby("userId")
        .size()
        .reset_index(name="num_train_ratings")
    )

    q1 = counts["num_train_ratings"].quantile(0.33)
    q2 = counts["num_train_ratings"].quantile(0.66)

    def assign_bucket(x: int) -> str:
        if x <= q1:
            return "low"
        if x <= q2:
            return "medium"
        return "high"

    counts["user_segment"] = counts["num_train_ratings"].apply(assign_bucket)
    return counts


def summarize_by_segment(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for segment, g in df.groupby("user_segment"):
        row = {
            "model": model_name,
            "user_segment": segment,
            "n_users": len(g),
        }
        for k in [5, 10, 20]:
            row[f"HR@{k}"] = float(g[f"HR@{k}"].mean())
            row[f"NDCG@{k}"] = float(g[f"NDCG@{k}"].mean())
            row[f"MRR@{k}"] = float(g[f"MRR@{k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    s = load_settings()
    base = s.paths.processed
    output_file = base / OUTPUT_NAME

    train = pd.read_csv(base / TRAIN_NAME)
    user_segments = build_user_segments(train)

    all_results = []

    for model_name, rel_path in MODEL_FILES.items():
        path = base / rel_path
        df = pd.read_csv(path)

        merged = df.merge(user_segments, on="userId", how="left")
        summary = summarize_by_segment(merged, model_name)
        all_results.append(summary)

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(output_file, index=False)

    print("[17a] saved ->", output_file)
    print(result_df)


if __name__ == "__main__":
    main()