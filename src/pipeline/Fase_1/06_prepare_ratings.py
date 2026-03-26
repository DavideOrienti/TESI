from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings


def print_basic_stats(df: pd.DataFrame, label: str) -> None:
    n_ratings = len(df)
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()

    density = 0.0
    if n_users > 0 and n_items > 0:
        density = n_ratings / (n_users * n_items)

    print(f"\n=== {label} ===")
    print(f"ratings: {n_ratings}")
    print(f"users: {n_users}")
    print(f"items: {n_items}")
    print(f"density: {density:.6f}")
    print(f"avg ratings per user: {n_ratings / n_users:.2f}" if n_users else "avg ratings per user: 0")
    print(f"avg ratings per item: {n_ratings / n_items:.2f}" if n_items else "avg ratings per item: 0")


def validate_columns(df: pd.DataFrame) -> None:
    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel ratings.csv: {missing}")


def filter_once(df: pd.DataFrame, min_user_ratings: int, min_item_ratings: int) -> pd.DataFrame:
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["userId"].isin(valid_users)].copy()

    item_counts = df["movieId"].value_counts()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    df = df[df["movieId"].isin(valid_items)].copy()

    return df


def iterative_filter(df: pd.DataFrame, min_user_ratings: int, min_item_ratings: int) -> pd.DataFrame:
    prev_shape = None
    current = df.copy()

    iteration = 0
    while prev_shape != current.shape:
        iteration += 1
        prev_shape = current.shape
        current = filter_once(current, min_user_ratings, min_item_ratings)
        print(f"[Iter {iteration}] shape={current.shape}")

    return current


def main():
    s = load_settings()

    input_file = s.paths.raw / "ratings.csv"
    output_file = s.paths.processed / "ratings_prepared.csv"

    min_user_ratings = s.filters.min_user_ratings
    min_item_ratings = s.filters.min_item_ratings
    iterative_filtering = s.filters.iterative_filtering

    s.paths.processed.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"File non trovato: {input_file}")

    print(f"[06] dataset={s.dataset}")
    print(f"[06] loading ratings from: {input_file}")
    print(
        f"[06] config -> min_user_ratings={min_user_ratings}, "
        f"min_item_ratings={min_item_ratings}, iterative_filtering={iterative_filtering}"
    )

    df = pd.read_csv(input_file)

    validate_columns(df)

    df["userId"] = pd.to_numeric(df["userId"], errors="raise").astype(int)
    df["movieId"] = pd.to_numeric(df["movieId"], errors="raise").astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"[06] removed corrupted rows: {dropped}")

    # check opzionale sul range rating
    bad_ratings = df[(df["rating"] < 0.5) | (df["rating"] > 5.0)]
    print(f"[06] bad rating rows: {len(bad_ratings)}")
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)].copy()

    df["timestamp"] = df["timestamp"].astype(int)

    df = df.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)

    print_basic_stats(df, "RAW RATINGS")

    if iterative_filtering:
        filtered = iterative_filter(df, min_user_ratings, min_item_ratings)
    else:
        filtered = filter_once(df, min_user_ratings, min_item_ratings)

    print_basic_stats(filtered, "FILTERED RATINGS")

    min_user_final = filtered["userId"].value_counts().min() if len(filtered) > 0 else 0
    min_item_final = filtered["movieId"].value_counts().min() if len(filtered) > 0 else 0

    print("\n=== FINAL CONSTRAINT CHECK ===")
    print(f"min ratings per user: {min_user_final}")
    print(f"min ratings per item: {min_item_final}")

    filtered.to_csv(output_file, index=False)
    print(f"\n[06] saved -> {output_file}")


if __name__ == "__main__":
    main()