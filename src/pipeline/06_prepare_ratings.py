from __future__ import annotations
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
DATASET = "small"   # "small" oppure "20m"

RAW_BASE = Path(f"data/raw/ml-latest-{DATASET}" if DATASET == "small" else "data/raw/ml-20m")
PROCESSED_BASE = Path(f"data/processed/{DATASET}")

INPUT_FILE = RAW_BASE / "ratings.csv"
OUTPUT_FILE = PROCESSED_BASE / "ratings_prepared.csv"
#questi  numeri solo perche sono sullo small ricordare dopo nel 2M di mert
MIN_USER_RATINGS = 8
MIN_ITEM_RATINGS = 8

# Se True, ripete i filtri finché il dataset si stabilizza
ITERATIVE_FILTERING = True


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
    # Filtra utenti con almeno min_user_ratings
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df["userId"].isin(valid_users)].copy()

    # Filtra film con almeno min_item_ratings
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
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_FILE}")

    print(f"[06] loading ratings from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    validate_columns(df)

    # Cast robusto
    df["userId"] = pd.to_numeric(df["userId"], errors="raise").astype(int)
    df["movieId"] = pd.to_numeric(df["movieId"], errors="raise").astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Rimuovi righe corrotte
    before_drop = len(df)
    df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"[06] removed corrupted rows: {dropped}")

    df["timestamp"] = df["timestamp"].astype(int)

    # Ordina temporalmente
    df = df.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)

    print_basic_stats(df, "RAW RATINGS")

    if ITERATIVE_FILTERING:
        filtered = iterative_filter(df, MIN_USER_RATINGS, MIN_ITEM_RATINGS)
    else:
        filtered = filter_once(df, MIN_USER_RATINGS, MIN_ITEM_RATINGS)

    print_basic_stats(filtered, "FILTERED RATINGS")

    # Controllo finale vincoli
    min_user_final = filtered["userId"].value_counts().min() if len(filtered) > 0 else 0
    min_item_final = filtered["movieId"].value_counts().min() if len(filtered) > 0 else 0

    print("\n=== FINAL CONSTRAINT CHECK ===")
    print(f"min ratings per user: {min_user_final}")
    print(f"min ratings per item: {min_item_final}")

    filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[06] saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()