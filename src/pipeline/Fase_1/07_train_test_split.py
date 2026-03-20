from __future__ import annotations
from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
DATASET = "small"

BASE = Path(f"data/processed/{DATASET}")
INPUT_FILE = BASE / "ratings_prepared.csv"

TRAIN_FILE = BASE / "ratings_train.csv"
VAL_FILE = BASE / "ratings_val.csv"
TEST_FILE = BASE / "ratings_test.csv"


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

    if n_ratings > 0:
        print(f"min timestamp: {df['timestamp'].min()}")
        print(f"max timestamp: {df['timestamp'].max()}")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file non trovato: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"userId", "movieId", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}")

    # Ordina per utente e tempo
    df = df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    # Controllo: ogni utente deve avere almeno 3 interazioni
    user_counts = df["userId"].value_counts()
    bad_users = user_counts[user_counts < 3]

    if len(bad_users) > 0:
        raise ValueError(
            f"Trovati {len(bad_users)} utenti con meno di 3 interazioni. "
            f"Serve filtrarli prima dello split."
        )

    train_parts = []
    val_parts = []
    test_parts = []

    for user_id, user_df in df.groupby("userId", sort=False):
        user_df = user_df.sort_values(["timestamp", "movieId"]).reset_index(drop=True)

        # split leave-last-2-out temporale
        train_u = user_df.iloc[:-2].copy()
        val_u = user_df.iloc[-2:-1].copy()
        test_u = user_df.iloc[-1:].copy()

        train_parts.append(train_u)
        val_parts.append(val_u)
        test_parts.append(test_u)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Riordino finale
    train_df = train_df.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)
    val_df = val_df.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)
    test_df = test_df.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True)

    print_stats(df, "FULL DATASET")
    print_stats(train_df, "TRAIN FINAL")
    print_stats(val_df, "VAL FINAL")
    print_stats(test_df, "TEST FINAL")

    # Check logici
    train_users = set(train_df["userId"].unique())
    val_users = set(val_df["userId"].unique())
    test_users = set(test_df["userId"].unique())

    assert val_users.issubset(train_users), "Ci sono utenti in val non presenti nel train"
    assert test_users.issubset(train_users), "Ci sono utenti in test non presenti nel train"

    # Check temporale per utente
    for user_id in train_users:
        max_train = train_df.loc[train_df["userId"] == user_id, "timestamp"].max()
        min_val = val_df.loc[val_df["userId"] == user_id, "timestamp"].min()
        min_test = test_df.loc[test_df["userId"] == user_id, "timestamp"].min()

        assert max_train <= min_val, f"Temporal leakage train-val per user {user_id}"
        assert min_val <= min_test, f"Temporal leakage val-test per user {user_id}"

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"\n[07] saved -> {TRAIN_FILE}")
    print(f"[07] saved -> {VAL_FILE}")
    print(f"[07] saved -> {TEST_FILE}")


if __name__ == "__main__":
    main()