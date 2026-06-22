from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings


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
    s = load_settings()

    base = s.paths.processed
    input_file = base / "ratings_prepared.csv"
    train_file = base / "ratings_train.csv"
    val_file = base / "ratings_val.csv"
    test_file = base / "ratings_test.csv"
    audit_file = base / "ratings_split_audit.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file non trovato: {input_file}")

    print(f"[07] dataset={s.dataset}")
    print(f"[07] loading input from: {input_file}")

    df = pd.read_csv(input_file)

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

    # Split leave-last-2-out vettorizzato: df è già ordinato per [userId, timestamp,
    # movieId], quindi cumcount(ascending=False) dà 0 all'ultimo rating dell'utente
    # (=> test), 1 al penultimo (=> val), >=2 al resto (=> train) — stesso esito di
    # iloc[-1:] / iloc[-2:-1] / iloc[:-2] per ciascun gruppo utente.
    rank_from_end = df.groupby("userId").cumcount(ascending=False)

    train_df = df[rank_from_end >= 2].copy()
    val_df = df[rank_from_end == 1].copy()
    test_df = df[rank_from_end == 0].copy()

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

    # Check temporale per utente, vettorizzato via groupby + merge (no loop Python).
    train_max = train_df.groupby("userId")["timestamp"].max().rename("train_max")
    val_min = val_df.groupby("userId")["timestamp"].min().rename("val_min")
    test_min = test_df.groupby("userId")["timestamp"].min().rename("test_min")

    audit_ts = pd.concat([train_max, val_min, test_min], axis=1)

    bad_train_val = audit_ts.index[audit_ts["train_max"] > audit_ts["val_min"]]
    if len(bad_train_val) > 0:
        raise AssertionError(
            f"Temporal leakage train-val per {len(bad_train_val)} utenti, es: {list(bad_train_val[:5])}"
        )

    bad_val_test = audit_ts.index[audit_ts["val_min"] > audit_ts["test_min"]]
    if len(bad_val_test) > 0:
        raise AssertionError(
            f"Temporal leakage val-test per {len(bad_val_test)} utenti, es: {list(bad_val_test[:5])}"
        )

    # Audit unseen items
    train_items = set(train_df["movieId"].unique())
    val_items = set(val_df["movieId"].unique())
    test_items = set(test_df["movieId"].unique())

    val_unseen_items = val_items - train_items
    test_unseen_items = test_items - train_items

    print("\n=== SPLIT AUDIT ===")
    print(f"train unique items: {len(train_items)}")
    print(f"val unique items: {len(val_items)}")
    print(f"test unique items: {len(test_items)}")
    print(f"val unseen items vs train: {len(val_unseen_items)}")
    print(f"test unseen items vs train: {len(test_unseen_items)}")

    # Audit tabellare
    audit_rows = [
        {
            "split": "full",
            "ratings": len(df),
            "users": df["userId"].nunique(),
            "items": df["movieId"].nunique(),
            "density": len(df) / (df["userId"].nunique() * df["movieId"].nunique())
            if df["userId"].nunique() > 0 and df["movieId"].nunique() > 0 else 0.0,
            "min_timestamp": df["timestamp"].min() if len(df) > 0 else None,
            "max_timestamp": df["timestamp"].max() if len(df) > 0 else None,
            "unseen_vs_train": 0,
        },
        {
            "split": "train",
            "ratings": len(train_df),
            "users": train_df["userId"].nunique(),
            "items": train_df["movieId"].nunique(),
            "density": len(train_df) / (train_df["userId"].nunique() * train_df["movieId"].nunique())
            if train_df["userId"].nunique() > 0 and train_df["movieId"].nunique() > 0 else 0.0,
            "min_timestamp": train_df["timestamp"].min() if len(train_df) > 0 else None,
            "max_timestamp": train_df["timestamp"].max() if len(train_df) > 0 else None,
            "unseen_vs_train": 0,
        },
        {
            "split": "val",
            "ratings": len(val_df),
            "users": val_df["userId"].nunique(),
            "items": val_df["movieId"].nunique(),
            "density": len(val_df) / (val_df["userId"].nunique() * val_df["movieId"].nunique())
            if val_df["userId"].nunique() > 0 and val_df["movieId"].nunique() > 0 else 0.0,
            "min_timestamp": val_df["timestamp"].min() if len(val_df) > 0 else None,
            "max_timestamp": val_df["timestamp"].max() if len(val_df) > 0 else None,
            "unseen_vs_train": len(val_unseen_items),
        },
        {
            "split": "test",
            "ratings": len(test_df),
            "users": test_df["userId"].nunique(),
            "items": test_df["movieId"].nunique(),
            "density": len(test_df) / (test_df["userId"].nunique() * test_df["movieId"].nunique())
            if test_df["userId"].nunique() > 0 and test_df["movieId"].nunique() > 0 else 0.0,
            "min_timestamp": test_df["timestamp"].min() if len(test_df) > 0 else None,
            "max_timestamp": test_df["timestamp"].max() if len(test_df) > 0 else None,
            "unseen_vs_train": len(test_unseen_items),
        },
    ]

    audit_df = pd.DataFrame(audit_rows)

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    audit_df.to_csv(audit_file, index=False)

    print(f"\n[07] saved -> {train_file}")
    print(f"[07] saved -> {val_file}")
    print(f"[07] saved -> {test_file}")
    print(f"[07] saved audit -> {audit_file}")


if __name__ == "__main__":
    main()