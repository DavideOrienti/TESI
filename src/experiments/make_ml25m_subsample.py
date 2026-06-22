"""
Genera un subsample deterministico (~1M rating) di ml-25m, drop-in compatibile
col formato MovieLens nativo (stessi 4 file, stesso schema colonne).

Strategia:
- seleziona utenti (deterministicamente, seed=42) MANTENENDO INTERA la loro
  sequenza cronologica di rating, accumulando finché si raggiunge ~1.000.000
  di rating. Nessun rating viene tagliato a metà utente: il leave-last-2-out
  di Fase_1/07_train_test_split.py resta valido sul subsample.
- scarta a priori gli utenti con meno di MIN_USER_RATINGS_RAW rating nel
  ml-25m completo, per non produrre utenti che verrebbero comunque eliminati
  dal filtro min_user_ratings (src/config/settings.yaml) o che violerebbero
  il vincolo ">= 3 interazioni" richiesto dallo split leave-last-2-out.
- movies.csv / links.csv / tags.csv vengono ristretti ai soli movieId
  comparsi nel subsample di ratings, così il dataset risultante è un
  ml-25m "completo" in miniatura (drop-in anche per Fase_0 e per la fase
  content-based successiva).

NON esegue alcun enrichment TMDB: questo script tocca solo i file raw,
Fase_0 verrà eseguita separatamente sul nuovo dataset "ml-25m-1m".
"""
from __future__ import annotations
import random

import numpy as np
import pandas as pd

from src.utils.io import load_settings

SEED = 42
TARGET_RATINGS = 1_000_000
MIN_USER_RATINGS_RAW = 8  # margine di sicurezza, coerente con filters.min_user_ratings


def load_raw_ratings(raw_dir) -> pd.DataFrame:
    return pd.read_csv(
        raw_dir / "ratings.csv",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )


def select_users(ratings: pd.DataFrame, target_ratings: int, min_user_ratings: int, seed: int) -> set[int]:
    """Seleziona deterministicamente un sottoinsieme di utenti, mantenendo per
    ciascuno l'intera sequenza di rating, finché la somma raggiunge ~target_ratings."""
    user_counts = ratings.groupby("userId").size()
    eligible_users = user_counts[user_counts >= min_user_ratings].index.to_numpy()
    eligible_users = np.sort(eligible_users)

    rng = random.Random(seed)
    shuffled = list(eligible_users)
    rng.shuffle(shuffled)

    selected: list[int] = []
    cumulative = 0
    counts_by_user = user_counts.to_dict()

    for user_id in shuffled:
        if cumulative >= target_ratings:
            break
        selected.append(int(user_id))
        cumulative += int(counts_by_user[user_id])

    return set(selected)


def main():
    s_out = load_settings(dataset="ml-25m-1m")
    out_dir = s_out.paths.raw
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_25m_dir = out_dir.parent / "ml-25m"  # sorgente: data/raw/ml-25m (non registrato in settings.yaml)

    print(f"[make_ml25m_subsample] sorgente: {raw_25m_dir}")
    print(f"[make_ml25m_subsample] output:   {out_dir}")

    print("\nCaricamento ratings.csv...")
    ratings = load_raw_ratings(raw_25m_dir)
    print(f"ratings.csv completo: {len(ratings):,} righe, {ratings['userId'].nunique():,} utenti, {ratings['movieId'].nunique():,} item")

    print(f"\nSelezione utenti (seed={SEED}, target~{TARGET_RATINGS:,}, min_user_ratings_raw={MIN_USER_RATINGS_RAW})...")
    selected_users = select_users(ratings, TARGET_RATINGS, MIN_USER_RATINGS_RAW, SEED)
    print(f"utenti selezionati: {len(selected_users):,}")

    ratings_sub = ratings[ratings["userId"].isin(selected_users)].copy()
    ratings_sub = ratings_sub.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    sampled_movie_ids = set(ratings_sub["movieId"].unique().tolist())

    print("\nRistretto movies.csv / links.csv / tags.csv ai movieId campionati...")
    movies = pd.read_csv(raw_25m_dir / "movies.csv")
    links = pd.read_csv(raw_25m_dir / "links.csv")
    tags = pd.read_csv(raw_25m_dir / "tags.csv")

    movies_sub = movies[movies["movieId"].isin(sampled_movie_ids)].copy()
    links_sub = links[links["movieId"].isin(sampled_movie_ids)].copy()
    tags_sub = tags[tags["movieId"].isin(sampled_movie_ids)].copy()

    ratings_sub.to_csv(out_dir / "ratings.csv", index=False)
    movies_sub.to_csv(out_dir / "movies.csv", index=False)
    links_sub.to_csv(out_dir / "links.csv", index=False)
    tags_sub.to_csv(out_dir / "tags.csv", index=False)

    n_ratings = len(ratings_sub)
    n_users = ratings_sub["userId"].nunique()
    n_items = ratings_sub["movieId"].nunique()
    density = n_ratings / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0

    print("\n=== STATS SUBSAMPLE ml-25m-1m ===")
    print(f"n_rating: {n_ratings:,}")
    print(f"n_utenti: {n_users:,}")
    print(f"n_item:   {n_items:,}")
    print(f"densita:  {density:.6f}")
    print(f"movies.csv ristretto: {len(movies_sub):,} / {len(movies):,} righe")
    print(f"links.csv  ristretto: {len(links_sub):,} / {len(links):,} righe")
    print(f"tags.csv   ristretto: {len(tags_sub):,} / {len(tags):,} righe")
    print(f"\nsalvato in -> {out_dir}")


if __name__ == "__main__":
    main()
