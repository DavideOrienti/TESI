from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings


def non_empty_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


def main():
    s = load_settings()
    path = s.paths.processed / "movies_enriched_tmdb.csv"

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    df = pd.read_csv(path)
    total = len(df)

    print("\n=== DATA AUDIT ===")
    print(f"dataset: {s.dataset}")
    print(f"input file: {path}")
    print(f"rows: {total}")
    print(f"unique movieId: {df['movieId'].nunique()}")

    if "tmdbId" in df.columns:
        tmdb_nonnull = int(df["tmdbId"].notna().sum())
        print(f"tmdbId non-null: {tmdb_nonnull} ({tmdb_nonnull / total:.2%})")

    text_cols = [
        "title_clean",
        "tags",
        "overview_it",
        "overview_en",
        "director",
        "actors_top5",
        "poster_url",
    ]

    for col in text_cols:
        if col in df.columns:
            non_null = non_empty_count(df[col])
            print(f"{col}: {non_null} non-empty ({non_null / total:.2%})")

    if "year" in df.columns:
        valid_year = int((df["year"] != 0).sum())
        print(f"year valid: {valid_year} ({valid_year / total:.2%})")

    dup = int(df["movieId"].duplicated().sum())
    print(f"duplicated movieId: {dup}")

    if "year" in df.columns:
        bad_year = df[
            (df["year"] != 0) &
            ((df["year"] < 1900) | (df["year"] > 2030))
        ]
        print(f"bad year rows: {len(bad_year)}")


if __name__ == "__main__":
    main()