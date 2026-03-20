from __future__ import annotations
from pathlib import Path
import pandas as pd

DATASET = "small"   # oppure "20m"
BASE = Path(f"data/processed/{DATASET}")

def main():
    path = BASE / "movies_enriched_tmdb.csv"
    df = pd.read_csv(path)

    print("\n=== DATA AUDIT ===")
    print(f"rows: {len(df)}")
    print(f"unique movieId: {df['movieId'].nunique()}")

    if "tmdbId" in df.columns:
        print(f"tmdbId non-null: {df['tmdbId'].notna().sum()}")

    for col in ["title_clean", "year", "tags", "overview_it", "overview_en", "director", "actors_top5", "poster_url"]:
        if col in df.columns:
            non_null = df[col].fillna("").astype(str).str.strip().ne("").sum()
            print(f"{col}: {non_null} non-empty")

    # duplicati movieId
    dup = df[df["movieId"].duplicated()]
    print(f"duplicated movieId: {len(dup)}")

    # anno anomalo
    if "year" in df.columns:
        bad_year = df[(df["year"] != 0) & ((df["year"] < 1900) | (df["year"] > 2030))]
        print(f"bad year rows: {len(bad_year)}")

if __name__ == "__main__":
    main()