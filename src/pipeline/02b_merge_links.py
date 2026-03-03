from __future__ import annotations
import pandas as pd
from src.utils.io import load_settings

def main():
    s = load_settings()

    movies_in = s.paths.processed / "movies_with_tags.csv"   # output step 02
    links_path = s.paths.raw / "links.csv"                   # MovieLens raw
    out_path = s.paths.processed / "movies_with_links.csv"   # nuovo output

    df_movies = pd.read_csv(movies_in)
    df_links = pd.read_csv(links_path)

    df_links = df_links[["movieId", "tmdbId"]].copy()
    df_links["tmdbId"] = pd.to_numeric(df_links["tmdbId"], errors="coerce")

    df = df_movies.merge(df_links, on="movieId", how="left")

    print(f"[02b] movies={len(df_movies)} links={len(df_links)} merged={len(df)} tmdb_nonnull={df['tmdbId'].notna().sum()}")
    df.to_csv(out_path, index=False)
    print(f"[02b] saved -> {out_path}")

if __name__ == "__main__":
    main()