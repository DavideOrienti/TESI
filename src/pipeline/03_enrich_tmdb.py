from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings
from src.utils.tmdb_client import build_client

def safe_tmdb_int(x) -> int | None:
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def extract_top5_cast(credits: dict | None) -> list[str]:
    if not credits:
        return [""] * 5
    cast = credits.get("cast", []) or []
    names = [c.get("name", "") for c in cast[:5] if c.get("name")]
    names += [""] * (5 - len(names))
    return names[:5]

def extract_director(credits: dict | None) -> str:
    if not credits:
        return ""
    crew = credits.get("crew", []) or []
    for m in crew:
        if m.get("job") == "Director":
            return m.get("name", "") or ""
    return ""

def main():
    s = load_settings()
    client = build_client(
        cache_dir=s.paths.cache_tmdb,
        api_key_env=s.tmdb.api_key_env,
        timeout_sec=s.tmdb.timeout_sec,
        max_retries=s.tmdb.max_retries,
        backoff_sec=s.tmdb.backoff_sec,
        sleep_sec=s.tmdb.sleep_sec,
    )

    in_path = s.paths.processed / "movies_with_tags.csv"
    out_path = s.paths.processed / "movies_enriched_tmdb.csv"

    df = pd.read_csv(in_path)

    # Resume: se output esiste, riprendi e completa i missing
    if out_path.exists():
        prev = pd.read_csv(out_path)
        # preferisci le colonne già calcolate
        df = df.merge(prev[["movieId"] + [c for c in prev.columns if c != "movieId"]], on="movieId", how="left", suffixes=("", "_prev"))
        for col in ["actors_top5", "director", "popularity", "overview_it", "overview_en", "poster_url"]:
            if f"{col}_prev" in df.columns:
                df[col] = df[col].fillna(df[f"{col}_prev"])
                df.drop(columns=[f"{col}_prev"], inplace=True)

    # colonne target
    for col in ["actors_top5", "director", "popularity", "overview_it", "overview_en", "poster_url"]:
        if col not in df.columns:
            df[col] = None

    todo = df["tmdbId"].notna() & df["actors_top5"].isna()
    total_todo = int(todo.sum())
    print(f"[03] to_enrich={total_todo} / total={len(df)}")

    for idx in df.index[todo]:
        tmdb_id = safe_tmdb_int(df.at[idx, "tmdbId"])
        if tmdb_id is None:
            df.at[idx, "actors_top5"] = ""
            df.at[idx, "director"] = ""
            df.at[idx, "popularity"] = None
            df.at[idx, "overview_it"] = ""
            df.at[idx, "overview_en"] = ""
            df.at[idx, "poster_url"] = ""
            continue

        credits = client.movie_credits(tmdb_id)
        details_it = client.movie_details(tmdb_id, "it-IT")
        details_en = client.movie_details(tmdb_id, "en-US")

        actors = extract_top5_cast(credits)
        director = extract_director(credits)
        popularity = (details_it or {}).get("popularity", None)
        overview_it = (details_it or {}).get("overview", "") or ""
        overview_en = (details_en or {}).get("overview", "") or ""
        poster_url = client.poster_url_from_details(details_it)

        df.at[idx, "actors_top5"] = ", ".join([a for a in actors if a])
        df.at[idx, "director"] = director
        df.at[idx, "popularity"] = popularity
        df.at[idx, "overview_it"] = overview_it
        df.at[idx, "overview_en"] = overview_en
        df.at[idx, "poster_url"] = poster_url

        if (int(idx) + 1) % 300 == 0:
            df.to_csv(out_path, index=False)
            print(f"[03] checkpoint saved -> {out_path}")

    df.to_csv(out_path, index=False)
    print(f"[03] saved -> {out_path}")

if __name__ == "__main__":
    main()