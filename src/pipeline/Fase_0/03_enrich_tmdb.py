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


def non_empty_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().ne("").sum())


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

    in_path = s.paths.processed / "movies_with_links.csv"
    out_path = s.paths.processed / "movies_enriched_tmdb.csv"

    df = pd.read_csv(in_path)

    print("Columns:", df.columns.tolist())
    print(df.head(2))

    # Resume: se output esiste, riprendi e completa i missing
    tmdb_cols = ["actors_top5", "director", "popularity", "overview_it", "overview_en", "poster_url"]

    if out_path.exists():
        prev = pd.read_csv(out_path)

        # prendi solo le colonne TMDB che esistono davvero nel prev
        cols_to_use = ["movieId"] + [c for c in tmdb_cols if c in prev.columns]

        # merge senza suffixes perché le colonne TMDB non esistono (o le riempiamo dopo)
        df = df.merge(prev[cols_to_use], on="movieId", how="left")

    # colonne target
    for col in ["actors_top5", "director", "popularity", "overview_it", "overview_en", "poster_url"]:
        if col not in df.columns:
            df[col] = None

    # tmdb_fetched = True dopo ogni chiamata completata (anche se i campi sono vuoti).
    # Usare questo flag invece di actors_top5.isna() evita di rielaborare film che
    # TMDB ha restituito legittimamente senza attori.
    if "tmdb_fetched" not in df.columns:
        df["tmdb_fetched"] = False

    todo = df["tmdbId"].notna() & (~df["tmdb_fetched"])
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
            df.at[idx, "tmdb_fetched"] = True
            continue

        credits = client.movie_credits(tmdb_id)
        details_it = client.movie_details(tmdb_id, "it-IT")
        details_en = client.movie_details(tmdb_id, "en-US")

        actors = extract_top5_cast(credits)
        director = extract_director(credits)
        popularity = (details_it or {}).get("popularity", None)
        overview_it = (details_it or {}).get("overview", "") or ""
        overview_en = (details_en or {}).get("overview", "") or ""
        poster_url = client.poster_url_from_details(details_it) or client.poster_url_from_details(details_en)

        df.at[idx, "actors_top5"] = ", ".join([a for a in actors if a])
        df.at[idx, "director"] = director
        df.at[idx, "popularity"] = popularity
        df.at[idx, "overview_it"] = overview_it
        df.at[idx, "overview_en"] = overview_en
        df.at[idx, "poster_url"] = poster_url
        df.at[idx, "tmdb_fetched"] = True

        if (int(idx) + 1) % 300 == 0:
            df.to_csv(out_path, index=False)
            print(f"[03] checkpoint saved -> {out_path}")

    df.to_csv(out_path, index=False)
    print(f"[03] saved -> {out_path}")

    # Audit finale di copertura
    total = len(df)

    overview_en_count = non_empty_count(df["overview_en"])
    overview_it_count = non_empty_count(df["overview_it"])
    poster_count = non_empty_count(df["poster_url"])
    director_count = non_empty_count(df["director"])
    actors_count = non_empty_count(df["actors_top5"])
    popularity_count = int(df["popularity"].notna().sum())

    print("[03] COVERAGE AUDIT")
    print(f"overview_en: {overview_en_count}/{total} ({overview_en_count / total:.2%})")
    print(f"overview_it: {overview_it_count}/{total} ({overview_it_count / total:.2%})")
    print(f"poster_url : {poster_count}/{total} ({poster_count / total:.2%})")
    print(f"director   : {director_count}/{total} ({director_count / total:.2%})")
    print(f"actors_top5: {actors_count}/{total} ({actors_count / total:.2%})")
    print(f"popularity : {popularity_count}/{total} ({popularity_count / total:.2%})")


if __name__ == "__main__":
    main()