from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings
from src.utils.ids import normalize_name, stable_int_id

def main():
    s = load_settings()
    in_path = s.paths.processed / "movies_enriched_tmdb.csv"

    out_actors = s.paths.processed / "people_actors.csv"
    out_directors = s.paths.processed / "people_directors.csv"

    df = pd.read_csv(in_path)

    # Attori: da stringa "a, b, c"
    actors = set()
    for v in df["actors_top5"].fillna("").astype(str):
        for a in v.split(","):
            a = normalize_name(a)
            if a:
                actors.add(a)

    # Registi
    directors = set()
    for d in df["director"].fillna("").astype(str):
        d = normalize_name(d)
        if d:
            directors.add(d)

    actors_df = pd.DataFrame(
        [{"actor_id": stable_int_id("actor", a), "name": a} for a in sorted(actors)]
    ).sort_values("actor_id")

    directors_df = pd.DataFrame(
        [{"director_id": stable_int_id("director", d), "name": d} for d in sorted(directors)]
    ).sort_values("director_id")

    print(f"[04] actors={len(actors_df)} directors={len(directors_df)}")

    actors_df.to_csv(out_actors, index=False)
    directors_df.to_csv(out_directors, index=False)

    print(f"[04] saved -> {out_actors}")
    print(f"[04] saved -> {out_directors}")

if __name__ == "__main__":
    main()