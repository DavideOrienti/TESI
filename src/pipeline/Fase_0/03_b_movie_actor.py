from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings


def parse_actors(actors_str: str) -> list[str]:
    if pd.isna(actors_str):
        return []
    actors = [a.strip() for a in str(actors_str).split(",") if a.strip()]
    return actors


def main():
    s = load_settings()

    in_path = s.paths.processed / "movies_enriched_tmdb.csv"
    out_path = s.paths.processed / "movie_actors_top5.csv"

    df = pd.read_csv(in_path)

    rows = []

    for _, row in df.iterrows():
        movie_id = row["movieId"]
        actors = parse_actors(row.get("actors_top5", ""))

        for rank, actor_name in enumerate(actors, start=1):
            rows.append({
                "movieId": movie_id,
                "actor_name": actor_name,
                "cast_rank": rank
            })

    movie_actors = pd.DataFrame(rows)

    print(f"[04a] input_movies={len(df)}")
    print(f"[04a] actor_relations={len(movie_actors)}")
    print(f"[04a] unique_actors={movie_actors['actor_name'].nunique() if not movie_actors.empty else 0}")

    movie_actors.to_csv(out_path, index=False)
    print(f"[04a] saved -> {out_path}")


if __name__ == "__main__":
    main()