from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings

def normalize_tag(t: str) -> str:
    t = str(t).strip().lower()
    t = " ".join(t.split())
    return t

def main():
    s = load_settings()

    movies_in = s.paths.processed / "movies_clean.csv"
    tags_path = s.paths.raw / "tags.csv"
    out_path = s.paths.processed / "movies_with_tags.csv"

    movies = pd.read_csv(movies_in)
    tags = pd.read_csv(tags_path)

    # Normalizza tag
    tags["tag_norm"] = tags["tag"].map(normalize_tag)

    # Dedup per movieId-tag
    tags = tags.drop_duplicates(subset=["movieId", "tag_norm"])

    grouped = (
        tags.groupby("movieId")["tag_norm"]
        .apply(lambda x: ", ".join(x.tolist()))
        .reset_index()
        .rename(columns={"tag_norm": "tags"})
    )

    merged = movies.merge(grouped, on="movieId", how="left")
    merged["tags"] = merged["tags"].fillna("")

    print(f"[02] movies={len(movies)} tags_rows={len(tags)} merged={len(merged)}")
    merged.to_csv(out_path, index=False)
    print(f"[02] saved -> {out_path}")

if __name__ == "__main__":
    main()