from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.io import load_settings

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variant",
    type=str,
    choices=["full", "no_tags", "no_overview", "genres_only"],
    required=True,
    help="Variante della text representation per l'ablation study"
)
args = parser.parse_args()
VARIANT = args.variant


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_name_list(value: object, sep: str = ",") -> str:
    if pd.isna(value):
        return ""
    items = []
    for x in str(value).split(sep):
        x = x.strip().lower()
        if x:
            items.append(x.replace(" ", "_"))
    return " ".join(items)


def normalize_genres(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("|", " ")


def build_text_representation(row: pd.Series, variant: str) -> str:
    genres = normalize_genres(row.get("genres", ""))
    actors = normalize_name_list(row.get("actors_top5", ""))
    director = normalize_name_list(row.get("director", ""))
    overview = normalize_text(row.get("overview_en", ""))
    tags = normalize_text(row.get("tags", ""))

    if variant == "genres_only":
        return f"genres {genres}. genres {genres}. genres {genres}."

    if variant == "no_tags":
        return (
            f"genres {genres}. genres {genres}. genres {genres}. "
            f"director {director}. director {director}. "
            f"actors {actors}. actors {actors}. "
            f"plot {overview}"
        ).strip()

    if variant == "no_overview":
        return (
            f"genres {genres}. genres {genres}. genres {genres}. "
            f"director {director}. director {director}. "
            f"actors {actors}. actors {actors}. "
            f"tags {tags}. tags {tags}."
        ).strip()

    return (
        f"genres {genres}. genres {genres}. genres {genres}. "
        f"director {director}. director {director}. "
        f"actors {actors}. actors {actors}. "
        f"tags {tags}. tags {tags}. "
        f"plot {overview}"
    ).strip()


def main():
    s = load_settings()
    base = s.paths.processed

    movies_file = base / "movies_enriched_tmdb.csv"
    out_embed = base / f"movie_embeddings_{VARIANT}.npy"
    out_index = base / f"movie_embeddings_index_{VARIANT}.csv"

    df = pd.read_csv(movies_file)
    df["text_repr"] = df.apply(lambda r: build_text_representation(r, VARIANT), axis=1)

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(
        df["text_repr"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(out_embed, embeddings)
    df[["movieId", "title_clean"]].to_csv(out_index, index=False)

    print(f"[18a] variant={VARIANT}")
    print("saved embeddings ->", out_embed)
    print("saved index ->", out_index)


if __name__ == "__main__":
    main()