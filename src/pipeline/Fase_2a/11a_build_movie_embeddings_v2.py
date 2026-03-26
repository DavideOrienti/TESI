from __future__ import annotations
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.io import load_settings

# =========================
# OUTPUT FILES
# =========================
OUTPUT_EMBEDDINGS_NAME = "movie_embeddings_v2.npy"
OUTPUT_INDEX_NAME = "movie_embeddings_index_v2.csv"


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


def normalize_tags(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


# =========================
# TEXT REPRESENTATION
# =========================
def build_text_representation(row: pd.Series) -> str:
    genres = normalize_genres(row.get("genres", ""))
    actors = normalize_name_list(row.get("actors_top5", ""))
    director = normalize_name_list(row.get("director", ""))
    overview = normalize_text(row.get("overview_en", ""))
    tags = normalize_tags(row.get("tags", ""))

    # pesi impliciti tramite ripetizione
    text = (
        f"genres {genres}. "
        f"genres {genres}. "
        f"genres {genres}. "
        f"director {director}. "
        f"director {director}. "
        f"actors {actors}. "
        f"actors {actors}. "
        f"tags {tags}. "
        f"tags {tags}. "
        f"plot {overview}"
    )

    return text.strip()


# =========================
# MAIN
# =========================
def main():
    s = load_settings()

    input_file = s.paths.processed / "movies_enriched_tmdb.csv"
    output_embeddings = s.paths.processed / OUTPUT_EMBEDDINGS_NAME
    output_index = s.paths.processed / OUTPUT_INDEX_NAME

    print(f"[11] dataset={s.dataset}")
    print(f"[11] loading movies from: {input_file}")

    df = pd.read_csv(input_file)
    print(f"Movies loaded: {len(df)}")

    print("Building text representation...")
    df["text_repr_v2"] = df.apply(build_text_representation, axis=1)

    model = SentenceTransformer("all-mpnet-base-v2")

    print("Generating embeddings...")
    embeddings = model.encode(
        df["text_repr_v2"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embedding shape:", embeddings.shape)

    np.save(output_embeddings, embeddings)

    df_index = df[["movieId", "title_clean"]].copy()
    df_index.to_csv(output_index, index=False)

    print("Saved embeddings ->", output_embeddings)
    print("Saved index ->", output_index)


if __name__ == "__main__":
    main()