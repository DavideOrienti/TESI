from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

INPUT_FILE = BASE / "movies_enriched_tmdb.csv"
OUTPUT_EMBEDDINGS = BASE / "movie_embeddings_v2.npy"
OUTPUT_INDEX = BASE / "movie_embeddings_index_v2.csv"


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


def build_text_representation(row: pd.Series) -> str:
    genres = normalize_genres(row.get("genres", ""))
    actors = normalize_name_list(row.get("actors_top5", ""))
    director = normalize_name_list(row.get("director", ""))
    overview = normalize_text(row.get("overview_en", ""))

    # pesi impliciti tramite ripetizione
    text = (
        f"genres {genres}. "
        f"genres {genres}. "
        f"genres {genres}. "
        f"director {director}. "
        f"director {director}. "
        f"actors {actors}. "
        f"actors {actors}. "
        f"plot {overview}"
    )

    return text.strip()


def main():
    print("Loading movies dataset...")
    df = pd.read_csv(INPUT_FILE)
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

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    df_index = df[["movieId", "title_clean"]].copy()
    df_index.to_csv(OUTPUT_INDEX, index=False)

    print("Saved embeddings ->", OUTPUT_EMBEDDINGS)
    print("Saved index ->", OUTPUT_INDEX)


if __name__ == "__main__":
    main()