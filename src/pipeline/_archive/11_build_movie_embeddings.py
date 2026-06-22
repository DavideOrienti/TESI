from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

INPUT_FILE = BASE / "movies_enriched_tmdb.csv"
OUTPUT_EMBEDDINGS = BASE / "movie_embeddings.npy"
OUTPUT_INDEX = BASE / "movie_embeddings_index.csv"


def build_text_representation(row):

    title = str(row.get("title_clean", ""))
    overview = str(row.get("overview_en", ""))
    genres = str(row.get("genres", ""))
    actors = str(row.get("actors_top5", ""))
    director = str(row.get("director", ""))

    text = (
        f"Title: {title}. "
        f"Genres: {genres}. "
        f"Director: {director}. "
        f"Actors: {actors}. "
        f"Plot: {overview}"
    )

    return text


def main():

    print("loading movies dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"movies loaded: {len(df)}")

    print("building text representation...")

    df["text_repr"] = df.apply(build_text_representation, axis=1)

    model = SentenceTransformer("all-mpnet-base-v2")

    print("generating embeddings...")

    embeddings = model.encode(
        df["text_repr"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("embedding shape:", embeddings.shape)

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    df_index = df[["movieId", "title_clean"]]
    df_index.to_csv(OUTPUT_INDEX, index=False)

    print("saved embeddings ->", OUTPUT_EMBEDDINGS)
    print("saved index ->", OUTPUT_INDEX)


if __name__ == "__main__":
    main()