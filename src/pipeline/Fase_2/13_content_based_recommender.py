import numpy as np
import pandas as pd
from pathlib import Path

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

SIM_FILE = BASE / "content_similarity_matrix.npy"
INDEX_FILE = BASE / "movie_embeddings_index.csv"


def recommend(movie_id, top_k=10):

    sim_matrix = np.load(SIM_FILE)

    df_index = pd.read_csv(INDEX_FILE)

    idx = df_index[df_index.movieId == movie_id].index[0]

    sim_scores = sim_matrix[idx]

    top_idx = np.argsort(sim_scores)[::-1][1:top_k+1]

    return df_index.iloc[top_idx]


def main():

    movie_id = 1

    recs = recommend(movie_id)

    print(recs)


if __name__ == "__main__":
    main()