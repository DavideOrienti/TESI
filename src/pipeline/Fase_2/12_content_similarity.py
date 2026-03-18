from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

EMBED_FILE = BASE / "movie_embeddings.npy"
INDEX_FILE = BASE / "movie_embeddings_index.csv"

OUTPUT_SIM = BASE / "content_similarity_matrix.npy"


def main():

    embeddings = np.load(EMBED_FILE)

    print("embedding shape:", embeddings.shape)

    print("computing cosine similarity...")

    sim_matrix = cosine_similarity(embeddings)

    print("similarity matrix shape:", sim_matrix.shape)

    np.save(OUTPUT_SIM, sim_matrix)

    print("saved ->", OUTPUT_SIM)


if __name__ == "__main__":
    main()