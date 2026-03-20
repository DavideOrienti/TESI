from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

EMBED_FILE = BASE / "movie_embeddings_v2.npy"
OUTPUT_SIM = BASE / "content_similarity_matrix_v2.npy"


def main():
    print("Loading embeddings...")
    embeddings = np.load(EMBED_FILE)
    print("Embedding shape:", embeddings.shape)

    print("Computing cosine similarity...")
    sim_matrix = cosine_similarity(embeddings)

    print("Similarity matrix shape:", sim_matrix.shape)

    np.save(OUTPUT_SIM, sim_matrix)
    print("Saved ->", OUTPUT_SIM)


if __name__ == "__main__":
    main()