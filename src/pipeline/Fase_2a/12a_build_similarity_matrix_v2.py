from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.io import load_settings

# =========================
# CONFIG
# =========================
TOP_N = 50  # numero di vicini da salvare per ogni film

OUTPUT_FILE_NAME = "content_top_neighbors_v2.csv"


def main():
    s = load_settings()
    base = s.paths.processed

    embed_file = base / "movie_embeddings_v2.npy"
    output_file = base / OUTPUT_FILE_NAME

    print(f"[12] dataset={s.dataset}")
    print(f"[12] loading embeddings from: {embed_file}")

    embeddings = np.load(embed_file).astype(np.float32)
    print("Embedding shape:", embeddings.shape)
    print("dtype:", embeddings.dtype)

    print("Computing cosine similarity...")
    sim_matrix = cosine_similarity(embeddings).astype(np.float32)

    print("Similarity matrix shape:", sim_matrix.shape)

    # =========================
    # EXTRACT TOP-N NEIGHBORS
    # =========================
    print(f"Extracting top-{TOP_N} neighbors per item...")

    rows = []
    num_items = sim_matrix.shape[0]

    for i in range(num_items):
        sims = sim_matrix[i]

        # escludi se stesso
        sims[i] = -1.0

        # top-N indici
        top_idx = np.argpartition(-sims, TOP_N)[:TOP_N]

        # ordina per similarità
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        for j in top_idx:
            rows.append({
                "movie_idx": i,
                "neighbor_idx": int(j),
                "similarity": float(sims[j])
            })

        if (i + 1) % 1000 == 0:
            print(f"[12] processed {i+1}/{num_items}")

    neighbors_df = pd.DataFrame(rows)

    print("Saving top neighbors...")
    neighbors_df.to_csv(output_file, index=False)

    print(f"[12] saved -> {output_file}")
    print(f"[12] total rows: {len(neighbors_df)}")


if __name__ == "__main__":
    main()