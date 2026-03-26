from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.io import load_settings


VARIANT = "full"
TOP_N = 50


def main():
    s = load_settings()
    base = s.paths.processed

    embed_file = base / f"movie_embeddings_{VARIANT}.npy"
    out_file = base / f"content_top_neighbors_{VARIANT}.csv"

    embeddings = np.load(embed_file).astype(np.float32)
    sim_matrix = cosine_similarity(embeddings).astype(np.float32)

    rows = []
    n = sim_matrix.shape[0]

    for i in range(n):
        sims = sim_matrix[i]
        sims[i] = -1.0

        top_idx = np.argpartition(-sims, TOP_N)[:TOP_N]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        for j in top_idx:
            rows.append({
                "movie_idx": i,
                "neighbor_idx": int(j),
                "similarity": float(sims[j])
            })

    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"[18b] variant={VARIANT}")
    print("saved ->", out_file)


if __name__ == "__main__":
    main()