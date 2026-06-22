"""
Genera embedding di ricerca semantica con paraphrase-MiniLM-L6-v2 (384-dim).
Usato SOLO per la ricerca semantica — le raccomandazioni continuano
ad usare all-mpnet-base-v2 (768-dim).

Output in data/deploy_artifacts/:
  - search_embeddings_minilm.npy  (float32, N×384)
  - search_embeddings_index.csv   (movieId, title_clean)
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.io import load_settings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MOVIES_CSV = load_settings().paths.processed / "movies_enriched_tmdb.csv"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "deploy_artifacts"
OUTPUT_NPY = OUTPUT_DIR / "search_embeddings_minilm.npy"
OUTPUT_IDX = OUTPUT_DIR / "search_embeddings_index.csv"


def build_search_text(row) -> str:
    parts = []
    if pd.notna(row.get("title_clean")):
        parts.append(str(row["title_clean"]))
    if pd.notna(row.get("year")):
        parts.append(str(int(row["year"])))
    if pd.notna(row.get("genres")):
        parts.append(str(row["genres"]).replace("|", " "))
    if pd.notna(row.get("director")):
        parts.append(str(row["director"]))
    if pd.notna(row.get("actors_top5")):
        actors = str(row["actors_top5"]).replace("_", " ")
        parts.append(actors)
    if pd.notna(row.get("overview_en")) and len(str(row["overview_en"])) > 10:
        parts.append(str(row["overview_en"]))
    return " | ".join(parts)


def main():
    print(f"[20] Loading movies from {MOVIES_CSV}")
    df = pd.read_csv(MOVIES_CSV)
    print(f"[20] {len(df)} movies loaded")

    texts = []
    for _, row in df.iterrows():
        texts.append(build_search_text(row))

    n_no_overview = df["overview_en"].isna().sum() + (df["overview_en"].str.len() <= 10).sum()
    print(f"[20] Films senza overview (solo titolo+generi): {n_no_overview}")

    print("[20] Loading paraphrase-MiniLM-L6-v2...")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    print("[20] Encoding...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype("float32")
    print(f"[20] Embedding shape: {embeddings.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_NPY), embeddings)

    index_df = pd.DataFrame({
        "movieId": df["movieId"].astype(int).values,
        "title_clean": df["title_clean"].fillna(df["title"]).values,
    })
    index_df.to_csv(str(OUTPUT_IDX), index=False)

    npy_mb = OUTPUT_NPY.stat().st_size / 1_048_576
    print(f"\n[20] Saved:")
    print(f"  {OUTPUT_NPY.name}: {npy_mb:.1f} MB")
    print(f"  {OUTPUT_IDX.name}: {OUTPUT_IDX.stat().st_size / 1_048_576:.2f} MB")
    print(f"\n[20] Done — {len(df)} films con embedding ({embeddings.shape[1]}-dim)")


if __name__ == "__main__":
    main()
