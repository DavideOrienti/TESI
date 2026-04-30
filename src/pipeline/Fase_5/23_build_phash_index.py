"""
Calcola il perceptual hash (pHash 16x16) di ogni poster
e salva un indice CSV leggero per il riconoscimento su Render.

Output: data/deploy_artifacts/poster_phash_index.csv
        (movieId, title_clean, phash)

Installa imagehash prima di eseguire:
  .venv/Scripts/pip.exe install imagehash
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import imagehash
from PIL import Image

POSTERS_DIR = _PROJECT_ROOT / "data" / "posters"
DEPLOY_DIR = _PROJECT_ROOT / "data" / "deploy_artifacts"


def main():
    clip_index = pd.read_csv(DEPLOY_DIR / "poster_embeddings_index.csv")
    print(f"[23] {len(clip_index)} film nel CLIP index")

    hashes = []
    failed = 0

    for i, (_, row) in enumerate(clip_index.iterrows()):
        movie_id = int(row["movieId"])
        poster_path = POSTERS_DIR / f"{movie_id}.jpg"

        if not poster_path.exists():
            hashes.append(None)
            failed += 1
            continue

        try:
            img = Image.open(str(poster_path)).convert("RGB")
            h = str(imagehash.phash(img, hash_size=16))
            hashes.append(h)
        except Exception:
            hashes.append(None)
            failed += 1

        if (i + 1) % 500 == 0:
            print(f"[23] {i+1}/{len(clip_index)} processed")

    clip_index["phash"] = hashes
    clip_index_clean = clip_index.dropna(subset=["phash"])

    out_path = DEPLOY_DIR / "poster_phash_index.csv"
    clip_index_clean.to_csv(out_path, index=False)

    print(f"\nDONE: {len(clip_index_clean)} hashes salvati")
    print(f"Failed: {failed}")
    print(f"File: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
