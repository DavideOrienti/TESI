"""
Encoda i poster con CLIP ViT-B/32 e salva gli embedding.

Output in data/deploy_artifacts/:
  - poster_embeddings_clip.npy   (float32, N×512)
  - poster_embeddings_index.csv  (movieId, title_clean)

Resume automatico tramite checkpoint ogni 500 immagini.

Installa CLIP prima di eseguire:
  .venv/Scripts/pip.exe install openai-clip
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image

POSTERS_DIR = _PROJECT_ROOT / "data" / "posters"
DEPLOY_DIR = _PROJECT_ROOT / "data" / "deploy_artifacts"
CHECKPOINT_PATH = DEPLOY_DIR / "clip_checkpoint.npy"
CHECKPOINT_IDX = DEPLOY_DIR / "clip_checkpoint_idx.json"
OUTPUT_EMB = DEPLOY_DIR / "poster_embeddings_clip.npy"
OUTPUT_IDX = DEPLOY_DIR / "poster_embeddings_index.csv"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[22] device: {device}")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    from src.utils.io import load_settings
    s = load_settings()
    df = pd.read_csv(s.paths.processed / "movies_enriched_tmdb.csv")
    df = df[df["poster_url"].notna()].copy()

    df["poster_path"] = df["movieId"].apply(
        lambda mid: POSTERS_DIR / f"{int(mid)}.jpg"
    )
    df = df[df["poster_path"].apply(lambda p: p.exists())].reset_index(drop=True)
    print(f"[22] film con poster: {len(df)}")

    # Resume da checkpoint
    start_idx = 0
    embeddings_so_far: list = []

    if CHECKPOINT_PATH.exists() and CHECKPOINT_IDX.exists():
        embeddings_so_far = np.load(str(CHECKPOINT_PATH)).tolist()
        with open(CHECKPOINT_IDX) as f:
            start_idx = json.load(f)["next_idx"]
        print(f"[22] resuming from idx {start_idx} "
              f"({len(embeddings_so_far)} embeddings già pronti)")

    df_todo = df.iloc[start_idx:].reset_index(drop=True)

    BATCH_SIZE = 32
    embeddings = list(embeddings_so_far)

    for batch_start in range(0, len(df_todo), BATCH_SIZE):
        batch = df_todo.iloc[batch_start : batch_start + BATCH_SIZE]

        images = []
        for _, row in batch.iterrows():
            try:
                img = Image.open(str(row["poster_path"])).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"[SKIP] {row['movieId']}: {e}")

        if not images:
            continue

        with torch.no_grad():
            img_tensor = torch.stack(images).to(device)
            emb = model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.extend(emb.cpu().numpy().tolist())

        global_idx = start_idx + batch_start + len(images)
        if global_idx % 500 < BATCH_SIZE:
            np.save(str(CHECKPOINT_PATH), np.array(embeddings, dtype=np.float32))
            with open(CHECKPOINT_IDX, "w") as f:
                json.dump({"next_idx": global_idx}, f)
            print(f"[22] checkpoint @ {global_idx}/{len(df)}")

    final_emb = np.array(embeddings, dtype=np.float32)
    np.save(str(OUTPUT_EMB), final_emb)

    n_encoded = len(final_emb)
    df_encoded = df.iloc[:n_encoded][["movieId", "title_clean"]].copy()
    df_encoded.to_csv(OUTPUT_IDX, index=False)

    print(f"\nDONE: {n_encoded} embeddings salvati")
    print(f"Shape: {final_emb.shape}")
    print(f"Size: {final_emb.nbytes / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
