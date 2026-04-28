"""
Scarica e ridimensiona i poster TMDB a 224x224 (input size CLIP).
Output: data/posters/<movieId>.jpg

Resume automatico: salta i poster già presenti.
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import io
import time

import pandas as pd
import requests
from PIL import Image

POSTERS_DIR = _PROJECT_ROOT / "data" / "posters"
POSTERS_DIR.mkdir(parents=True, exist_ok=True)


def download_poster(movie_id: int, url: str) -> bool:
    out_path = POSTERS_DIR / f"{movie_id}.jpg"
    if out_path.exists():
        return True  # già scaricato

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return False
            img = Image.open(io.BytesIO(resp.content))
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(out_path, "JPEG", quality=85)
            return True
        except Exception as e:
            if attempt == 2:
                print(f"[SKIP] movie {movie_id}: {e}")
                return False
            time.sleep(1)
    return False


def main():
    from src.utils.io import load_settings
    s = load_settings()
    df = pd.read_csv(s.paths.processed / "movies_enriched_tmdb.csv")
    df = df[df["poster_url"].notna()]

    # ── test: rimuovi questa riga per il download completo ──
    # df = df.head(10)

    total = len(df)
    downloaded = 0
    skipped = 0
    failed = 0

    for i, (_, row) in enumerate(df.iterrows()):
        movie_id = int(row["movieId"])
        url = str(row["poster_url"])

        out_path = POSTERS_DIR / f"{movie_id}.jpg"
        already_exists = out_path.exists()

        result = download_poster(movie_id, url)
        if result:
            if already_exists:
                skipped += 1
            else:
                downloaded += 1
        else:
            failed += 1

        time.sleep(0.1)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{total}] downloaded={downloaded} "
                  f"skipped={skipped} failed={failed}")

    print(f"\nDONE: {downloaded} downloaded, "
          f"{skipped} skipped, {failed} failed")
    print(f"Poster dir: {POSTERS_DIR} "
          f"({sum(1 for _ in POSTERS_DIR.glob('*.jpg'))} files)")


if __name__ == "__main__":
    main()
