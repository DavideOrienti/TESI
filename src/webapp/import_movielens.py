"""
Script per importare i dati MovieLens 20M nel database.

USO:
  python import_movielens.py --movies path/to/movies.csv --ratings path/to/ratings.csv --tags path/to/tags.csv

I file CSV si trovano nella cartella scaricata da MovieLens:
  ml-2/movies.csv
  ml-20m/ratings.csv
  ml-20m/tags.csv      (opzionale)
  ml-20m/links.csv     (per TMDB poster)
"""
import argparse
import csv
import re
import time
import requests
import os
from sqlalchemy.orm import Session
from database import engine, SessionLocal
import models

models.Base.metadata.create_all(bind=engine)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")   # Metti la tua chiave .env
TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"


def extract_year(title: str):
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        return int(match.group(1)), re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    return None, title.strip()


def fetch_tmdb_poster(tmdb_id: int) -> tuple[str, str]:
    """Restituisce (poster_url, overview) da TMDB"""
    if not TMDB_API_KEY or not tmdb_id:
        return None, None
    try:
        resp = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY, "language": "it-IT"},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            poster = f"{POSTER_BASE}{data['poster_path']}" if data.get("poster_path") else None
            overview = data.get("overview", "")
            return poster, overview
    except Exception:
        pass
    return None, None


def import_movies(db: Session, movies_path: str, links_path: str = None, fetch_posters: bool = False):
    print("📽  Importazione film...")

    # Mappa movieId -> tmdbId da links.csv
    tmdb_map = {}
    if links_path:
        with open(links_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tmdb_map[int(row["movieId"])] = int(row["tmdbId"]) if row["tmdbId"] else None
                except (ValueError, KeyError):
                    pass

    batch = []
    with open(movies_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            movie_id = int(row["movieId"])
            year, clean_title = extract_year(row["title"])
            tmdb_id = tmdb_map.get(movie_id)

            poster_url, overview = None, None
            if fetch_posters and tmdb_id:
                poster_url, overview = fetch_tmdb_poster(tmdb_id)
                time.sleep(0.1)  # Rate limiting TMDB

            batch.append(models.Film(
                id=movie_id,
                title=clean_title,
                genres=row.get("genres", ""),
                year=year,
                tmdb_id=tmdb_id,
                poster_url=poster_url,
                overview=overview,
            ))

            if len(batch) >= 1000:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []
                print(f"  → {i+1} film importati...")

    if batch:
        db.bulk_save_objects(batch)
        db.commit()
    print(f"✅ Film importati: {db.query(models.Film).count()}")


def import_ratings(db: Session, ratings_path: str, max_rows: int = 2_000_000):
    """
    Importa le valutazioni e calcola avg_rating + num_ratings per ogni film.
    Con 20M di rating, importiamo un sottoinsieme per velocità.
    """
    print(f"⭐ Importazione valutazioni (max {max_rows:,})...")

    # Accumula per film
    film_scores: dict[int, list[float]] = {}

    with open(ratings_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            film_id = int(row["movieId"])
            score = float(row["rating"])
            if film_id not in film_scores:
                film_scores[film_id] = []
            film_scores[film_id].append(score)

            if i % 500_000 == 0 and i > 0:
                print(f"  → {i:,} rating letti...")

    print(f"  → Aggiornamento avg_rating per {len(film_scores)} film...")
    for film_id, scores in film_scores.items():
        avg = round(sum(scores) / len(scores), 2)
        db.query(models.Film).filter_by(id=film_id).update({
            "avg_rating": avg,
            "num_ratings": len(scores)
        })

    db.commit()
    print("✅ Valutazioni aggregate aggiornate")


def import_tags(db: Session, tags_path: str, max_rows: int = 500_000):
    print("🏷  Importazione tag...")
    batch = []
    with open(tags_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                batch.append(models.FilmTag(
                    film_id=int(row["movieId"]),
                    tag=row["tag"].strip().lower()
                ))
            except (ValueError, KeyError):
                continue

            if len(batch) >= 5000:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []

    if batch:
        db.bulk_save_objects(batch)
        db.commit()
    print(f"✅ Tag importati")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Importa MovieLens nel DB")
    parser.add_argument("--movies",  required=True, help="Path a movies.csv")
    parser.add_argument("--ratings", required=True, help="Path a ratings.csv")
    parser.add_argument("--tags",    default=None,  help="Path a tags.csv (opzionale)")
    parser.add_argument("--links",   default=None,  help="Path a links.csv (per TMDB)")
    parser.add_argument("--posters", action="store_true", help="Scarica poster da TMDB (lento)")
    parser.add_argument("--max-ratings", type=int, default=2_000_000,
                        help="Numero massimo di rating da importare (default 2M)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        import_movies(db, args.movies, args.links, args.posters)
        import_ratings(db, args.ratings, args.max_ratings)
        if args.tags:
            import_tags(db, args.tags)
        print("\n🎉 Import completato!")
    finally:
        db.close()
