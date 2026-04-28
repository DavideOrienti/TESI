from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os

search_bp = Blueprint("search", __name__)

_tfidf_matrix = None
_tfidf_vectorizer = None
_search_index = None
_search_ready = False


def _build_search_text(row) -> str:
    parts = []
    for field in ("title_clean", "title"):
        val = row.get(field)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val))
            break
    if pd.notna(row.get("year")):
        parts.append(str(int(row["year"])))
    if pd.notna(row.get("genres")):
        parts.append(str(row["genres"]).replace("|", " "))
    if pd.notna(row.get("tags")):
        parts.append(str(row["tags"]).replace(",", " "))
    if pd.notna(row.get("director")):
        parts.append(str(row["director"]))
    if pd.notna(row.get("actors_top5")):
        parts.append(str(row["actors_top5"]).replace("_", " "))
    for ov_field in ("overview_it", "overview_en"):
        val = row.get(ov_field)
        if pd.notna(val) and len(str(val)) > 10:
            parts.append(str(val))
    return " ".join(parts)


def _load_search_artefacts():
    global _tfidf_matrix, _tfidf_vectorizer, _search_index, _search_ready
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        base = Path(os.environ.get(
            "APP_BASE_DIR",
            str(Path(__file__).resolve().parent.parent.parent.parent)
        )) / "data" / "deploy_artifacts"

        movies_path = base / "movies_enriched_tmdb.csv"
        if not movies_path.exists():
            print("[search] movies CSV not found — search disabled")
            return

        df = pd.read_csv(movies_path)
        texts = [_build_search_text(row) for _, row in df.iterrows()]

        _tfidf_vectorizer = TfidfVectorizer(
            analyzer="word",
            min_df=2,
            ngram_range=(1, 2),
            max_features=60_000,
            sublinear_tf=True,
        )
        _tfidf_matrix = _tfidf_vectorizer.fit_transform(texts)
        _search_index = df[["movieId"]].copy().reset_index(drop=True)

        _search_ready = True
        print(f"[search] TF-IDF ready — {len(_search_index)} films, matrix {_tfidf_matrix.shape}")

    except Exception as e:
        print(f"[search] load error: {e}")


_load_search_artefacts()


@search_bp.route("/version")
def search_version():
    return jsonify({
        "version": "tfidf_v1",
        "search_ready": _search_ready,
        "films_indexed": len(_search_index) if _search_index is not None else 0,
        "matrix_shape": list(_tfidf_matrix.shape) if _tfidf_matrix is not None else None,
    })


@search_bp.route("/version", methods=["OPTIONS"])
def search_version_options():
    return "", 200


@search_bp.route("/semantic")
def semantic_search():
    query = request.args.get("q", "").strip()
    top_k = min(int(request.args.get("top_k", 10)), 50)

    if not query or len(query) < 3:
        return jsonify({"error": "query too short (min 3 chars)"}), 400

    if not _search_ready:
        return jsonify({"error": "search index not available"}), 503

    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]

    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(sims[idx])
        if score < 0.01:
            break
        movie_id = int(_search_index.iloc[idx]["movieId"])
        movie = _db_get_movie(movie_id)
        if movie:
            results.append({
                "movie_id": movie_id,
                "title": movie.get("title"),
                "year": movie.get("year"),
                "genres": movie.get("genres"),
                "poster_url": movie.get("poster_url"),
                "overview_en": movie.get("overview_en"),
                "similarity_score": round(score, 4),
            })

    return jsonify({
        "query": query,
        "results": results,
        "n_results": len(results),
        "model": "tfidf_v1",
    })


def _db_get_movie(movie_id: int) -> dict | None:
    try:
        from .models import Movie
        m = Movie.query.get(movie_id)
        if not m:
            return None
        return {
            "title": m.title_clean or m.title,
            "year": m.year,
            "genres": m.genres,
            "poster_url": m.poster_url,
            "overview_en": m.overview_en,
        }
    except Exception:
        return None
