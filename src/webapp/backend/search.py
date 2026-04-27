from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os

search_bp = Blueprint("search", __name__)

_search_embeddings = None
_search_index = None
_model = None


def _load_search_artefacts():
    global _search_embeddings, _search_index, _model

    base = Path(os.environ.get(
        "APP_BASE_DIR",
        str(Path(__file__).resolve().parent.parent.parent.parent)
    )) / "data" / "deploy_artifacts"

    emb_path = base / "search_embeddings_minilm.npy"
    idx_path = base / "search_embeddings_index.csv"

    if not emb_path.exists():
        print("[search] WARNING: search embeddings not found, semantic search disabled")
        return False

    _search_embeddings = np.load(str(emb_path))
    _search_index = pd.read_csv(idx_path)

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        print(f"[search] ready — {len(_search_index)} films, model loaded")
        return True
    except Exception as e:
        print(f"[search] WARNING: model load failed: {e}")
        return False


_search_ready = _load_search_artefacts()


@search_bp.route("/semantic")
def semantic_search():
    """GET /api/search/semantic?q=film+con+maghi+anni+70&top_k=10"""
    from .models import Movie

    query = request.args.get("q", "").strip()
    top_k = min(int(request.args.get("top_k", 10)), 50)

    if not query:
        return jsonify({"error": "query parameter 'q' is required"}), 400

    if not _search_ready or _model is None:
        return jsonify({
            "error": "semantic search not available",
            "fallback": "use /api/movies?search= for title search",
        }), 503

    if len(query) < 3:
        return jsonify({"error": "query too short (min 3 characters)"}), 400

    query_embedding = _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    similarities = _search_embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        movie_id = int(_search_index.iloc[idx]["movieId"])
        score = float(similarities[idx])
        movie = Movie.query.get(movie_id)
        if movie:
            results.append({
                "movie_id": movie_id,
                "title": movie.title_clean or movie.title,
                "year": movie.year,
                "genres": movie.genres,
                "poster_url": movie.poster_url,
                "overview_en": movie.overview_en,
                "similarity_score": round(score, 4),
            })

    return jsonify({
        "query": query,
        "results": results,
        "n_results": len(results),
        "model": "paraphrase-MiniLM-L6-v2",
    })
