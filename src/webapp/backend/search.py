from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os
import requests as http_requests

search_bp = Blueprint("search", __name__)

_search_embeddings = None
_search_index = None
_search_ready = False


def _load_search_artefacts():
    global _search_embeddings, _search_index, _search_ready
    try:
        base = Path(os.environ.get(
            "APP_BASE_DIR",
            str(Path(__file__).resolve().parent.parent.parent.parent)
        )) / "data" / "deploy_artifacts"

        emb_path = base / "search_embeddings_minilm.npy"
        idx_path = base / "search_embeddings_index.csv"

        if not emb_path.exists():
            print("[search] embeddings not found — semantic search disabled")
            return

        _search_embeddings = np.load(str(emb_path))
        _search_index = pd.read_csv(idx_path)
        _search_ready = True
        print(f"[search] ready — {len(_search_index)} films indexed")

    except Exception as e:
        print(f"[search] load error: {e}")


_load_search_artefacts()


def _encode_via_hf_api(query: str) -> np.ndarray | None:
    """Chiama Hugging Face Inference API — nessun modello caricato in locale."""
    token = os.environ.get("HF_API_TOKEN", "")
    if not token:
        print("[search] HF_API_TOKEN not set")
        return None

    url = (
        "https://api-inference.huggingface.co/pipeline/"
        "feature-extraction/"
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = http_requests.post(
            url,
            headers=headers,
            json={"inputs": query, "options": {"wait_for_model": True}},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[search] HF API error: {resp.status_code}")
            return None

        data = resp.json()
        arr = np.array(data, dtype=np.float32)

        # HF restituisce shape (1, seq_len, dim) o (seq_len, dim)
        if arr.ndim == 3:
            arr = arr[0]       # → (seq_len, dim)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # mean pooling → (dim,)

        norm = np.linalg.norm(arr)
        if norm > 1e-9:
            arr = arr / norm

        return arr

    except Exception as e:
        print(f"[search] HF API exception: {e}")
        return None


@search_bp.route("/semantic")
def semantic_search():
    query = request.args.get("q", "").strip()
    top_k = min(int(request.args.get("top_k", 10)), 50)

    if not query or len(query) < 3:
        return jsonify({"error": "query too short (min 3 chars)"}), 400

    if not _search_ready:
        return jsonify({"error": "search index not available"}), 503

    embedding = _encode_via_hf_api(query)
    if embedding is None:
        return jsonify({
            "error": "encoding service unavailable — retry in a moment",
            "retry": True,
        }), 503

    if embedding.shape[0] != _search_embeddings.shape[1]:
        return jsonify({
            "error": (
                f"embedding dim mismatch: "
                f"got {embedding.shape[0]}, "
                f"expected {_search_embeddings.shape[1]}"
            )
        }), 500

    similarities = _search_embeddings @ embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        movie_id = int(_search_index.iloc[idx]["movieId"])
        score = float(similarities[idx])
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
        "model": "paraphrase-MiniLM-L6-v2 via HF API",
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
