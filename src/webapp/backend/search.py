from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os

search_bp = Blueprint("search", __name__)

# ── CONFIGURAZIONE ───────────────────────────────
BASE_DIR = Path(os.environ.get(
    "APP_BASE_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent)
)) / "data" / "deploy_artifacts"

# ── INDICE MINILM (per ONNX) ─────────────────────
_minilm_embeddings = None
_minilm_index = None
_minilm_ready = False

# ── INDICE TF-IDF (fallback) ─────────────────────
_tfidf_matrix = None
_tfidf_vectorizer = None
_tfidf_index = None
_tfidf_ready = False

# ── ONNX SESSION ─────────────────────────────────
_onnx_session = None
_onnx_tokenizer = None
_onnx_ready = False
_onnx_expected_inputs = None


def _load_minilm_index():
    global _minilm_embeddings, _minilm_index, _minilm_ready
    try:
        emb_path = BASE_DIR / "search_embeddings_minilm.npy"
        idx_path = BASE_DIR / "search_embeddings_index.csv"
        if emb_path.exists() and idx_path.exists():
            _minilm_embeddings = np.load(str(emb_path))
            _minilm_index = pd.read_csv(idx_path)
            _minilm_ready = True
            print(f"[search] MiniLM index ready — "
                  f"{len(_minilm_index)} films, "
                  f"dim={_minilm_embeddings.shape[1]}")
        else:
            print("[search] MiniLM index not found — will use TF-IDF only")
    except Exception as e:
        print(f"[search] MiniLM index error: {e}")


def _load_onnx_model():
    global _onnx_session, _onnx_tokenizer, _onnx_ready, _onnx_expected_inputs
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = BASE_DIR / "minilm_onnx" / "model.onnx"
        tok_path = BASE_DIR / "minilm_onnx"
        if not onnx_path.exists():
            print("[search] ONNX model not found — using TF-IDF only")
            return

        _onnx_tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
        _onnx_session = ort.InferenceSession(str(onnx_path))
        _onnx_expected_inputs = {i.name for i in _onnx_session.get_inputs()}
        _onnx_ready = True
        print(f"[search] ONNX MiniLM ready — inputs: {_onnx_expected_inputs}")
    except Exception as e:
        print(f"[search] ONNX load error: {e}")


def _build_tfidf_index():
    global _tfidf_matrix, _tfidf_vectorizer, _tfidf_index, _tfidf_ready
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        movies_path = BASE_DIR / "movies_enriched_tmdb.csv"
        if not movies_path.exists():
            return

        df = pd.read_csv(movies_path)

        def build_text(row):
            parts = []
            for field in ["title_clean", "genres", "director",
                          "actors_top5", "overview_en", "overview_it"]:
                val = row.get(field)
                if pd.notna(val) and str(val).strip():
                    parts.append(str(val).replace("|", " ").replace("_", " "))
            return " ".join(parts)

        texts = df.apply(build_text, axis=1).tolist()
        _tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        _tfidf_matrix = _tfidf_vectorizer.fit_transform(texts)
        _tfidf_index = df[["movieId", "title_clean"]].reset_index(drop=True)
        _tfidf_ready = True
        print(f"[search] TF-IDF ready — {len(texts)} films, matrix {_tfidf_matrix.shape}")
    except Exception as e:
        print(f"[search] TF-IDF error: {e}")


def _encode_with_onnx(query: str) -> np.ndarray | None:
    """Encoding con MiniLM ONNX — qualità semantica piena."""
    try:
        inputs = _onnx_tokenizer(
            query,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128,
        )
        # Cast int64 e filtra solo gli input attesi dal grafo ONNX
        onnx_inputs = {
            k: v.astype(np.int64)
            for k, v in inputs.items()
            if k in _onnx_expected_inputs
        }
        outputs = _onnx_session.run(None, onnx_inputs)
        token_embeddings = outputs[0]
        mask = inputs["attention_mask"][:, :, np.newaxis].astype(np.float32)
        sum_emb = (token_embeddings * mask).sum(axis=1)
        sum_mask = mask.sum(axis=1).clip(min=1e-9)
        embedding = sum_emb / sum_mask
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return (embedding / norm)[0].astype(np.float32)
    except Exception as e:
        print(f"[search] ONNX encode error: {e}")
        return None


def _search_with_minilm(query: str, top_k: int) -> list[dict]:
    """Ricerca con MiniLM ONNX + indice precomputato."""
    embedding = _encode_with_onnx(query)
    if embedding is None:
        return []

    similarities = _minilm_embeddings @ embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] < 0.1:
            break
        movie_id = int(_minilm_index.iloc[idx]["movieId"])
        score = float(similarities[idx])
        movie = _db_get_movie(movie_id)
        if movie:
            movie["similarity_score"] = round(score, 4)
            movie["search_model"] = "minilm-onnx"
            results.append(movie)
    return results


def _search_with_tfidf(query: str, top_k: int) -> list[dict]:
    """Ricerca con TF-IDF — fallback."""
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if sims[idx] < 0.01:
            break
        movie_id = int(_tfidf_index.iloc[idx]["movieId"])
        movie = _db_get_movie(movie_id)
        if movie:
            movie["similarity_score"] = round(float(sims[idx]), 4)
            movie["search_model"] = "tfidf"
            results.append(movie)
    return results


def _db_get_movie(movie_id: int) -> dict | None:
    try:
        from .models import Movie
        m = Movie.query.get(movie_id)
        if not m:
            return None
        return {
            "movie_id": m.id,
            "title": m.title_clean or m.title,
            "year": m.year,
            "genres": m.genres,
            "poster_url": m.poster_url,
            "overview_en": m.overview_en,
        }
    except Exception:
        return None


# Carica tutto all'avvio
_load_minilm_index()
_load_onnx_model()
_build_tfidf_index()


@search_bp.route("/semantic")
def semantic_search():
    query = request.args.get("q", "").strip()
    top_k = min(int(request.args.get("top_k", 10)), 50)

    if not query or len(query) < 3:
        return jsonify({"error": "query too short"}), 400

    if _onnx_ready and _minilm_ready:
        results = _search_with_minilm(query, top_k)
        model_used = "minilm-onnx"
        if not results:
            results = _search_with_tfidf(query, top_k)
            model_used = "tfidf-fallback"
    elif _tfidf_ready:
        results = _search_with_tfidf(query, top_k)
        model_used = "tfidf"
    else:
        return jsonify({"error": "search not available"}), 503

    return jsonify({
        "query": query,
        "results": results,
        "n_results": len(results),
        "model": model_used,
    })


@search_bp.route("/version")
def search_version():
    return jsonify({
        "version": "minilm_onnx_v1",
        "onnx_ready": _onnx_ready,
        "tfidf_ready": _tfidf_ready,
        "minilm_index_ready": _minilm_ready,
        "films_indexed": len(_minilm_index) if _minilm_ready else 0,
        "active_model": "minilm-onnx" if (_onnx_ready and _minilm_ready) else "tfidf",
    })


@search_bp.route("/semantic", methods=["OPTIONS"])
@search_bp.route("/version", methods=["OPTIONS"])
def search_options(**kwargs):
    return "", 200
