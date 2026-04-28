from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os
import base64
import io

visual_bp = Blueprint("visual", __name__)

IS_RENDER = os.environ.get("RENDER", "")

_clip_embeddings = None
_clip_index = None
_clip_ready = False


def _load_clip_artefacts():
    global _clip_embeddings, _clip_index, _clip_ready
    try:
        base = Path(os.environ.get(
            "APP_BASE_DIR",
            str(Path(__file__).resolve().parent.parent.parent.parent)
        )) / "data" / "deploy_artifacts"

        emb_path = base / "poster_embeddings_clip.npy"
        idx_path = base / "poster_embeddings_index.csv"

        if not emb_path.exists():
            print("[visual] CLIP embeddings not found — visual search disabled")
            return

        _clip_embeddings = np.load(str(emb_path))
        _clip_index = pd.read_csv(idx_path)
        _clip_ready = True
        print(f"[visual] ready — {len(_clip_index)} posters indexed, "
              f"dim={_clip_embeddings.shape[1]}")

    except Exception as e:
        print(f"[visual] load error: {e}")


_load_clip_artefacts()


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


def _get_similar_by_embedding(
    query_embedding: np.ndarray,
    top_k: int = 10,
    exclude_movie_id: int = None,
) -> list[dict]:
    similarities = _clip_embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        movie_id = int(_clip_index.iloc[idx]["movieId"])
        if exclude_movie_id and movie_id == exclude_movie_id:
            continue
        score = float(similarities[idx])
        movie = _db_get_movie(movie_id)
        if movie:
            results.append({
                "movie_id": movie_id,
                "title": movie.get("title"),
                "year": movie.get("year"),
                "genres": movie.get("genres"),
                "poster_url": movie.get("poster_url"),
                "visual_similarity": round(score, 4),
            })
    return results


@visual_bp.route("/similar/<int:movie_id>")
def similar_by_movie(movie_id: int):
    if not _clip_ready:
        return jsonify({"error": "visual search not available"}), 503

    top_k = min(int(request.args.get("top_k", 10)), 50)

    mask = _clip_index["movieId"] == movie_id
    if not mask.any():
        return jsonify({"error": f"no poster embedding for movie {movie_id}"}), 404

    idx = mask.idxmax()
    query_embedding = _clip_embeddings[idx]

    results = _get_similar_by_embedding(
        query_embedding=query_embedding,
        top_k=top_k,
        exclude_movie_id=movie_id,
    )

    return jsonify({
        "movie_id": movie_id,
        "results": results,
        "n_results": len(results),
        "method": "CLIP ViT-B/32 cosine similarity",
    })


@visual_bp.route("/search", methods=["POST"])
def search_by_image():
    if IS_RENDER:
        return jsonify({
            "error": "image upload not available on free tier",
            "message": "Use /api/visual/similar/<movie_id> instead",
            "upgrade_needed": True,
        }), 503

    if not _clip_ready:
        return jsonify({"error": "visual search not available"}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "image field required (base64)"}), 400

    top_k = min(int(data.get("top_k", 10)), 50)

    try:
        import PIL.Image as PILImage
        img_data = base64.b64decode(data["image"])
        img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224), PILImage.LANCZOS)
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    try:
        import torch
        import clip as openai_clip

        if not hasattr(search_by_image, "_model"):
            device = "cpu"
            search_by_image._model, search_by_image._preprocess = \
                openai_clip.load("ViT-B/32", device=device)
            search_by_image._model.eval()
            search_by_image._device = device

        img_tensor = search_by_image._preprocess(img).unsqueeze(0)
        img_tensor = img_tensor.to(search_by_image._device)

        with torch.no_grad():
            emb = search_by_image._model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            query_embedding = emb.cpu().numpy()[0]

    except Exception as e:
        return jsonify({"error": f"encoding failed: {e}"}), 500

    results = _get_similar_by_embedding(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return jsonify({
        "results": results,
        "n_results": len(results),
        "method": "CLIP ViT-B/32 image upload",
    })


@visual_bp.route("/version")
def visual_version():
    return jsonify({
        "version": "clip_v1",
        "clip_ready": _clip_ready,
        "posters_indexed": len(_clip_index) if _clip_index is not None else 0,
        "embedding_dim": int(_clip_embeddings.shape[1]) if _clip_ready else 0,
        "image_upload_available": not bool(IS_RENDER),
    })


@visual_bp.route("/similar/<int:movie_id>", methods=["OPTIONS"])
@visual_bp.route("/search", methods=["OPTIONS"])
@visual_bp.route("/version", methods=["OPTIONS"])
def visual_options(**kwargs):
    return "", 200
