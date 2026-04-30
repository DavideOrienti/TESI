from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import os
import base64
import io

visual_bp = Blueprint("visual", __name__)

# True solo su Render (RENDER env var è impostata da Render automaticamente)
IS_RENDER = bool(os.environ.get("RENDER", ""))

_clip_embeddings = None
_clip_index = None
_phash_index = None
_clip_ready = False
_phash_ready = False


def _load_artefacts():
    global _clip_embeddings, _clip_index, _clip_ready
    global _phash_index, _phash_ready

    base = Path(os.environ.get(
        "APP_BASE_DIR",
        str(Path(__file__).resolve().parent.parent.parent.parent)
    )) / "data" / "deploy_artifacts"

    emb_path = base / "poster_embeddings_clip.npy"
    idx_path = base / "poster_embeddings_index.csv"
    if emb_path.exists() and idx_path.exists():
        _clip_embeddings = np.load(str(emb_path))
        _clip_index = pd.read_csv(idx_path)
        _clip_ready = True
        print(f"[visual] CLIP ready — {len(_clip_index)} posters")

    phash_path = base / "poster_phash_index.csv"
    if phash_path.exists():
        _phash_index = pd.read_csv(phash_path)
        _phash_ready = True
        print(f"[visual] phash ready — {len(_phash_index)} hashes")


_load_artefacts()


def _db_get_movie_full(movie_id: int) -> dict | None:
    try:
        from .models import Movie
        m = Movie.query.get(movie_id)
        if not m:
            return None
        return {
            "movie_id": int(m.id),
            "title": m.title_clean or m.title,
            "year": m.year,
            "genres": m.genres,
            "poster_url": m.poster_url,
            "overview_en": m.overview_en,
            "director": m.director,
            "actors_top5": m.actors_top5,
        }
    except Exception:
        return None


def _get_similar_clip(query_emb, top_k, exclude_id=None):
    if not _clip_ready:
        return []
    sims = _clip_embeddings @ query_emb
    indices = np.argsort(sims)[::-1]
    results = []
    for idx in indices:
        if len(results) >= top_k:
            break
        mid = int(_clip_index.iloc[idx]["movieId"])
        if exclude_id and mid == exclude_id:
            continue
        movie = _db_get_movie_full(mid)
        if movie:
            movie["visual_similarity"] = round(float(sims[idx]), 4)
            results.append(movie)
    return results


def _recognize_phash(img_pil, top_k=10):
    if not _phash_ready:
        return None, []
    try:
        import imagehash
        query_hash = imagehash.phash(img_pil, hash_size=16)

        best_dist = float("inf")
        best_idx = None

        for i, row in _phash_index.iterrows():
            try:
                stored = imagehash.hex_to_hash(str(row["phash"]))
                dist = query_hash - stored
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            except Exception:
                continue

        if best_idx is None:
            return None, []

        best_movie_id = int(_phash_index.iloc[best_idx]["movieId"])
        similarity = round(1.0 - best_dist / 256, 4)

        recognized = _db_get_movie_full(best_movie_id)
        if recognized:
            recognized["similarity"] = similarity
            recognized["method"] = "phash"
            recognized["hash_distance"] = int(best_dist)

        similar = []
        if _clip_ready:
            mask = _clip_index["movieId"] == best_movie_id
            if mask.any():
                idx = mask.idxmax()
                similar = _get_similar_clip(
                    _clip_embeddings[idx], top_k, exclude_id=best_movie_id
                )

        return recognized, similar

    except Exception as e:
        print(f"[visual] phash error: {e}")
        return None, []


def _recognize_clip(img_pil, top_k=10):
    try:
        import torch
        import clip as openai_clip

        if not hasattr(_recognize_clip, "_model"):
            device = "cpu"
            _recognize_clip._model, _recognize_clip._preprocess = \
                openai_clip.load("ViT-B/32", device=device)
            _recognize_clip._model.eval()
            _recognize_clip._device = device
            print("[visual] CLIP model loaded for inference")

        img_tensor = _recognize_clip._preprocess(img_pil).unsqueeze(0)
        img_tensor = img_tensor.to(_recognize_clip._device)

        with torch.no_grad():
            emb = _recognize_clip._model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            query_emb = emb.cpu().numpy()[0]

        similar_all = _get_similar_clip(query_emb, top_k + 1)
        if not similar_all:
            return None, []

        recognized = dict(similar_all[0])
        recognized["similarity"] = recognized.pop("visual_similarity", 0)
        recognized["method"] = "clip"
        similar = similar_all[1 : top_k + 1]

        return recognized, similar

    except Exception as e:
        print(f"[visual] CLIP inference error: {e}")
        return None, []


@visual_bp.route("/search", methods=["POST"])
def search_by_image():
    print(f"[visual/search] Content-Type: {request.content_type}")
    print(f"[visual/search] Content-Length: {request.content_length}")

    data = request.get_json(silent=True)
    if data is None:
        raw = request.get_data(as_text=True)
        print(f"[visual/search] get_json failed, raw[:100]: {raw[:100]}")
        return jsonify({"error": "invalid JSON", "content_type": request.content_type}), 400

    if "image" not in data:
        print(f"[visual/search] missing image field, keys: {list(data.keys())}")
        return jsonify({"error": "image field required (base64)", "received_keys": list(data.keys())}), 400

    image_b64 = data["image"]
    print(f"[visual/search] image b64 length: {len(image_b64)}")
    print(f"[visual/search] image b64 prefix: {image_b64[:30]}")

    top_k = min(int(data.get("top_k", 5)), 20)

    try:
        from PIL import Image as PILImage
        img_data = base64.b64decode(image_b64)
        print(f"[visual/search] decoded bytes: {len(img_data)}")
        img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
        img = img.resize((224, 224), PILImage.LANCZOS)
        print(f"[visual/search] image resized OK")
    except Exception as e:
        print(f"[visual/search] image decode/open error: {e}")
        return jsonify({"error": f"invalid image: {e}"}), 400

    print(f"[visual/search] IS_RENDER={IS_RENDER}, routing to {'phash' if IS_RENDER else 'clip'}")

    if IS_RENDER:
        recognized, similar = _recognize_phash(img, top_k)
        method_used = "phash"
    else:
        recognized, similar = _recognize_clip(img, top_k)
        method_used = "clip"

    print(f"[visual/search] recognized={recognized is not None}, n_similar={len(similar)}")

    if not recognized:
        return jsonify({
            "error": "could not recognize film",
            "method": method_used,
        }), 404

    return jsonify({
        "recognized_movie": recognized,
        "similar_movies": similar,
        "n_similar": len(similar),
        "method_used": method_used,
    })


@visual_bp.route("/similar/<int:movie_id>")
def similar_by_movie(movie_id: int):
    if not _clip_ready:
        return jsonify({"error": "visual search not available"}), 503

    top_k = min(int(request.args.get("top_k", 10)), 50)

    mask = _clip_index["movieId"] == movie_id
    if not mask.any():
        return jsonify({"error": f"no poster for movie {movie_id}"}), 404

    idx = mask.idxmax()
    query_emb = _clip_embeddings[idx]
    results = _get_similar_clip(query_emb, top_k, exclude_id=movie_id)

    return jsonify({
        "movie_id": movie_id,
        "results": results,
        "n_results": len(results),
        "method": "CLIP ViT-B/32",
    })


@visual_bp.route("/version")
def visual_version():
    return jsonify({
        "version": "clip_phash_v2",
        "clip_ready": _clip_ready,
        "phash_ready": _phash_ready,
        "posters_indexed": len(_clip_index) if _clip_ready else 0,
        "embedding_dim": int(_clip_embeddings.shape[1]) if _clip_ready else 0,
        "is_render": IS_RENDER,
        "image_upload_method": "phash" if IS_RENDER else "clip",
    })


@visual_bp.route("/search", methods=["OPTIONS"])
@visual_bp.route("/similar/<int:movie_id>", methods=["OPTIONS"])
@visual_bp.route("/version", methods=["OPTIONS"])
def visual_options(**kwargs):
    return "", 200
