from __future__ import annotations
import math
from flask import Blueprint, jsonify
from .models import db, User, Rating, Movie
from .auth import token_required

social_bp = Blueprint("social", __name__)

SIMILARITY_THRESHOLD = 0.3
MAX_USERS = 200


def _cosine_similarity(vec_a: dict[int, float], vec_b: dict[int, float]) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@social_bp.route("/graph", methods=["GET"])
@token_required
def get_social_graph(current_user):
    users = User.query.order_by(User.id).limit(MAX_USERS).all()
    user_ids = [u.id for u in users]

    all_ratings = (
        db.session.query(Rating.user_id, Rating.movie_id, Rating.rating)
        .filter(Rating.user_id.in_(user_ids))
        .all()
    )

    rating_vectors: dict[int, dict[int, float]] = {uid: {} for uid in user_ids}
    for user_id, movie_id, rating in all_ratings:
        rating_vectors[user_id][movie_id] = rating

    top_movies_map: dict[int, list[str]] = {}
    for uid in user_ids:
        vec = rating_vectors[uid]
        top = sorted(vec.items(), key=lambda x: -x[1])[:3]
        movie_ids = [mid for mid, _ in top]
        if movie_ids:
            movies = Movie.query.filter(Movie.id.in_(movie_ids)).all()
            title_map = {m.id: m.title for m in movies}
            top_movies_map[uid] = [title_map.get(mid, str(mid)) for mid in movie_ids]
        else:
            top_movies_map[uid] = []

    user_map = {u.id: u for u in users}

    nodes = []
    for uid in user_ids:
        vec = rating_vectors[uid]
        n = len(vec)
        avg = (sum(vec.values()) / n) if n > 0 else 0.0
        nodes.append({
            "id": uid,
            "username": user_map[uid].username,
            "n_ratings": n,
            "avg_rating": round(avg, 2),
            "top_movies": top_movies_map[uid],
        })

    edges = []
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            uid_a = user_ids[i]
            uid_b = user_ids[j]
            sim = _cosine_similarity(rating_vectors[uid_a], rating_vectors[uid_b])
            if sim >= SIMILARITY_THRESHOLD:
                edges.append({
                    "source": uid_a,
                    "target": uid_b,
                    "similarity": round(sim, 4),
                })

    for node in nodes:
        deg = sum(
            1 for e in edges
            if e["source"] == node["id"] or e["target"] == node["id"]
        )
        node["degree"] = deg

    return jsonify({"nodes": nodes, "edges": edges})
