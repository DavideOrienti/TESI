from __future__ import annotations

from flask import Blueprint, request, jsonify, make_response

from .models import db, Movie, Rating, Favorite, User
from .auth import token_required
from . import recommender_service

recs_bp = Blueprint("recommendations", __name__)


@recs_bp.route("", methods=["OPTIONS"])
@recs_bp.route("/popular", methods=["OPTIONS"])
@recs_bp.route("/social", methods=["OPTIONS"])
def options_handler():
    return make_response("", 200)


def _enrich(rec: dict) -> dict | None:
    movie = db.session.get(Movie, rec["movie_id"])
    if movie is None:
        return None
    return {
        "movie_id": movie.id,
        "title": movie.title_clean or movie.title,
        "year": movie.year,
        "genres": movie.genres,
        "poster_url": movie.poster_url,
        "director": movie.director,
        "score": rec["score"],
        "explanation": {
            "type": rec["explanation"]["type"],
            "seed_movies": [
                {"title": s["title"], "similarity": s["similarity"]}
                for s in rec["explanation"]["seed_movies"]
            ],
        },
    }


# ---------------------------------------------------------------------------
# GET /api/recommendations
# ---------------------------------------------------------------------------

@recs_bp.route("", methods=["GET"])
@token_required
def recommendations(current_user):
    top_k = min(50, max(1, int(request.args.get("top_k", 20))))

    user_ratings_rows = Rating.query.filter_by(user_id=current_user.id).all()
    seen_movie_ids = [r.movie_id for r in user_ratings_rows]
    user_ratings = {r.movie_id: r.rating for r in user_ratings_rows}

    favorites = Favorite.query.filter_by(user_id=current_user.id).all()
    favorite_ids = [f.movie_id for f in favorites]

    raw = recommender_service.get_recommendations(
        user_id=current_user.id,
        seen_movie_ids=seen_movie_ids,
        user_ratings=user_ratings,
        favorite_movie_ids=favorite_ids,
        top_k=top_k,
    )

    enriched = [e for rec in raw if (e := _enrich(rec)) is not None]

    return jsonify({"recommendations": enriched}), 200


# ---------------------------------------------------------------------------
# GET /api/recommendations/popular
# ---------------------------------------------------------------------------

@recs_bp.route("/popular", methods=["GET"])
def popular():
    top_k = min(50, max(1, int(request.args.get("top_k", 20))))
    movie_ids = recommender_service.get_popular_movies(top_k=top_k)

    movies = []
    for mid in movie_ids:
        m = db.session.get(Movie, mid)
        if m:
            movies.append({
                "movie_id": m.id,
                "title": m.title_clean or m.title,
                "year": m.year,
                "genres": m.genres,
                "poster_url": m.poster_url,
                "popularity": m.popularity,
            })

    return jsonify({"movies": movies}), 200


# ---------------------------------------------------------------------------
# GET /api/recommendations/social
# ---------------------------------------------------------------------------

@recs_bp.route("/social", methods=["GET"])
@token_required
def social_recommendations(current_user):
    top_k = min(50, max(1, int(request.args.get("top_k", 10))))

    # Rating utente corrente
    my_rows = Rating.query.filter_by(user_id=current_user.id).all()
    my_ratings = {r.movie_id: r.rating for r in my_rows}

    if len(my_ratings) < 2:
        return jsonify({
            "recommendations": [],
            "has_enough_data": False,
            "message": "Valuta almeno 2 film per attivare le raccomandazioni sociali.",
        }), 200

    # Rating di tutti gli altri utenti
    all_rows = Rating.query.filter(Rating.user_id != current_user.id).all()
    users_ratings: dict[int, dict[int, float]] = {}
    for r in all_rows:
        users_ratings.setdefault(r.user_id, {})[r.movie_id] = r.rating

    all_users_list = [
        {"user_id": uid, "ratings": ratings}
        for uid, ratings in users_ratings.items()
    ]

    raw = recommender_service.get_social_recommendations(
        current_user_id=current_user.id,
        current_user_ratings=my_ratings,
        all_users_ratings=all_users_list,
        top_k=top_k,
    )

    if len(raw) < 1:
        return jsonify({
            "recommendations": [],
            "has_enough_data": False,
            "message": "Non ci sono ancora abbastanza utenti con gusti simili ai tuoi.",
        }), 200

    enriched = []
    for rec in raw:
        m = db.session.get(Movie, rec["movie_id"])
        if m is None:
            continue
        enriched.append({
            "movie_id": m.id,
            "title": m.title_clean or m.title,
            "year": m.year,
            "genres": m.genres,
            "poster_url": m.poster_url,
            "director": m.director,
            "score": rec["score"],
            "explanation": rec["explanation"],
        })

    return jsonify({
        "recommendations": enriched,
        "has_enough_data": True,
        "message": "",
    }), 200
