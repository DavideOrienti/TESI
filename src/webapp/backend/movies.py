from __future__ import annotations

from flask import Blueprint, request, jsonify, make_response

from .models import db, Movie, Rating, Favorite
from .auth import token_required
from . import recommender_service

movies_bp = Blueprint("movies", __name__)


@movies_bp.route("", methods=["OPTIONS"])
@movies_bp.route("/<int:movie_id>", methods=["OPTIONS"])
@movies_bp.route("/<int:movie_id>/rate", methods=["OPTIONS"])
@movies_bp.route("/<int:movie_id>/favorite", methods=["OPTIONS"])
def options_handler(**kwargs):
    return make_response("", 200)


def _movie_to_dict(movie: Movie) -> dict:
    return {
        "id": movie.id,
        "title": movie.title,
        "title_clean": movie.title_clean,
        "year": movie.year,
        "genres": movie.genres,
        "overview_en": movie.overview_en,
        "overview_it": movie.overview_it,
        "director": movie.director,
        "actors_top5": movie.actors_top5,
        "poster_url": movie.poster_url,
        "popularity": movie.popularity,
        "tmdb_id": movie.tmdb_id,
    }


# ---------------------------------------------------------------------------
# GET /api/movies
# ---------------------------------------------------------------------------

@movies_bp.route("", methods=["GET"])
def list_movies():
    page = max(1, int(request.args.get("page", 1)))
    limit = min(100, max(1, int(request.args.get("limit", 20))))
    genre = request.args.get("genre", "").strip()
    search = request.args.get("search", "").strip()

    query = Movie.query
    if genre:
        query = query.filter(Movie.genres.ilike(f"%{genre}%"))
    if search:
        query = query.filter(Movie.title_clean.ilike(f"%{search}%"))

    total = query.count()
    movies = query.order_by(Movie.popularity.desc().nullslast()).offset((page - 1) * limit).limit(limit).all()

    return jsonify({
        "movies": [_movie_to_dict(m) for m in movies],
        "total": total,
        "page": page,
        "limit": limit,
    }), 200


# ---------------------------------------------------------------------------
# GET /api/movies/<movie_id>
# ---------------------------------------------------------------------------

@movies_bp.route("/<int:movie_id>", methods=["GET"])
def movie_detail(movie_id: int):
    movie = db.session.get(Movie, movie_id)
    if movie is None:
        return jsonify({"error": "Film non trovato"}), 404

    user_rating = None
    is_favorite = False

    # Dati utente opzionali se il token è presente
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        import jwt
        from flask import current_app
        try:
            from .models import User
            token = auth_header.split(" ", 1)[1]
            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET_KEY"],
                algorithms=["HS256"],
            )
            user_id = payload["user_id"]
            r = Rating.query.filter_by(user_id=user_id, movie_id=movie_id).first()
            if r:
                user_rating = r.rating
            fav = Favorite.query.filter_by(user_id=user_id, movie_id=movie_id).first()
            is_favorite = fav is not None
        except Exception:
            pass

    similar = recommender_service.get_similar_movies(movie_id, top_k=10)

    return jsonify({
        "movie": _movie_to_dict(movie),
        "user_rating": user_rating,
        "is_favorite": is_favorite,
        "similar_movies": similar,
    }), 200


# ---------------------------------------------------------------------------
# POST /api/movies/<movie_id>/rate
# ---------------------------------------------------------------------------

@movies_bp.route("/<int:movie_id>/rate", methods=["POST"])
@token_required
def rate_movie(current_user, movie_id: int):
    if db.session.get(Movie, movie_id) is None:
        return jsonify({"error": "Film non trovato"}), 404
    data = request.get_json(silent=True) or {}
    try:
        rating_val = float(data["rating"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Campo 'rating' mancante o non valido"}), 400

    if not (0.5 <= rating_val <= 5.0):
        return jsonify({"error": "Il rating deve essere tra 0.5 e 5.0"}), 400

    existing = Rating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    if existing:
        existing.rating = rating_val
    else:
        db.session.add(Rating(user_id=current_user.id, movie_id=movie_id, rating=rating_val))
    db.session.commit()

    return jsonify({"success": True, "rating": rating_val}), 200


# ---------------------------------------------------------------------------
# POST /api/movies/<movie_id>/favorite
# ---------------------------------------------------------------------------

@movies_bp.route("/<int:movie_id>/favorite", methods=["POST"])
@token_required
def toggle_favorite(current_user, movie_id: int):
    if db.session.get(Movie, movie_id) is None:
        return jsonify({"error": "Film non trovato"}), 404
    existing = Favorite.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({"success": True, "is_favorite": False}), 200

    db.session.add(Favorite(user_id=current_user.id, movie_id=movie_id))
    db.session.commit()
    return jsonify({"success": True, "is_favorite": True}), 200
