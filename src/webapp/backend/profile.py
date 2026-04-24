from __future__ import annotations

from flask import Blueprint, jsonify

from .models import db, Movie, Rating, Favorite
from .auth import token_required

profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/ratings", methods=["GET"])
@token_required
def get_ratings(current_user):
    rows = Rating.query.filter_by(user_id=current_user.id).order_by(Rating.created_at.desc()).all()
    result = []
    for r in rows:
        m = db.session.get(Movie, r.movie_id)
        result.append({
            "movie_id": r.movie_id,
            "rating": r.rating,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "title": m.title_clean or m.title if m else "",
            "poster_url": m.poster_url if m else None,
            "year": m.year if m else None,
            "genres": m.genres if m else None,
        })
    return jsonify({"ratings": result}), 200


@profile_bp.route("/favorites", methods=["GET"])
@token_required
def get_favorites(current_user):
    rows = Favorite.query.filter_by(user_id=current_user.id).order_by(Favorite.created_at.desc()).all()
    result = []
    for f in rows:
        m = db.session.get(Movie, f.movie_id)
        if m:
            result.append({
                "id": m.id,
                "title": m.title_clean or m.title,
                "title_clean": m.title_clean,
                "poster_url": m.poster_url,
                "year": m.year,
                "genres": m.genres,
            })
    return jsonify({"favorites": result}), 200
