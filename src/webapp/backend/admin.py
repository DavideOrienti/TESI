from flask import Blueprint, request, jsonify
import os
from .models import db, User, Rating, Favorite
from sqlalchemy import func

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/stats")
def stats():
    key = request.args.get("key", "")
    admin_key = os.environ.get("ADMIN_KEY", "admin-dev-key")

    if key != admin_key:
        return jsonify({"error": "Unauthorized"}), 403

    users = User.query.all()

    users_data = []
    for u in users:
        n_ratings = Rating.query.filter_by(user_id=u.id).count()
        n_favorites = Favorite.query.filter_by(user_id=u.id).count()
        avg_rating = db.session.query(
            func.avg(Rating.rating)
        ).filter_by(user_id=u.id).scalar()

        users_data.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "registered_at": u.created_at.isoformat() if u.created_at else None,
            "n_ratings": n_ratings,
            "n_favorites": n_favorites,
            "avg_rating": round(float(avg_rating), 2) if avg_rating else 0.0,
        })

    users_data.sort(key=lambda x: x["registered_at"] or "", reverse=True)

    total_ratings = Rating.query.count()
    total_favorites = Favorite.query.count()

    return jsonify({
        "totals": {
            "n_users": len(users_data),
            "n_ratings": total_ratings,
            "n_favorites": total_favorites,
        },
        "users": users_data,
    })

