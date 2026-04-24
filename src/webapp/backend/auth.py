from __future__ import annotations
import jwt
from datetime import datetime, timezone, timedelta
from functools import wraps

from flask import Blueprint, request, jsonify, current_app, make_response
from werkzeug.security import generate_password_hash, check_password_hash

from .models import db, User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["OPTIONS"])
@auth_bp.route("/login", methods=["OPTIONS"])
@auth_bp.route("/me", methods=["OPTIONS"])
def options_handler():
    return make_response("", 200)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token mancante"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET_KEY"],
                algorithms=["HS256"],
            )
            current_user = User.query.get(payload["user_id"])
            if current_user is None:
                return jsonify({"error": "Utente non trovato"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token scaduto"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Token non valido"}), 401
        return f(current_user, *args, **kwargs)
    return decorated


def _make_token(user: User, app) -> str:
    expires = datetime.now(timezone.utc) + timedelta(
        seconds=app.config["JWT_ACCESS_TOKEN_EXPIRES"]
    )
    payload = {"user_id": user.id, "exp": expires}
    return jwt.encode(payload, app.config["JWT_SECRET_KEY"], algorithm="HS256")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not email or not password:
        return jsonify({"error": "username, email e password sono obbligatori"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username già in uso"}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email già in uso"}), 409

    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()

    token = _make_token(user, current_app._get_current_object())
    return jsonify({"token": token, "user": {"id": user.id, "username": user.username}}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if user is None or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Credenziali non valide"}), 401

    token = _make_token(user, current_app._get_current_object())
    return jsonify({"token": token, "user": {"id": user.id, "username": user.username}}), 200


@auth_bp.route("/me", methods=["GET"])
@token_required
def me(current_user):
    return jsonify({
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
        }
    }), 200
