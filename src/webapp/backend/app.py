from flask import Flask, request
from .config import Config
from .models import db
from .auth import auth_bp
from .movies import movies_bp
from .recommendations import recs_bp
from .profile import profile_bp
import math
import pandas as pd
import os

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "https://tesi-ten.vercel.app",
]


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    frontend_url = os.environ.get("FRONTEND_URL", "")
    if frontend_url and frontend_url not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append(frontend_url.rstrip("/"))

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        return response

    db.init_app(app)

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(movies_bp, url_prefix="/api/movies")
    app.register_blueprint(recs_bp, url_prefix="/api/recommendations")
    app.register_blueprint(profile_bp, url_prefix="/api/profile")

    with app.app_context():
        db.create_all()
        from .models import Movie
        if Movie.query.count() == 0:
            print("[app] Movie table is empty — importing from CSV...")
            _import_movies_from_csv(app)
            print(f"[app] imported {Movie.query.count()} movies")

        from . import recommender_service
        recommender_service.init(app.config)

    return app


def _nan_to_none(val):
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _import_movies_from_csv(app: Flask) -> None:
    from .models import Movie

    movies_csv = app.config["MOVIES_CSV"]
    df = pd.read_csv(movies_csv)

    batch = []
    for _, row in df.iterrows():
        year_raw = _nan_to_none(row.get("year"))
        year = int(year_raw) if year_raw is not None else None

        tmdb_raw = _nan_to_none(row.get("tmdbId"))
        tmdb_id = int(float(tmdb_raw)) if tmdb_raw is not None else None

        pop_raw = _nan_to_none(row.get("popularity"))
        popularity = float(pop_raw) if pop_raw is not None else None

        batch.append(Movie(
            id          = int(row["movieId"]),
            title       = str(row["title"]),
            title_clean = str(row["title_clean"]) if _nan_to_none(row.get("title_clean")) else None,
            year        = year,
            genres      = str(row["genres"]) if _nan_to_none(row.get("genres")) else None,
            overview_en = str(row["overview_en"]) if _nan_to_none(row.get("overview_en")) else None,
            overview_it = str(row["overview_it"]) if _nan_to_none(row.get("overview_it")) else None,
            director    = str(row["director"]) if _nan_to_none(row.get("director")) else None,
            actors_top5 = str(row["actors_top5"]) if _nan_to_none(row.get("actors_top5")) else None,
            poster_url  = str(row["poster_url"]) if _nan_to_none(row.get("poster_url")) else None,
            popularity  = popularity,
            tmdb_id     = tmdb_id,
        ))

        if len(batch) >= 500:
            db.session.bulk_save_objects(batch)
            db.session.commit()
            batch = []

    if batch:
        db.session.bulk_save_objects(batch)
        db.session.commit()


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
