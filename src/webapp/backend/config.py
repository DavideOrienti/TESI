import os
from pathlib import Path

# Percorso base del progetto
BASE_DIR = Path(os.environ.get(
    "APP_BASE_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent)
))

# Database path — /tmp/ su Render, locale altrimenti
_IS_RENDER = os.environ.get("RENDER", "").lower() in ("1", "true", "yes")
_DB_PATH = "/tmp/cinerec.db" if _IS_RENDER else str(
    Path(__file__).resolve().parent / "database.db"
)


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-dev-secret")
    JWT_ACCESS_TOKEN_EXPIRES = 86400

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{_DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Artefatti recommender
    DEPLOY_ARTIFACTS = BASE_DIR / "data" / "deploy_artifacts"
    PROCESSED_DIR    = BASE_DIR / "data" / "processed" / "small"

    SVD_PARQUET       = DEPLOY_ARTIFACTS / "svd_matrix.parquet"
    SVD_MATRIX        = PROCESSED_DIR / "baseline_pure_svd" / "predicted_scores_matrix.csv"
    CONTENT_INDEX     = DEPLOY_ARTIFACTS / "movie_embeddings_index_v2.csv"
    CONTENT_NEIGHBORS = DEPLOY_ARTIFACTS / "content_top_neighbors_v2.csv"
    MOVIES_CSV        = DEPLOY_ARTIFACTS / "movies_enriched_tmdb.csv"
    RATINGS_TRAIN     = DEPLOY_ARTIFACTS / "ratings_train.csv"

    # CORS origins
    FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
