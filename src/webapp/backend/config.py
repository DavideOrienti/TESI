import os
from pathlib import Path

# Percorso base del progetto
BASE_DIR = Path(os.environ.get(
    "APP_BASE_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent)
))

# Su Render usa PostgreSQL se DATABASE_URL è presente
# In locale usa SQLite
_DATABASE_URL = os.environ.get("DATABASE_URL", "")

if _DATABASE_URL:
    # Render fornisce postgres:// ma SQLAlchemy vuole postgresql://
    if _DATABASE_URL.startswith("postgres://"):
        _DATABASE_URL = _DATABASE_URL.replace("postgres://", "postgresql://", 1)
    DB_URI = _DATABASE_URL
else:
    # Locale: SQLite nella cartella backend
    _IS_RENDER = os.environ.get("RENDER", "")
    if _IS_RENDER:
        DB_URI = "sqlite:////tmp/cinerec.db"
    else:
        _LOCAL_DB = Path(__file__).resolve().parent / "database.db"
        DB_URI = f"sqlite:///{_LOCAL_DB}"


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-dev-secret")
    JWT_ACCESS_TOKEN_EXPIRES = 86400

    SQLALCHEMY_DATABASE_URI = DB_URI
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
