import os
from pathlib import Path

# In produzione su Render la CWD è la root del repo.
# In locale saliamo di 4 livelli da backend/.
BASE_DIR = Path(os.environ.get("APP_BASE_DIR",
               str(Path(__file__).resolve().parent.parent.parent.parent)))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-dev-secret")
    JWT_ACCESS_TOKEN_EXPIRES = 86400  # 24 ore

    # Database: in produzione usa /tmp/ (filesystem efimero su Render)
    _db_path = os.environ.get(
        "DATABASE_URL",
        str(BASE_DIR / "src" / "webapp" / "backend" / "database.db"),
    )
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{_db_path}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Artefatti recommender — deploy_artifacts ha la precedenza
    DEPLOY_ARTIFACTS = BASE_DIR / "data" / "deploy_artifacts"
    PROCESSED_DIR    = BASE_DIR / "data" / "processed" / "small"

    SVD_PARQUET       = DEPLOY_ARTIFACTS / "svd_matrix.parquet"
    SVD_MATRIX        = PROCESSED_DIR / "baseline_pure_svd" / "predicted_scores_matrix.csv"
    CONTENT_INDEX     = DEPLOY_ARTIFACTS / "movie_embeddings_index_v2.csv"
    CONTENT_NEIGHBORS = DEPLOY_ARTIFACTS / "content_top_neighbors_v2.csv"
    MOVIES_CSV        = DEPLOY_ARTIFACTS / "movies_enriched_tmdb.csv"
    RATINGS_TRAIN     = DEPLOY_ARTIFACTS / "ratings_train.csv"
