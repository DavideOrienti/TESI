"""
Carica gli artefatti del recommender una sola volta all'import.
Espone get_recommendations() e get_similar_movies().
"""
from __future__ import annotations
import sys
from pathlib import Path

# Assicura che il root del progetto sia nel sys.path per importare src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.recommenders.content_based import (
    build_content_neighbors_dict,
    get_similar_movies as get_content_similar_movies,
    score_content_candidates,
)
from src.recommenders.collaborative import get_svd_scores_for_user
from src.recommenders.hybrid import (
    merge_explicit_and_implicit_feedback,
    rank_hybrid_scores,
    user_mean_or_default,
)
from src.recommenders.popularity import (
    ranking_from_movies_metadata,
    recommend_popular,
)

# ---------------------------------------------------------------------------
# Caricamento artefatti — eseguito UNA SOLA VOLTA all'import
# ---------------------------------------------------------------------------

_initialized = False
_svd_matrix: pd.DataFrame | None = None
_content_neighbors_dict: dict[int, list[tuple[int, float]]] = {}
_index_to_movieid: dict[int, int] = {}
_candidate_items: list[int] = []
_movie_titles: dict[int, str] = {}  # movie_id -> title_clean
_popularity_ranking: list[int] = []  # movie_ids ordinati per popolarità


def init(config) -> None:
    """Chiamato da create_app() con l'oggetto Config."""
    global _initialized, _svd_matrix, _content_neighbors_dict
    global _index_to_movieid, _candidate_items, _movie_titles, _popularity_ranking

    if _initialized:
        return

    print("[recommender_service] loading artefacts...")

    # Flask config è un dict — accesso via [] oppure .get()
    def _cfg(key):
        return config[key] if hasattr(config, "__getitem__") else getattr(config, key)

    # SVD matrix: userId (index) × movieId (columns)
    # Prefer parquet (float16, ~6 MB) over CSV (~33 MB) when available
    svd_path = Path(_cfg("SVD_MATRIX"))
    # Config may expose SVD_PARQUET directly; otherwise fall back to _PROJECT_ROOT heuristic
    parquet_path = Path(_cfg("SVD_PARQUET")) if "SVD_PARQUET" in config else (
        _PROJECT_ROOT / "data" / "deploy_artifacts" / "svd_matrix.parquet"
    )
    if parquet_path.exists():
        svd_raw = pd.read_parquet(parquet_path)
        svd_raw.columns = [int(c) for c in svd_raw.columns]
        svd_raw.index = [int(i) for i in svd_raw.index]
        print(f"[recommender_service] loaded SVD from parquet ({parquet_path.stat().st_size / 1_048_576:.1f} MB)")
    else:
        svd_raw = pd.read_csv(svd_path, index_col=0)
        svd_raw.columns = [int(c) for c in svd_raw.columns]
        svd_raw.index = [int(i) for i in svd_raw.index]
    _svd_matrix = svd_raw

    # Content index: row position → movieId
    index_df = pd.read_csv(_cfg("CONTENT_INDEX"))
    _index_to_movieid = {
        int(idx): int(mid)
        for idx, mid in enumerate(index_df["movieId"].tolist())
    }

    # Content neighbors
    neighbors_df = pd.read_csv(_cfg("CONTENT_NEIGHBORS"))
    _content_neighbors_dict = build_content_neighbors_dict(neighbors_df, _index_to_movieid)

    # Candidate items: film presenti sia in training che nell'indice content
    train_df = pd.read_csv(_cfg("RATINGS_TRAIN"))
    train_movie_ids = set(train_df["movieId"].unique())
    content_movie_ids = set(index_df["movieId"].unique())
    _candidate_items = sorted(int(x) for x in train_movie_ids.intersection(content_movie_ids))

    # Titoli e popolarità per le spiegazioni
    movies_df = pd.read_csv(_cfg("MOVIES_CSV"))
    _movie_titles = {
        int(row["movieId"]): str(row["title_clean"]) if pd.notna(row.get("title_clean")) else str(row["title"])
        for _, row in movies_df.iterrows()
    }

    # Ranking popolarità (fallback utenti nuovi)
    _popularity_ranking = ranking_from_movies_metadata(movies_df, _candidate_items)

    _initialized = True
    print(
        f"[recommender_service] ready — "
        f"svd={_svd_matrix.shape}, "
        f"candidates={len(_candidate_items)}, "
        f"content_neighbors={len(_content_neighbors_dict)}"
    )


# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

# GAMMA = 0.7 selezionato tramite grid search su validation set
# Grid: [0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
# Criterio: NDCG@10 su validation set (il test set NON è stato usato per la selezione)
# Risultati: gamma=0.7 → NDCG@10_val=0.0316 (migliore)
# Ref: Tabella 4.3 della tesi, script Fase_3/17_hybrid_svd_content_eval.py
GAMMA = 0.7


def _content_explanation(
    movie_id: int,
    seen_ratings: dict[int, float],
) -> list[dict]:
    """Top-3 vicini content che l'utente ha valutato positivamente (>= 3.5)."""
    neighbors = _content_neighbors_dict.get(movie_id, [])
    seeds = []
    for neighbor_id, sim in neighbors:
        if neighbor_id in seen_ratings and seen_ratings[neighbor_id] >= 3.5:
            seeds.append({
                "movie_id": neighbor_id,
                "title": _movie_titles.get(neighbor_id, ""),
                "similarity": round(float(sim), 4),
            })
        if len(seeds) == 3:
            break
    return seeds


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def get_recommendations(
    user_id: int,
    seen_movie_ids: list[int],
    user_ratings: dict[int, float],
    favorite_movie_ids: list[int] = [],
    top_k: int = 20,
) -> list[dict]:
    """
    Hybrid PureSVD + Content-Based con gamma=0.7.
    I preferiti senza rating esplicito vengono trattati come rating implicito 4.5;
    i preferiti con rating vengono rinforzati del 10% (max 5.0).
    """
    merged_ratings = merge_explicit_and_implicit_feedback(user_ratings, favorite_movie_ids)
    seen_items = set(seen_movie_ids)
    user_mean = user_mean_or_default(merged_ratings)
    unseen = [m for m in _candidate_items if m not in seen_items]

    svd_raw = get_svd_scores_for_user(user_id, unseen, _svd_matrix, user_mean)
    content_raw = score_content_candidates(
        candidate_items=unseen,
        seen_ratings=merged_ratings,
        user_mean_rating=user_mean,
        content_neighbors_dict=_content_neighbors_dict,
        exclude_seen=False,
    )

    is_new_user = _svd_matrix is None or user_id not in _svd_matrix.index

    top = rank_hybrid_scores(
        collaborative_scores=svd_raw,
        content_scores=content_raw,
        collaborative_weight=GAMMA,
        top_k=top_k,
        content_only=is_new_user,
    )

    # Fallback: se tutti i punteggi sono 0 (utente senza rating) → popolarità
    if not top or all(item.score == 0.0 for item in top):
        fallback = recommend_popular(_popularity_ranking, seen_items, top_k)
        return [
            {
                "movie_id": mid,
                "score": 0.0,
                "explanation": {"type": "popular", "svd_score": 0.0, "content_score": 0.0, "seed_movies": []},
            }
            for mid in fallback
        ]

    results = []
    for item in top:
        mid = item.movie_id
        seeds = _content_explanation(mid, merged_ratings)
        results.append({
            "movie_id": int(mid),
            "score": round(float(item.score), 6),
            "explanation": {
                "type": "hybrid" if not is_new_user else "content",
                "svd_score": round(float(item.collaborative_score), 6),
                "content_score": round(float(item.content_score), 6),
                "seed_movies": seeds,
            },
        })

    return results


def get_similar_movies(movie_id: int, top_k: int = 10) -> list[dict]:
    """Film più simili per embedding content-based."""
    return get_content_similar_movies(movie_id, _content_neighbors_dict, top_k)


def get_popular_movies(top_k: int = 20, exclude: set[int] | None = None) -> list[int]:
    """Restituisce i movie_id più popolari, escludendo quelli in exclude."""
    return recommend_popular(_popularity_ranking, exclude, top_k)


def get_social_recommendations(
    current_user_id: int,
    current_user_ratings: dict[int, float],
    all_users_ratings: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    User-Based CF: raccomandazioni basate su utenti simili.
    Similarità coseno sui film in comune (minimo 2 film).
    Prende i top-5 vicini e calcola media pesata dei loro rating.
    """
    import math

    if len(current_user_ratings) < 2:
        return []

    # --- Step 1: calcola similarità coseno con ogni altro utente ---
    similarities: list[tuple[int, float]] = []

    for other in all_users_ratings:
        other_id = other["user_id"]
        other_ratings = other["ratings"]

        common = [mid for mid in current_user_ratings if mid in other_ratings]
        if len(common) < 2:
            continue

        a = [current_user_ratings[mid] for mid in common]
        b = [other_ratings[mid] for mid in common]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        sim = dot / (norm_a * norm_b + 1e-9)

        if sim > 0:
            similarities.append((other_id, sim))

    if len(similarities) < 1:
        return []

    # --- Step 2: top-5 vicini per similarità ---
    similarities.sort(key=lambda x: -x[1])
    neighbors = similarities[:5]

    # --- Step 3: score per ogni film non visto dall'utente corrente ---
    seen_items = set(current_user_ratings.keys())
    candidate_scores: dict[int, list[tuple[float, float]]] = {}  # movie_id -> [(rating, sim), ...]

    for neighbor_id, sim in neighbors:
        neighbor_ratings = next(
            (u["ratings"] for u in all_users_ratings if u["user_id"] == neighbor_id), {}
        )
        for movie_id, rating in neighbor_ratings.items():
            if movie_id in seen_items:
                continue
            if movie_id not in candidate_scores:
                candidate_scores[movie_id] = []
            candidate_scores[movie_id].append((rating, sim))

    if not candidate_scores:
        return []

    # Media pesata per similarità
    scored: list[tuple[int, float, float, int]] = []
    for movie_id, entries in candidate_scores.items():
        total_weight = sum(sim for _, sim in entries)
        weighted_avg = sum(r * sim for r, sim in entries) / (total_weight + 1e-9)
        scored.append((movie_id, weighted_avg, weighted_avg, len(entries)))

    scored.sort(key=lambda x: (-x[1], x[0]))

    return [
        {
            "movie_id": int(movie_id),
            "score": round(float(score), 6),
            "explanation": {
                "type": "social",
                "similar_users_count": count,
                "avg_rating_similar_users": round(float(avg), 2),
            },
        }
        for movie_id, score, avg, count in scored[:top_k]
    ]
