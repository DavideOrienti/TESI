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

import numpy as np
import pandas as pd

from src.recommenders.scoring import (
    minmax_normalize_scores,
    score_candidate_content,
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


def _build_content_neighbors_dict(
    neighbors_df: pd.DataFrame,
    index_to_movieid: dict[int, int],
) -> dict[int, list[tuple[int, float]]]:
    neigh: dict[int, list[tuple[int, float]]] = {}
    for movie_idx, g in neighbors_df.groupby("movie_idx"):
        candidate_movie_id = index_to_movieid.get(int(movie_idx))
        if candidate_movie_id is None:
            continue
        rows = []
        for _, row in g.iterrows():
            neighbor_movie_id = index_to_movieid.get(int(row["neighbor_idx"]))
            if neighbor_movie_id is None:
                continue
            rows.append((neighbor_movie_id, float(row["similarity"])))
        rows.sort(key=lambda x: (-x[1], x[0]))
        neigh[candidate_movie_id] = rows
    return neigh


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
    _content_neighbors_dict = _build_content_neighbors_dict(neighbors_df, _index_to_movieid)

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
    if "popularity" in movies_df.columns:
        pop_sorted = movies_df[movies_df["movieId"].isin(_candidate_items)].sort_values(
            "popularity", ascending=False, na_position="last"
        )
        _popularity_ranking = [int(mid) for mid in pop_sorted["movieId"].tolist()]
    else:
        _popularity_ranking = list(_candidate_items)

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


def _svd_scores_for_user(
    user_id: int,
    unseen_candidates: list[int],
    user_mean_rating: float,
) -> dict[int, float]:
    if _svd_matrix is None or user_id not in _svd_matrix.index:
        # utente nuovo: fallback al valor medio globale (o 0)
        return {mid: user_mean_rating for mid in unseen_candidates}
    user_row = _svd_matrix.loc[user_id]
    svd_cols = set(user_row.index)
    return {
        mid: float(user_row[mid]) if mid in svd_cols else user_mean_rating
        for mid in unseen_candidates
    }


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
    # Costruisce merged_ratings combinando rating espliciti e segnale implicito dei preferiti
    merged_ratings: dict[int, float] = dict(user_ratings)
    for movie_id in favorite_movie_ids:
        if movie_id not in merged_ratings:
            merged_ratings[movie_id] = 4.5
        else:
            merged_ratings[movie_id] = min(5.0, merged_ratings[movie_id] * 1.1)

    seen_items = set(seen_movie_ids)
    user_mean = float(np.mean(list(merged_ratings.values()))) if merged_ratings else 3.5
    unseen = [m for m in _candidate_items if m not in seen_items]

    svd_raw = _svd_scores_for_user(user_id, unseen, user_mean)
    content_raw = {
        mid: score_candidate_content(
            candidate_movie_id=mid,
            seen_ratings=merged_ratings,
            user_mean_rating=user_mean,
            content_neighbors_dict=_content_neighbors_dict,
        )
        for mid in unseen
    }

    svd_norm = minmax_normalize_scores(svd_raw)
    content_norm = minmax_normalize_scores(content_raw)

    is_new_user = _svd_matrix is None or user_id not in _svd_matrix.index

    scored = []
    for mid in unseen:
        s_svd = svd_norm.get(mid, 0.0)
        s_content = content_norm.get(mid, 0.0)
        if is_new_user:
            final = s_content
        else:
            final = GAMMA * s_svd + (1.0 - GAMMA) * s_content
        scored.append((mid, final, s_svd, s_content))

    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:top_k]

    # Fallback: se tutti i punteggi sono 0 (utente senza rating) → popolarità
    if not top or all(s == 0.0 for _, s, _, _ in top):
        fallback = [mid for mid in _popularity_ranking if mid not in seen_items][:top_k]
        return [
            {
                "movie_id": mid,
                "score": 0.0,
                "explanation": {"type": "popular", "svd_score": 0.0, "content_score": 0.0, "seed_movies": []},
            }
            for mid in fallback
        ]

    results = []
    for mid, final, s_svd, s_content in top:
        seeds = _content_explanation(mid, merged_ratings)
        results.append({
            "movie_id": int(mid),
            "score": round(float(final), 6),
            "explanation": {
                "type": "hybrid" if not is_new_user else "content",
                "svd_score": round(float(s_svd), 6),
                "content_score": round(float(s_content), 6),
                "seed_movies": seeds,
            },
        })

    return results


def get_similar_movies(movie_id: int, top_k: int = 10) -> list[dict]:
    """Film più simili per embedding content-based."""
    neighbors = _content_neighbors_dict.get(movie_id, [])
    return [
        {"movie_id": nid, "similarity": round(float(sim), 4)}
        for nid, sim in neighbors[:top_k]
    ]


def get_popular_movies(top_k: int = 20, exclude: set[int] | None = None) -> list[int]:
    """Restituisce i movie_id più popolari, escludendo quelli in exclude."""
    excl = exclude or set()
    return [mid for mid in _popularity_ranking if mid not in excl][:top_k]


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
