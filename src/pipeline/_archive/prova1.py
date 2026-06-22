"""
28_coverage_novelty.py
Calcola Coverage@10 e Novelty@10 per tutti i modelli.
Basato su 17_hybrid_svd_content.py — usa gli stessi artefatti.

Esecuzione: python prova.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.io import load_settings
from src.recommenders.collaborative import get_svd_scores_for_user
from src.recommenders.content_based import (
    build_content_neighbors_dict,
    build_index_to_movieid,
    score_content_candidates,
)
from src.recommenders.hybrid import rank_hybrid_scores
from src.recommenders.scoring import build_user_seen_ratings, build_user_mean_ratings

BEST_GAMMA   = 0.7
K            = 10
TRAIN_NAME   = "ratings_train.csv"
SVD_MATRIX   = "baseline_pure_svd/predicted_scores_matrix.csv"
CONTENT_IDX  = "movie_embeddings_index_v2.csv"
CONTENT_NB   = "content_top_neighbors_v2.csv"

def load_all(base):
    train = pd.read_csv(base / TRAIN_NAME)
    svd   = pd.read_csv(base / SVD_MATRIX, index_col=0)
    svd.columns = [int(c) for c in svd.columns]
    svd.index   = [int(i) for i in svd.index]
    cidx  = pd.read_csv(base / CONTENT_IDX)
    cnb   = pd.read_csv(base / CONTENT_NB)
    return train, svd, cidx, cnb

def get_popularity_recs(user_id, candidate_items, seen_items, train, k=10):
    pop = train.groupby("movieId")["rating"].count()
    unseen = [m for m in candidate_items if m not in seen_items]
    scores = {m: pop.get(m, 0) for m in unseen}
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return top

def get_svd_recs(user_id, candidate_items, seen_items, svd_matrix, user_mean, k=10):
    unseen = [m for m in candidate_items if m not in seen_items]
    scores = get_svd_scores_for_user(user_id, unseen, svd_matrix, user_mean)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return top

def get_cb_recs(user_id, candidate_items, seen_items, seen_ratings,
                user_mean, content_nb_dict, k=10):
    unseen = [m for m in candidate_items if m not in seen_items]
    scores = score_content_candidates(unseen, seen_ratings, user_mean,
                                      content_nb_dict, exclude_seen=False)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return top

def get_hybrid_recs(user_id, candidate_items, seen_items, seen_ratings,
                    user_mean, svd_matrix, content_nb_dict, gamma=0.7, k=10):
    unseen = [m for m in candidate_items if m not in seen_items]
    svd_s  = get_svd_scores_for_user(user_id, unseen, svd_matrix, user_mean)
    cb_s   = score_content_candidates(unseen, seen_ratings, user_mean,
                                      content_nb_dict, exclude_seen=False)
    recs = [r.movie_id for r in rank_hybrid_scores(svd_s, cb_s, gamma, k)]
    return recs

def coverage_novelty(all_recs, catalog_size, item_pop):
    recommended = set()
    novelty_scores = []
    for recs in all_recs:
        recommended.update(recs[:K])
        for item in recs[:K]:
            p = item_pop.get(item, 1e-9)
            novelty_scores.append(-np.log2(p))
    cov = len(recommended) / catalog_size
    nov = float(np.mean(novelty_scores)) if novelty_scores else 0.0
    return round(cov * 100, 2), round(nov, 4), len(recommended)

def main():
    s    = load_settings()
    base = s.paths.processed
    print(f"[28] dataset={s.dataset}")

    train, svd_matrix, cidx, cnb = load_all(base)
    i2m          = build_index_to_movieid(cidx)
    nb_dict      = build_content_neighbors_dict(cnb, i2m)
    seen_map     = build_user_seen_ratings(train)
    mean_map     = build_user_mean_ratings(train)
    catalog_size = train["movieId"].nunique()
    item_pop     = (train.groupby("movieId").size() / len(seen_map)).to_dict()

    candidates = sorted(
        set(train["movieId"].unique()) & set(cidx["movieId"].unique())
    )
    users = sorted(train["userId"].unique())
    print(f"Utenti: {len(users)} | Catalogo: {catalog_size} | Candidati: {len(candidates)}")

    models = {
        "Popularity":    [],
        "PureSVD":       [],
        "Content-Based": [],
        "Hybrid SVD+CB": [],
    }

    print("Calcolo raccomandazioni per tutti gli utenti...")
    for i, uid in enumerate(users):
        if i % 100 == 0:
            print(f"  {i}/{len(users)}")
        uid         = int(uid)
        seen_r      = seen_map.get(uid, {})
        seen_items  = set(seen_r.keys())
        mean_r      = mean_map.get(uid, 0.0)

        models["Popularity"].append(
            get_popularity_recs(uid, candidates, seen_items, train))
        models["PureSVD"].append(
            get_svd_recs(uid, candidates, seen_items, svd_matrix, mean_r))
        models["Content-Based"].append(
            get_cb_recs(uid, candidates, seen_items, seen_r, mean_r, nb_dict))
        models["Hybrid SVD+CB"].append(
            get_hybrid_recs(uid, candidates, seen_items, seen_r,
                            mean_r, svd_matrix, nb_dict, BEST_GAMMA))

    print("\n" + "="*55)
    print(f"{'Modello':<20} {'Coverage@10':>12} {'N item':>8} {'Novelty@10':>12}")
    print("="*55)
    for name, all_recs in models.items():
        cov, nov, n_items = coverage_novelty(all_recs, catalog_size, item_pop)
        print(f"{name:<20} {cov:>11.2f}% {n_items:>8} {nov:>12.4f}")
    print("="*55)

if __name__ == "__main__":
    main()