import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

TRAIN_FILE = BASE / "ratings_train.csv"
SIM_FILE = BASE / "content_similarity_matrix.npy"
INDEX_FILE = BASE / "movie_embeddings_index_v2.csv"


def load_data():
    train = pd.read_csv(TRAIN_FILE)
    sim_matrix = np.load(SIM_FILE)
    index_df = pd.read_csv(INDEX_FILE)

    return train, sim_matrix, index_df


def build_user_item_matrix(train):
    return train.pivot(index="userId", columns="movieId", values="rating").fillna(0)


def compute_user_similarity(user_item):
    return cosine_similarity(user_item)


def collaborative_score(user_id, movie_id, user_item, user_sim):
    if movie_id not in user_item.columns:
        return 0

    user_idx = user_item.index.get_loc(user_id)
    sims = user_sim[user_idx]

    ratings = user_item[movie_id].values

    return np.dot(sims, ratings) / (np.sum(np.abs(sims)) + 1e-8)


def content_score(movie_id, seen_movies, sim_matrix, movieid_to_index):
    if movie_id not in movieid_to_index:
        return 0

    idx = movieid_to_index[movie_id]

    score = 0
    for seen in seen_movies:
        if seen not in movieid_to_index:
            continue

        seen_idx = movieid_to_index[seen]
        score += sim_matrix[idx, seen_idx]

    return score / (len(seen_movies) + 1e-8)


def recommend(user_id, train, sim_matrix, index_df, alpha=0.5, top_k=10):
    movieid_to_index = {mid: i for i, mid in enumerate(index_df["movieId"])}

    user_item = build_user_item_matrix(train)
    user_sim = compute_user_similarity(user_item)

    seen = train[train.userId == user_id]["movieId"].tolist()
    all_movies = index_df["movieId"].tolist()

    scores = []

    for movie in all_movies:
        if movie in seen:
            continue

        c_score = content_score(movie, seen, sim_matrix, movieid_to_index)
        cf_score = collaborative_score(user_id, movie, user_item, user_sim)

        final = alpha * c_score + (1 - alpha) * cf_score

        scores.append((movie, final))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_k]


if __name__ == "__main__":
    train, sim_matrix, index_df = load_data()

    recs = recommend(
        user_id=1,
        train=train,
        sim_matrix=sim_matrix,
        index_df=index_df
    )

    movie_map = dict(zip(index_df["movieId"], index_df.index))
    titles = pd.read_csv(BASE / "movies_enriched_tmdb.csv")[["movieId", "title_clean"]]
    title_map = dict(zip(titles["movieId"], titles["title_clean"]))

    print("\n🎬 Recommendations for user 1:\n")
    for movie_id, score in recs:
        title = title_map.get(movie_id, "Unknown title")
        print(f"{movie_id} | {title} | score={score:.4f}")