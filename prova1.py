import numpy as np
import pandas as pd
import random

# =============================
# LOAD DATA
# =============================
movies = pd.read_csv("data/raw/ml-latest-small/movies.csv")
similarity = np.load("data/processed/small/content_similarity_matrix.npy")

# =============================
# RECOMMENDER BY INDEX
# =============================
def recommend_by_index(idx, top_n=15):
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    return [(i, movies.iloc[i]["title"], score) for i, score in scores]

# =============================
# RECOMMENDER BY TITLE
# =============================
def recommend(title, top_n=15, verbose=True):
    matches = movies[
        movies["title"].str.lower().str.contains(title.lower(), na=False, regex=False)
    ]

    if matches.empty:
        if verbose:
            print("\n❌ Film non trovato")
        return []

    idx = matches.index[0]
    results = recommend_by_index(idx, top_n=top_n)

    if verbose:
        print(f"\n🎬 Consigli per: {movies.iloc[idx]['title']}\n")
        for _, film, score in results:
            print(f"👉 {film}  (score: {score:.3f})")

    return results

# =============================
# EVALUATION FUNCTION
# =============================
def evaluate_sample(n_tests=20):
    correct = 0

    for _ in range(n_tests):
        idx = random.randint(0, len(movies) - 1)
        genre = movies.iloc[idx]["genres"]

        results = recommend_by_index(idx, top_n=5)

        found_same_genre = False
        for rec_idx, _, _ in results:
            rec_genre = movies.iloc[rec_idx]["genres"]

            if any(g in rec_genre.split("|") for g in genre.split("|")):
                found_same_genre = True
                break

        if found_same_genre:
            correct += 1

    print(f"\n📊 Accuracy (grezza): {correct / n_tests:.2f}")

# =============================
# INTERACTIVE LOOP
# =============================
if __name__ == "__main__":
    print("🎥 Movie Recommender Test")
    print("Scrivi un titolo (es: Toy Story) oppure 'exit'\n")

    evaluate_sample(50)

    while True:
        title = input("Film: ").strip()

        if title.lower() == "exit":
            break

        recommend(title)