import numpy as np
import pandas as pd
import random

# =============================
# LOAD DATA
# =============================
movies = pd.read_csv("data/raw/ml-latest-small/movies.csv")
similarity = np.load("data/processed/small/content_similarity_matrix.npy")

# =============================
# RECOMMENDER FUNCTION
# =============================
def recommend(title, top_n=15, verbose=True):
    matches = movies[movies["title"].str.lower().str.contains(title.lower(), na=False)]

    if matches.empty:
        if verbose:
            print("\n❌ Film non trovato")
        return []

    idx = matches.index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    results = [(movies.iloc[i]["title"], score) for i, score in scores]

    if verbose:
        print(f"\n🎬 Consigli per: {movies.iloc[idx]['title']}\n")
        for film, score in results:
            print(f"👉 {film}  (score: {score:.3f})")

    return results

# =============================
# EVALUATION FUNCTION
# =============================
def evaluate_sample(n_tests=20):
    correct = 0
    valid_tests = 0

    for _ in range(n_tests):
        idx = random.randint(0, len(movies) - 1)
        title = movies.iloc[idx]["title"]
        genre = movies.iloc[idx]["genres"]

        results = recommend(title, top_n=5, verbose=False)

        if not results:
            continue

        valid_tests += 1

        for film, _ in results:
            rec_genre = movies[movies["title"] == film]["genres"].values[0]

            if any(g in rec_genre for g in genre.split("|")):
                correct += 1
                break

    if valid_tests == 0:
        print("\n📊 Nessun test valido eseguito.")
    else:
        print(f"\n📊 Accuracy (grezza): {correct / valid_tests:.2f}")

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