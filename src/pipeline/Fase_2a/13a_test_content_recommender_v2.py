from pathlib import Path
import numpy as np
import pandas as pd

DATASET = "small"
BASE = Path(f"data/processed/{DATASET}")

MOVIES_FILE = BASE / "movies_enriched_tmdb.csv"
SIM_FILE = BASE / "content_similarity_matrix_v2.npy"
INDEX_FILE = BASE / "movie_embeddings_index_v2.csv"


class ContentRecommender:
    def __init__(self, movies_file: Path, sim_file: Path, index_file: Path):
        self.movies = pd.read_csv(movies_file)
        self.sim_matrix = np.load(sim_file)
        self.index_df = pd.read_csv(index_file)

        self.movieid_to_index = {
            int(mid): int(i) for i, mid in enumerate(self.index_df["movieId"].tolist())
        }

    def search_movies(self, query: str, max_results: int = 10) -> pd.DataFrame:
        matches = self.movies[
            self.movies["title_clean"].str.lower().str.contains(
                query.lower(), na=False, regex=False
            )
        ].copy()

        return matches[["movieId", "title_clean"]].head(max_results)

    def recommend_by_movie_id(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        if movie_id not in self.movieid_to_index:
            raise ValueError(f"movieId {movie_id} not found in index.")

        idx = self.movieid_to_index[movie_id]
        sim_scores = self.sim_matrix[idx]

        top_idx = np.argsort(sim_scores)[::-1][1:top_k + 1]

        recs = self.index_df.iloc[top_idx].copy()
        recs["score"] = sim_scores[top_idx]
        return recs.reset_index(drop=True)

    def recommend_by_title(self, title_query: str, top_k: int = 10) -> tuple[pd.DataFrame, str]:
        matches = self.search_movies(title_query, max_results=10)

        if matches.empty:
            raise ValueError("No movie found.")

        selected = matches.iloc[0]
        movie_id = int(selected["movieId"])
        selected_title = str(selected["title_clean"])

        recs = self.recommend_by_movie_id(movie_id=movie_id, top_k=top_k)
        return recs, selected_title


def main():
    recommender = ContentRecommender(
        movies_file=MOVIES_FILE,
        sim_file=SIM_FILE,
        index_file=INDEX_FILE
    )

    print("🎥 Content Recommender V2")
    print("Scrivi un titolo oppure 'exit'\n")

    while True:
        query = input("Film: ").strip()

        if query.lower() == "exit":
            break

        try:
            matches = recommender.search_movies(query, max_results=5)

            if matches.empty:
                print("\n❌ Nessun film trovato.\n")
                continue

            print("\nPossibili match:")
            for i, row in enumerate(matches.itertuples(index=False), start=1):
                print(f"{i}. {row.title_clean} (movieId={row.movieId})")

            choice = input("\nScegli il numero del film [default=1]: ").strip()
            selected_idx = int(choice) - 1 if choice else 0

            if selected_idx < 0 or selected_idx >= len(matches):
                print("\n❌ Scelta non valida.\n")
                continue

            selected_movie_id = int(matches.iloc[selected_idx]["movieId"])
            selected_title = str(matches.iloc[selected_idx]["title_clean"])

            recs = recommender.recommend_by_movie_id(selected_movie_id, top_k=15)

            print(f"\n🎬 Consigli per: {selected_title}\n")
            for row in recs.itertuples(index=False):
                print(f"👉 {row.title_clean}  (score: {row.score:.3f})")
            print()

        except Exception as e:
            print(f"\n❌ Errore: {e}\n")


if __name__ == "__main__":
    main()