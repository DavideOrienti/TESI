from __future__ import annotations
import pandas as pd

from src.utils.io import load_settings


class ContentRecommender:
    def __init__(self, movies_file, neighbors_file, index_file):
        self.movies = pd.read_csv(movies_file)
        self.neighbors_df = pd.read_csv(neighbors_file)
        self.index_df = pd.read_csv(index_file)

        # movieId -> movie_idx
        self.movieid_to_index = {
            int(movie_id): int(idx)
            for idx, movie_id in enumerate(self.index_df["movieId"].tolist())
        }

        # movie_idx -> movieId
        self.index_to_movieid = {
            int(idx): int(movie_id)
            for idx, movie_id in enumerate(self.index_df["movieId"].tolist())
        }

        # movieId -> title_clean
        self.movieid_to_title = {
            int(row.movieId): str(row.title_clean)
            for row in self.index_df.itertuples(index=False)
        }

    def search_movies(self, query: str, max_results: int = 10) -> pd.DataFrame:
        matches = self.movies[
            self.movies["title_clean"].fillna("").str.lower().str.contains(
                query.lower(), na=False, regex=False
            )
        ].copy()

        return matches[["movieId", "title_clean"]].head(max_results)

    def recommend_by_movie_id(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        if movie_id not in self.movieid_to_index:
            raise ValueError(f"movieId {movie_id} not found in index.")

        movie_idx = self.movieid_to_index[movie_id]

        neigh = self.neighbors_df[self.neighbors_df["movie_idx"] == movie_idx].copy()

        if neigh.empty:
            return pd.DataFrame(columns=["movieId", "title_clean", "score"])

        neigh = neigh.sort_values("similarity", ascending=False).head(top_k).copy()

        neigh["movieId"] = neigh["neighbor_idx"].map(self.index_to_movieid)
        neigh["title_clean"] = neigh["movieId"].map(self.movieid_to_title)
        neigh["score"] = neigh["similarity"]

        return neigh[["movieId", "title_clean", "score"]].reset_index(drop=True)

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
    s = load_settings()
    base = s.paths.processed

    movies_file = base / "movies_enriched_tmdb.csv"
    neighbors_file = base / "content_top_neighbors_v2.csv"
    index_file = base / "movie_embeddings_index_v2.csv"

    recommender = ContentRecommender(
        movies_file=movies_file,
        neighbors_file=neighbors_file,
        index_file=index_file,
    )

    print(f"[13] dataset={s.dataset}")
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

            if recs.empty:
                print(f"\n⚠️ Nessun vicino trovato per: {selected_title}\n")
                continue

            print(f"\n🎬 Consigli per: {selected_title}\n")
            for row in recs.itertuples(index=False):
                print(f"👉 {row.title_clean}  (score: {row.score:.3f})")
            print()

        except Exception as e:
            print(f"\n❌ Errore: {e}\n")


if __name__ == "__main__":
    main()