from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

from src.utils.io import load_settings


def normalize_tag(t: str) -> str:
    t = str(t).strip().lower()
    t = " ".join(t.split())
    t = t.replace(" ", "_")
    return t


def main():
    s = load_settings()

    movies_in = s.paths.processed / "movies_clean.csv"
    tags_path = s.paths.raw / "tags.csv"

    out_movies = s.paths.processed / "movies_with_tags.csv"
    out_tag_counts = s.paths.processed / "movie_tag_counts.csv"
    out_vocab = s.paths.processed / "tag_tfidf_vocabulary.csv"
    out_index = s.paths.processed / "tag_tfidf_index.csv"
    out_matrix = s.paths.processed / "tag_tfidf_matrix.npz"

    movies = pd.read_csv(movies_in)
    tags = pd.read_csv(tags_path)

    # 1) Normalizzazione tag
    tags["tag_norm"] = tags["tag"].map(normalize_tag)

    # Rimuove tag vuoti
    tags = tags[tags["tag_norm"].ne("")].copy()

    raw_rows = len(tags)

    # 2) Conteggio frequenze per (movieId, tag_norm)
    tag_counts = (
        tags.groupby(["movieId", "tag_norm"])
        .size()
        .reset_index(name="tag_count")
        .sort_values(["movieId", "tag_count", "tag_norm"], ascending=[True, False, True])
    )

    unique_pairs = len(tag_counts)
    duplicate_rows = raw_rows - unique_pairs

    # Salva tabella strutturata film-tag-count
    tag_counts.to_csv(out_tag_counts, index=False)

    # 3) Costruzione testo aggregato semplice
    grouped_tags = (
        tag_counts.groupby("movieId")["tag_norm"]
        .apply(lambda x: ", ".join(x.tolist()))
        .reset_index()
        .rename(columns={"tag_norm": "tags"})
    )

    # 4) Costruzione testo pesato con frequenza
    # Ripete ogni tag in base alla frequenza locale, utile per far emergere il TF nella TF-IDF
    weighted_docs = (
        tag_counts.groupby("movieId")
        .apply(
            lambda df: " ".join(
                [tag for tag, count in zip(df["tag_norm"], df["tag_count"]) for _ in range(int(count))]
            )
        )
        .reset_index(name="tags_weighted_text")
    )

    # 5) Merge con movies
    merged = movies.merge(grouped_tags, on="movieId", how="left")
    merged = merged.merge(weighted_docs, on="movieId", how="left")
    merged["tags"] = merged["tags"].fillna("")
    merged["tags_weighted_text"] = merged["tags_weighted_text"].fillna("")

    # 6) TF-IDF sui documenti tag dei film
    tfidf_input = merged["tags_weighted_text"].tolist()

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[^\s]+\b",
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(tfidf_input)

    # 7) Salvataggio matrice sparse
    save_npz(out_matrix, tfidf_matrix)

    # 8) Salvataggio vocabolario
    vocab_df = pd.DataFrame({
        "feature_index": range(len(vectorizer.get_feature_names_out())),
        "tag_term": vectorizer.get_feature_names_out()
    })
    vocab_df.to_csv(out_vocab, index=False)

    # 9) Salvataggio indice righe matrice -> movieId
    index_df = merged[["movieId", "title_clean"]].copy()
    index_df["matrix_row"] = range(len(index_df))
    index_df.to_csv(out_index, index=False)

    # 10) Output film-level finale
    merged.to_csv(out_movies, index=False)

    # 11) Audit
    movies_with_any_tag = int((merged["tags"] != "").sum())
    avg_unique_tags = (
        tag_counts.groupby("movieId")["tag_norm"].count().mean()
        if len(tag_counts) > 0 else 0.0
    )

    print(
        f"[02] movies={len(movies)} | raw_tag_rows={raw_rows} | "
        f"unique_movie_tag_pairs={unique_pairs} | duplicates_removed={duplicate_rows}"
    )
    print(
        f"[02] movies_with_tags={movies_with_any_tag} | "
        f"avg_unique_tags_per_tagged_movie={avg_unique_tags:.2f}"
    )
    print(f"[02] saved movies -> {out_movies}")
    print(f"[02] saved tag counts -> {out_tag_counts}")
    print(f"[02] saved tfidf matrix -> {out_matrix}")
    print(f"[02] saved tfidf vocab -> {out_vocab}")
    print(f"[02] saved tfidf index -> {out_index}")


if __name__ == "__main__":
    main()