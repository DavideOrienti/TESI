from __future__ import annotations
import re
import pandas as pd

from src.utils.io import load_settings

def main():
    s = load_settings()

    movies_path = s.paths.raw / "movies.csv"
    out_path = s.paths.processed / "movies_clean.csv"

    df = pd.read_csv(movies_path)

    # Estrazione anno
    pattern = re.compile(s.clean_year_regex)
    year = df["title"].astype(str).str.extract(pattern)[0]
    df["year"] = pd.to_numeric(year, errors="coerce")
    df["has_year"] = df["year"].notna()
    df["year"] = df["year"].fillna(0).astype(int)

    # Titolo pulito
    df["title_clean"] = df["title"].astype(str).str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()

    # Audit minimale
    missing_year = int((df["year"] == 0).sum())
    print(f"[01] movies rows={len(df)} | missing_year={missing_year}")

    df.to_csv(out_path, index=False)
    print(f"[01] saved -> {out_path}")

if __name__ == "__main__":
    main()