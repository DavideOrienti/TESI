"""
Test di Wilcoxon (paired, two-sided) tra NDCG@10 di PureSVD e Hybrid SVD+CB
sul test set, per verificare la significatività della differenza osservata.
"""
import json
import pandas as pd
from scipy.stats import wilcoxon

from src.utils.io import load_settings


def main():
    s = load_settings()
    base = s.paths.processed

    svd    = pd.read_csv(base / "baseline_pure_svd" / "test_per_user_metrics.csv")
    hybrid = pd.read_csv(base / "hybrid_svd_content_v1" / "test_per_user_metrics.csv")

    merged = svd[["userId", "NDCG@10"]].merge(
        hybrid[["userId", "NDCG@10"]], on="userId", suffixes=("_svd", "_hybrid")
    )

    stat, p = wilcoxon(merged["NDCG@10_hybrid"], merged["NDCG@10_svd"],
                       zero_method="wilcox", alternative="two-sided")

    result = {
        "n_users":     len(merged),
        "svd_mean":    float(merged["NDCG@10_svd"].mean()),
        "hybrid_mean": float(merged["NDCG@10_hybrid"].mean()),
        "W":           float(stat),
        "p_value":     float(p),
        "significant": bool(p < 0.05)
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
