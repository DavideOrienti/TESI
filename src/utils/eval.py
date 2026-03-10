from __future__ import annotations

def hit_rate_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    rec_k = recommended[:k]
    return 1.0 if ground_truth in rec_k else 0.0

def ndcg_at_k_single(recommended: list[int], ground_truth: int, k: int) -> float:
    rec_k = recommended[:k]
    if ground_truth in rec_k:
        rank = rec_k.index(ground_truth) + 1  # rank parte da 1
        from math import log2
        return 1.0 / log2(rank + 1)
    return 0.0

def mrr_at_k_single(recommended: list[int], ground_truth: int, k: int) -> float:
    rec_k = recommended[:k]
    if ground_truth in rec_k:
        rank = rec_k.index(ground_truth) + 1
        return 1.0 / rank
    return 0.0