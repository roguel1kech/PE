from typing import List, Set


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Precision@k = |relevant ∩ top_k| / k
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hit = sum(1 for item in top_k if item in relevant)
    return hit / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Recall@k = |relevant ∩ top_k| / |relevant|
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hit = sum(1 for item in top_k if item in relevant)
    return hit / len(relevant)
