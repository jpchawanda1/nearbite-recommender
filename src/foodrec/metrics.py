"""Top-K ranking metrics with binary relevance."""
import math


def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(recs, relevant, k):
    return float(bool(set(recs[:k]) & relevant))


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recs[:k]) if item in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def catalogue_coverage(all_recs, catalogue_size):
    """Share of the catalogue that appears in at least one recommendation list."""
    shown = set()
    for recs in all_recs:
        shown.update(recs)
    return len(shown) / catalogue_size


def intra_list_diversity(recs, item_category):
    """Distinct categories (cuisines) in the list divided by list length."""
    if not recs:
        return 0.0
    return len({item_category.get(i) for i in recs}) / len(recs)
