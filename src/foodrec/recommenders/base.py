"""Common interface for every recommender in the project."""
from typing import List, Optional, Set

import pandas as pd


class Recommender:
    name = "base"

    def fit(self, history: pd.DataFrame, vendors: pd.DataFrame) -> "Recommender":
        """Learn from past orders (one row per order) and vendor metadata."""
        raise NotImplementedError

    def recommend(self, customer_id: str, geohash: str, k: int,
                  feasible: Optional[Set[str]] = None) -> List[str]:
        """Return up to k vendor_ids, best first.

        feasible: vendors that can deliver to `geohash`. When given, every
        returned vendor must be in this set (hard delivery constraint).
        """
        raise NotImplementedError


def top_k_from_scores(scores: pd.Series, k: int, feasible=None, exclude=()) -> List[str]:
    """Pick the k highest-scoring vendors, honouring feasibility and exclusions."""
    out = []
    for vendor in scores.index:
        if vendor in exclude or (feasible is not None and vendor not in feasible):
            continue
        out.append(vendor)
        if len(out) == k:
            break
    return out
