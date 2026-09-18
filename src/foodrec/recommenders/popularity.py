"""Non-personalised baselines and a repeat-order reference heuristic."""
import pandas as pd

from .base import Recommender, top_k_from_scores


class GlobalPopularity(Recommender):
    """Most-ordered vendors city-wide; the same list for every customer, only
    filtered by the delivery constraint when one is given."""
    name = "GlobalPopularity"

    def fit(self, history, vendors):
        self.scores = history["vendor_id"].value_counts()
        return self

    def recommend(self, customer_id, geohash, k, feasible=None):
        return top_k_from_scores(self.scores, k, feasible)


class LocalPopularity(Recommender):
    """Most-ordered vendors by customers in the same geohash cell, restricted to
    vendors that can deliver there. Ties and unseen vendors fall back to
    city-wide popularity. This is the project's main baseline."""
    name = "LocalPopularity"

    def fit(self, history, vendors):
        self.global_scores = history["vendor_id"].value_counts()
        counts = history.groupby(["geohash", "vendor_id"]).size()
        # small global term breaks ties among equally popular local vendors
        tiebreak = self.global_scores / (self.global_scores.max() + 1)
        self.local_scores = {
            cell: (grp.droplevel(0) + tiebreak.reindex(grp.droplevel(0).index))
                  .sort_values(ascending=False)
            for cell, grp in counts.groupby(level=0)
        }
        return self

    def recommend(self, customer_id, geohash, k, feasible=None, exclude=()):
        local = self.local_scores.get(geohash)
        recs = [] if local is None else top_k_from_scores(local, k, feasible, exclude)
        if len(recs) < k:
            recs += top_k_from_scores(self.global_scores, k - len(recs), feasible,
                                      set(exclude) | set(recs))
        return recs


class PersonalFrequency(Recommender):
    """Reference heuristic: the customer's own most frequently ordered vendors
    (most recent first on ties), padded with LocalPopularity. Captures repeat
    ordering, which is common in food delivery, but never suggests anything new."""
    name = "PersonalFrequency"

    def fit(self, history, vendors):
        stats = (history.groupby(["customer_id", "vendor_id"])
                        .agg(n=("order_id", "size"), last=("order_day", "max"))
                        .reset_index()
                        .sort_values(["customer_id", "n", "last"],
                                     ascending=[True, False, False]))
        self.personal = stats.groupby("customer_id")["vendor_id"].apply(list).to_dict()
        self.fallback = LocalPopularity().fit(history, vendors)
        return self

    def recommend(self, customer_id, geohash, k, feasible=None):
        own = [v for v in self.personal.get(customer_id, [])
               if feasible is None or v in feasible][:k]
        if len(own) < k:
            own += self.fallback.recommend(customer_id, geohash, k - len(own),
                                           feasible, exclude=set(own))
        return own
