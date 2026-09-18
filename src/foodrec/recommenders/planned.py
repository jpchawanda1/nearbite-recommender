"""Approaches planned for later milestones (interfaces only for now).

Each class documents the intended design so the evaluation harness and the
interface can be wired up before the models are implemented.
"""
from .base import Recommender


class ContentBased(Recommender):
    """Weeks 3-4. Build a customer profile from the cuisines, chains, price
    level and menu keywords (TF-IDF over product names) of vendors they ordered
    from, then rank feasible vendors by cosine similarity to that profile.
    Works for new vendors that have menus but no orders yet."""
    name = "ContentBased"


class ItemKNN(Recommender):
    """Weeks 5-6. Item-based collaborative filtering on the implicit
    customer x vendor order-count matrix (log-scaled, cosine similarity).
    Score a vendor by its similarity to vendors in the customer's history."""
    name = "ItemKNN"


class MatrixFactorization(Recommender):
    """Weeks 8-10. Implicit-feedback matrix factorisation (weighted ALS or
    BPR) on the same matrix; latent factors capture taste beyond cuisine
    labels. Scores are masked by the delivery constraint before ranking."""
    name = "MatrixFactorization"


class Hybrid(Recommender):
    """Weeks 8-10. Weighted blend of MF, content-based and a repeat/recency
    signal, with weights tuned on the validation week. Falls back to
    LocalPopularity for new customers and to content scores for new vendors."""
    name = "Hybrid"
