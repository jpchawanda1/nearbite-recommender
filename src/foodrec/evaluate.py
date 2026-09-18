"""Offline evaluation harness shared by all recommenders."""
import time

import numpy as np
import pandas as pd

from . import config, metrics
from .data import customer_cells
from .geo import feasible_vendors_by_geohash


def build_eval_cases(history: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """One case per customer active in the target period.

    location  = geohash of the customer's first target order (the delivery
                address is known when the app is opened)
    relevant  = distinct vendors ordered in the target period
    new_relevant = relevant vendors the customer never ordered from before
    warm      = customer has at least one order in the history period
    """
    first = target.drop_duplicates("customer_id")[["customer_id", "geohash"]]
    relevant = target.groupby("customer_id")["vendor_id"].agg(set).rename("relevant")
    seen = history.groupby("customer_id")["vendor_id"].agg(set).rename("seen")
    cases = first.join(relevant, on="customer_id").join(seen, on="customer_id")
    cases["warm"] = cases["seen"].notna()
    cases["seen"] = cases["seen"].apply(lambda s: s if isinstance(s, set) else set())
    cases["new_relevant"] = [r - s for r, s in zip(cases["relevant"], cases["seen"])]
    return cases.reset_index(drop=True)


def evaluate(model, history, target, vendors, k=config.TOP_K,
             constrain=True, sample=None, seed=config.RANDOM_SEED):
    cases = build_eval_cases(history, target)
    if sample is not None and sample < len(cases):
        cases = cases.sample(sample, random_state=seed)
    feasible = feasible_vendors_by_geohash(
        customer_cells(pd.concat([history, target])), vendors, config.MAX_DELIVERY_KM)
    cuisine = dict(zip(vendors["vendor_id"], vendors["primary_cuisine"]))

    rows, all_recs = [], []
    start = time.perf_counter()
    for c in cases.itertuples(index=False):
        recs = model.recommend(c.customer_id, c.geohash, k,
                               feasible.get(c.geohash) if constrain else None)
        all_recs.append(recs)
        explore = c.new_relevant
        rows.append({
            "warm": c.warm,
            "precision": metrics.precision_at_k(recs, c.relevant, k),
            "recall": metrics.recall_at_k(recs, c.relevant, k),
            "ndcg": metrics.ndcg_at_k(recs, c.relevant, k),
            "hit": metrics.hit_rate_at_k(recs, c.relevant, k),
            "explore_recall": metrics.recall_at_k(recs, explore, k) if explore else np.nan,
            "diversity": metrics.intra_list_diversity(recs, cuisine),
            "feasible_ok": all(r in feasible.get(c.geohash, ()) for r in recs),
        })
    elapsed = time.perf_counter() - start
    per_case = pd.DataFrame(rows)

    def summarise(df):
        return {
            f"precision@{k}": df["precision"].mean(),
            f"recall@{k}": df["recall"].mean(),
            f"ndcg@{k}": df["ndcg"].mean(),
            f"hit_rate@{k}": df["hit"].mean(),
            f"explore_recall@{k}": df["explore_recall"].mean(),
            "cuisine_diversity": df["diversity"].mean(),
            "users": len(df),
        }

    result = {"model": model.name, "constrained": constrain, **summarise(per_case)}
    result["warm_users"] = int(per_case["warm"].sum())
    result[f"ndcg@{k}_warm"] = per_case.loc[per_case["warm"], "ndcg"].mean()
    result[f"ndcg@{k}_cold"] = per_case.loc[~per_case["warm"], "ndcg"].mean()
    result["catalogue_coverage"] = metrics.catalogue_coverage(all_recs, len(vendors))
    result["feasibility_rate"] = per_case["feasible_ok"].mean()
    result["ms_per_user"] = 1000 * elapsed / max(len(cases), 1)
    return result
