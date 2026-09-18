"""Evaluate the baselines on the validation and test weeks.

Usage: python scripts/run_baseline.py [--sample N]
Writes results/baseline_metrics.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodrec import config  # noqa: E402
from foodrec.data import load_orders, load_vendors, temporal_split  # noqa: E402
from foodrec.evaluate import evaluate  # noqa: E402
from foodrec.recommenders import GlobalPopularity, LocalPopularity, PersonalFrequency  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="evaluate on a random sample of customers")
    args = parser.parse_args()

    orders, vendors = load_orders(), load_vendors()
    results = []
    for stage in ("valid", "test"):
        history, target = temporal_split(orders, stage)
        print(f"[{stage}] history orders={len(history):,}  target orders={len(target):,}")
        runs = [(GlobalPopularity(), False), (GlobalPopularity(), True),
                (LocalPopularity(), True), (PersonalFrequency(), True)]
        for model, constrain in runs:
            model.fit(history, vendors)
            res = evaluate(model, history, target, vendors,
                           constrain=constrain, sample=args.sample)
            res["stage"] = stage
            results.append(res)
            print(f"  {model.name:<18} constrained={constrain!s:<5} "
                  f"P@10={res['precision@10']:.4f} R@10={res['recall@10']:.4f} "
                  f"NDCG@10={res['ndcg@10']:.4f} cov={res['catalogue_coverage']:.3f}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(results)
    out.to_csv(config.RESULTS_DIR / "baseline_metrics.csv", index=False, float_format="%.4f")
    print(f"saved {config.RESULTS_DIR / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
