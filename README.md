# NearBite: Delivery-Aware Restaurant Recommender

**DSA 4060 Recommender Systems, semester project (individual)**
**Student:** Justice Chawanda, 670444. Repository: https://github.com/jpchawanda1/nearbite-recommender

NearBite recommends restaurants to food-delivery customers (Glovo / foodpanda style). For each customer
it returns a ranked top-10 list that (1) only contains restaurants able to deliver to the customer's
location, (2) reflects their order history, and (3) mixes familiar favourites with relevant restaurants
they have not tried. A later extension ranks nearby riders for each order.

Proposal: [docs/NearBite_Proposal.pdf](docs/NearBite_Proposal.pdf) ([.docx](docs/NearBite_Proposal.docx)). Data inspection: [notebooks/01_data_inspection.ipynb](notebooks/01_data_inspection.ipynb)

## Problem

Delivery apps list thousands of restaurants, but customers see only a few on the home screen. Generic
"popular" lists ignore both taste and delivery reach, so customers scroll, reorder the same place, or leave.
Choosing well is hard because the catalogue is huge, order histories are short (41.5% of customers have one
order), and every suggestion must be deliverable to the customer's address.

## Dataset

[Delivery Hero Recommendation Dataset (DHRD)](https://github.com/deliveryhero/dh-reco-dataset), Singapore subset, MIT licence
(Assylbekov et al., RecSys 2023). Data is **not** committed; see [data/README.md](data/README.md).

| | Singapore training file |
|---|---|
| Orders / order lines | 1,709,414 / 3,431,870 |
| Customers | 476,150 |
| Restaurants (vendors) | 7,411 (7,203 with orders), 78 cuisines, 1,854 chains |
| Menu products | 1,066,840 |
| Period | 76 days (days 0-75) |
| Customer-vendor matrix density | 0.037% |

## Approaches

| Stage | Approach | Status |
|---|---|---|
| Baseline | Local popularity within delivery radius (+ global popularity for reference) | implemented |
| Reference | Personal repeat frequency | implemented |
| Approach 1 | Content-based (cuisine, chain, price, TF-IDF of menu names) | weeks 3-4 |
| Approach 2 | Item-based collaborative filtering (implicit feedback) | weeks 5-6 |
| Approach 3 | Matrix factorisation (ALS/BPR) and a hybrid with repeat + content signals | weeks 8-10 |
| Extension | Rider ranking by distance, rating, load, acceptance (simulated riders) | weeks 11-12 |
| Interface | Streamlit app with explanations and preference controls | weeks 11-12 |

Evaluation uses a temporal split: train on days 0-61, tune on days 62-68, test on days 69-75
(retrained on days 0-68). Metrics: Precision/Recall/NDCG/Hit-rate@10, exploration recall
(new restaurants only), catalogue coverage, cuisine diversity and response time.

### Current baseline results (test week, 104,890 customers)

| Model | P@10 | R@10 | NDCG@10 | Coverage |
|---|---|---|---|---|
| Global popularity (no delivery filter) | 0.0001 | 0.0004 | 0.0002 | 0.001 |
| Global popularity (delivery filter) | 0.0004 | 0.0032 | 0.0012 | 0.012 |
| **Local popularity (baseline)** | **0.0063** | **0.0508** | **0.0233** | **0.044** |
| Personal frequency (reference) | 0.0416 | 0.3059 | 0.2437 | 0.934 |

Full table: [results/baseline_metrics.csv](results/baseline_metrics.csv).

## Milestones

| Week | Deliverable |
|---|---|
| 1-2 | Proposal, repository, data inspection, baseline + evaluation harness (done) |
| 3-4 | Data preparation, content-based prototype |
| 5-6 | Item-based CF, preliminary comparison |
| 7 | Mid-semester exam |
| 8-10 | Matrix factorisation and hybrid; tuning on validation week |
| 11-12 | Streamlit interface, rider-ranking extension, final evaluation, documentation |
| 13 | Presentation and demo |

## Repository layout

```
data/                 download instructions (data itself is git-ignored)
docs/                 proposal, figures, dataset samples
notebooks/            01_data_inspection.ipynb
results/              evaluation outputs
scripts/              download_data.py, run_baseline.py
src/foodrec/          data loading, geo constraint, metrics, evaluation, recommenders
tests/                unit tests for metrics
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_baseline.py          # add --sample 5000 for a quick run
pytest -q
```
