"""Project-wide paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

ORDERS_FILE = RAW_DIR / "orders_sg_train_clean.csv"
VENDORS_FILE = RAW_DIR / "vendors_sg_clean.csv"
PRODUCTS_FILE = RAW_DIR / "products_sg.csv"

# Temporal split on `order_day` (days 0-75 in the Singapore training file).
TRAIN_END_DAY = 61   # days 0-61  -> training (development)
VALID_END_DAY = 68   # days 62-68 -> validation (tuning); days 69-75 -> test

# Delivery constraint: maximum distance between the centres of the customer's
# and the restaurant's geohash cells. 5-character geohash cells are ~4.9 km
# wide, so 7 km admits the same cell plus its adjacent and diagonal neighbours
# (covers ~99% of historical orders).
MAX_DELIVERY_KM = 7.0

TOP_K = 10
RANDOM_SEED = 42
