"""Loading, cleaning and temporal splitting of the DHRD Singapore data.

The raw orders file has one row per *order line* (product). For restaurant
recommendation we collapse it to one row per order: customer -> vendor.
"""
import pandas as pd

from . import config

ID_COLS = {"customer_id": str, "vendor_id": str, "product_id": str,
           "geohash": str, "chain_id": str}


def load_vendors() -> pd.DataFrame:
    v = pd.read_csv(config.VENDORS_FILE, dtype=ID_COLS).drop(columns=["Unnamed: 0"])
    # 1,499 vendors have no chain: treat each as its own (independent) chain.
    v["chain_id"] = v["chain_id"].fillna("solo_" + v["vendor_id"])
    return v


def load_products() -> pd.DataFrame:
    p = pd.read_csv(config.PRODUCTS_FILE, dtype=ID_COLS, index_col=0)
    p["name"] = p["name"].fillna("")
    return p


def load_order_lines() -> pd.DataFrame:
    o = pd.read_csv(config.ORDERS_FILE, dtype=ID_COLS).drop(columns=["Unnamed: 0"])
    o["order_day"] = o["order_day"].str.replace(" days", "", regex=False).astype(int)
    o["hour"] = o["order_time"].str.slice(0, 2).astype(int)
    return o


def load_orders(use_cache: bool = True) -> pd.DataFrame:
    """One row per order: order_id, customer_id, vendor_id, geohash, lat/lon, day, time."""
    cache = config.PROCESSED_DIR / "orders.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    lines = load_order_lines()
    orders = (lines.drop(columns=["product_id"])
                   .drop_duplicates("order_id")
                   .sort_values(["order_day", "order_time", "order_id"])
                   .reset_index(drop=True))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_pickle(cache)
    return orders


def temporal_split(orders: pd.DataFrame, stage: str = "test"):
    """Return (history, target) for a leakage-free, time-ordered evaluation.

    stage="valid": history = days 0-61,  target = days 62-68 (model tuning)
    stage="test":  history = days 0-68,  target = days 69-75 (final comparison)
    """
    if stage == "valid":
        hist_end, tgt_end = config.TRAIN_END_DAY, config.VALID_END_DAY
    elif stage == "test":
        hist_end, tgt_end = config.VALID_END_DAY, orders["order_day"].max()
    else:
        raise ValueError("stage must be 'valid' or 'test'")
    history = orders[orders["order_day"] <= hist_end]
    target = orders[(orders["order_day"] > hist_end) & (orders["order_day"] <= tgt_end)]
    return history, target


def customer_cells(orders: pd.DataFrame) -> pd.DataFrame:
    """Unique customer geohash cells with their centre coordinates."""
    return orders.drop_duplicates("geohash")[["geohash", "latitude", "longitude"]]
