"""Distance helpers for the delivery-feasibility constraint."""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def feasible_vendors_by_geohash(customer_cells: pd.DataFrame,
                                vendors: pd.DataFrame,
                                max_km: float) -> dict:
    """Map each customer geohash to the set of vendor_ids that can deliver to it.

    customer_cells: columns geohash, latitude, longitude (one row per cell)
    vendors:        columns vendor_id, latitude, longitude
    """
    feasible = {}
    for cell in customer_cells.itertuples(index=False):
        dist = haversine_km(cell.latitude, cell.longitude,
                            vendors["latitude"].values, vendors["longitude"].values)
        feasible[cell.geohash] = set(vendors.loc[dist <= max_km, "vendor_id"])
    return feasible
