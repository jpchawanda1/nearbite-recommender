"""Extension module (weeks 11-12): rank riders for a ready order.

DHRD contains no rider data, so riders will be *simulated* around vendor
locations; results from this module are illustrative, not evidence about real
couriers. Each candidate rider is scored with a transparent weighted sum:

    score = w_dist * closeness + w_rating * rating_norm
          + w_load * (1 - load_norm) + w_accept * acceptance_rate

closeness = 1 - distance_to_vendor / max_pickup_km (riders beyond the radius
are ineligible). Weights are exposed in the interface so they can be adjusted
and each ranking can show its per-factor breakdown.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Rider:
    rider_id: str
    latitude: float
    longitude: float
    rating: float           # 1-5
    active_orders: int      # current load
    acceptance_rate: float  # 0-1


def rank_riders(vendor_lat: float, vendor_lon: float, riders: List[Rider],
                k: int = 3, max_pickup_km: float = 3.0, weights=None):
    raise NotImplementedError("Planned for weeks 11-12")
