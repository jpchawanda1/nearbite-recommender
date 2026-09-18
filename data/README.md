# Data

Raw and processed data are **not** stored in this repository.

```bash
python scripts/download_data.py   # ~420 MB into data/raw/
```

| File | Content |
|---|---|
| `orders_sg_train_clean.csv` | one row per ordered product: customer, geohash, order, vendor, product, day/time |
| `vendors_sg_clean.csv` | vendor_id, chain_id, geohash, primary_cuisine, latitude, longitude |
| `products_sg.csv` | vendor_id, product_id, name, unit_price (scaled) |

Source: Delivery Hero Recommendation Dataset (DHRD), <https://github.com/deliveryhero/dh-reco-dataset>, MIT licence.
Customer IDs are pseudonymous hashes; locations are ~4.9 km geohash cells. `data/processed/` holds caches built by the code.
