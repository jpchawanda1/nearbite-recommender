"""Download the DHRD Singapore files into data/raw/.

The official source is https://github.com/deliveryhero/dh-reco-dataset
(MIT licence), which links to a Google Drive folder. That folder may ask you to
sign in to Google; if so, download orders_sg_train, vendors_sg and products_sg
from it manually and place them in data/raw/ using the file names below.

This script downloads publicly shared copies of the same Singapore files
(the orders/vendors copies add latitude/longitude decoded from the geohash).
Total download: about 420 MB. The data is NOT committed to this repository.
"""
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "vendors_sg_clean.csv": "1FPYHV73PfNmC3p8AZEYeyoatJy9XUbAE",       # ~0.5 MB
    "products_sg.csv": "13V5LUuQB8-VTZX45Eh_UynuQ0ZatAebf",            # ~65 MB
    "orders_sg_train_clean.csv": "1ag0k8lzUWQ8J-CopiR4fjN_qv60ErLll",  # ~354 MB
}
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, file_id in FILES.items():
        target = RAW / name
        if target.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} ...", flush=True)
        urllib.request.urlretrieve(URL.format(file_id), target)
        print(f"  {target.stat().st_size / 1e6:.1f} MB")
    print("done ->", RAW)


if __name__ == "__main__":
    sys.exit(main())
