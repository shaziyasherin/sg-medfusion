"""Download the six tables of the eICU Collaborative Research Database Demo
v2.0.1 that this study uses (about 21 MB).  Open access (ODbL v1.0): no login.

    python scripts/download_eicu.py --out D:\\sgmedfusion\\data\\eicu_demo
"""
import argparse
import time
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/eicu-crd-demo/2.0.1/"
FILES = ["patient", "vitalPeriodic", "respiratoryCare", "respiratoryCharting", "infusiondrug", "treatment"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for name in ["LICENSE.txt"] + [f + ".csv.gz" for f in FILES]:
    dst = out / name
    if dst.exists() and dst.stat().st_size > 0:
        print("already done:", name)
        continue
    for attempt in range(6):
        try:
            print("downloading", name, "...", flush=True)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(BASE + name, tmp)
            tmp.replace(dst)
            print("  saved %s (%.1f MB)" % (name, dst.stat().st_size / 2 ** 20))
            break
        except Exception as e:
            print("  error (%s), retrying in %ds" % (e, 10 * (attempt + 1)))
            time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit("could not download %s - check the internet connection and run again" % name)
print("eICU demo ready in", out)
