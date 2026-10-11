"""Download only the six MIMIC-IV tables this study needs (a fraction of
the full database). Works on Windows without wget.

    python scripts/download_mimic.py --version 3.1 --out D:\\sgmedfusion\\data\\mimiciv\\3.1

You type your OWN PhysioNet username and password at the prompt; they are sent
only to physionet.org and are not stored anywhere.  Interrupted downloads
resume where they stopped.
"""
import argparse
import getpass
import sys
import urllib.request
from pathlib import Path

FILES = ["hosp/patients.csv.gz", "icu/icustays.csv.gz", "icu/d_items.csv.gz",
         "icu/procedureevents.csv.gz", "icu/inputevents.csv.gz", "icu/chartevents.csv.gz"]

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="3.1")
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = "https://physionet.org/files/mimiciv/%s/" % a.version
user = input("PhysioNet username: ").strip()
pw = getpass.getpass("PhysioNet password (not shown): ")
mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
mgr.add_password(None, "https://physionet.org/", user, pw)
opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(mgr))

for rel in FILES:
    dst = Path(a.out) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(base + rel, headers={"Range": "bytes=%d-" % have} if have else {})
    try:
        resp = opener.open(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:                       # already complete
            print("ok (complete)  %s" % rel)
            continue
        sys.exit("HTTP %d for %s - check username/password and that your PhysioNet "
                 "account has signed the MIMIC-IV v%s data use agreement." % (e.code, rel, a.version))
    total = int(resp.headers.get("Content-Length", 0)) + have
    mode = "ab" if resp.status == 206 else "wb"
    done = have if mode == "ab" else 0
    with open(dst, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print("\r%-30s %7.1f / %7.1f MB" % (rel, done / 2 ** 20, total / 2 ** 20), end="", flush=True)
    print()
print("All files in", a.out)
