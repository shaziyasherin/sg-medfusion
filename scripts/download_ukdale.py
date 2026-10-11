"""Download ONLY the five whole-house mains files the study needs
(house_1..5/channel_1.dat) out of the 3.3 GB UK-DALE ukdale.zip, using HTTP
range requests.  No login needed (UK-DALE is CC BY 4.0).

    python scripts/download_ukdale.py --out D:\\sgmedfusion\\data\\ukdale
    python scripts/download_ukdale.py --list      # just show what is in the zip
"""
import argparse
import io
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/"
       "UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip")


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with retries."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (self.pos, end)})
                data = urllib.request.urlopen(req, timeout=120).read()
                self.pos += len(data)
                return data
            except Exception as e:                      # flaky network: wait and retry
                print("  network error (%s), retrying in %ds" % (e, 5 * (attempt + 1)), flush=True)
                time.sleep(5 * (attempt + 1))
        raise IOError("download failed repeatedly")

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    zf = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=4 << 20))
    wanted = [i for i in zf.infolist()
              if i.filename.replace("\\", "/").split("/")[-1] == "channel_1.dat"
              and "house_" in i.filename]
    if a.list:
        for i in wanted:
            print("%-45s %8.1f MB compressed, %8.1f MB unzipped" % (i.filename, i.compress_size / 2 ** 20, i.file_size / 2 ** 20))
        print("total download: %.1f MB" % (sum(i.compress_size for i in wanted) / 2 ** 20))
        return
    if not a.out:
        sys.exit("give --out D:\\sgmedfusion\\data\\ukdale")
    for i in wanted:
        house = [p for p in i.filename.replace("\\", "/").split("/") if p.startswith("house_")][0]
        dst = Path(a.out) / house / "channel_1.dat"
        if dst.exists() and dst.stat().st_size == i.file_size:
            print("already done:", dst)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("downloading %s (%.0f MB) ..." % (i.filename, i.compress_size / 2 ** 20), flush=True)
        tmp = dst.with_suffix(".part")
        with zf.open(i) as src, open(tmp, "wb") as f:
            shutil.copyfileobj(src, f, 8 << 20)
        tmp.replace(dst)
        print("  saved", dst)
    print("UK-DALE ready in", a.out)


if __name__ == "__main__":
    main()
