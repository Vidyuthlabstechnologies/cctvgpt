#!/usr/bin/env python3
"""Download the UR Fall Detection Dataset (URFD), camera 0 RGB videos + labels.

Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
Citation: B. Kwolek, M. Kepski, "Human fall detection on embedded platform using
depth maps and wireless accelerometer", CMPB 117(3), 2014.

LICENSE: CC BY-NC-SA 4.0, non-commercial academic use. VidyuthLabs uses it only
as an internal evaluation set. It is never redistributed, committed or used to
train a shipped product without written permission from the authors.

    python benchmarks/download_urfd.py --out external_data/urfd
"""

import argparse
import concurrent.futures as cf
import urllib.request
from pathlib import Path

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
FILES = [f"fall-{i:02d}-cam0.mp4" for i in range(1, 31)] + [f"adl-{i:02d}-cam0.mp4" for i in range(1, 41)]
FILES += ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]


def fetch(name: str, out: Path) -> str:
    dest = out / name
    if dest.exists() and dest.stat().st_size > 0:
        return f"cached {name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(BASE + name, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(dest)
    return f"ok {name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="external_data/urfd")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "LICENSE_NOTICE.txt").write_text("URFD - CC BY-NC-SA 4.0, non-commercial academic use only. Do not redistribute.\n")
    with cf.ThreadPoolExecutor(6) as ex:
        for msg in ex.map(lambda n: fetch(n, out), FILES):
            print(msg)


if __name__ == "__main__":
    main()
