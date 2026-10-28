#!/usr/bin/env python3
"""S8's cityHash64 cross-check (1,000 values), with no personal data.

The VM hashes 1,000 deterministic synthetic strings shaped like the pass's keys
(did:plc / did:web DIDs, at:// post URIs, the salted concatenations the sample
predicates use) with its clickhouse-local and writes `string \\t hash`. On the
Mac:

  graph/.venv/bin/python graph/vmpass/crosscheck_hash.py cityhash64_crosscheck.tsv

recomputes every value with the pure-Python CityHash64 and with the Mac's
clickhouse, and exits non-zero on any mismatch or a missing/extra string.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import sys

from cityhash import city_hash64

B32 = "234567abcdefghijklmnopqrstuvwxyz"


def strings(n: int = 1000) -> list[str]:
    rnd = random.Random(20261028)
    out: list[str] = []
    while len(out) < n:
        k = len(out) % 5
        plc = "did:plc:" + "".join(rnd.choice(B32) for _ in range(24))
        if k == 0:
            out.append(plc)
        elif k == 1:
            host = ".".join("".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789-") for _ in range(rnd.randint(1, 20))) for _ in range(rnd.randint(2, 4)))
            out.append("did:web:" + host)
        elif k == 2:
            out.append(f"at://{plc}/app.bsky.feed.post/" + "".join(rnd.choice(B32) for _ in range(13)))
        elif k == 3:
            out.append(plc + rnd.choice(["h2", "h3", "|listsplit", "o12", "xc"]))
        else:
            out.append("".join(rnd.choice(B32 + ":/.-_") for _ in range(rnd.randint(0, 200))))
    return out


def compare_text(tsv: str) -> int:
    """Mismatches between a `string \\t hash` listing and the Python implementation,
    plus strings that should be present and are not."""
    want = set(strings())
    seen = set()
    bad = 0
    for line in tsv.splitlines():
        if not line:
            continue
        s, h = line.rsplit("\t", 1)
        seen.add(s)
        if city_hash64(s) != int(h):
            bad += 1
    return bad + len(want - seen) + len(seen - want)


def main() -> int:
    tsv = open(sys.argv[1]).read()
    py_bad = compare_text(tsv)
    ch = shutil.which("clickhouse")
    ch_bad = None
    if ch:
        vals = "\n".join(strings()) + "\n"
        mac = subprocess.run([ch, "local", "--input-format", "LineAsString", "--structure", "s String", "-q",
                              "SELECT s, cityHash64(s) FROM table FORMAT TabSeparatedRaw"],
                             input=vals.encode(), capture_output=True, check=True).stdout.decode()
        vm = dict(line.rsplit("\t", 1) for line in tsv.splitlines() if line)
        ch_bad = sum(1 for line in mac.splitlines() if line and vm.get(line.rsplit("\t", 1)[0]) != line.rsplit("\t", 1)[1])
    lines = len([x for x in tsv.splitlines() if x])
    print(f"values={lines} python_mismatches={py_bad} mac_clickhouse_mismatches={ch_bad}")
    return 0 if py_bad == 0 and (ch_bad in (0, None)) and lines == 1000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
