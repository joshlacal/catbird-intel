#!/usr/bin/env python3
"""Turn repo_supplement output into fold inputs.

Reads <supplement>/events-*.jsonl through the same row builder as the archive
extractor and writes <events_out>/events-supplement-<host>-NNN.parquet. Copies
<supplement>/snapshots-<host>.jsonl to
<snapshots_out>/repo_snapshots_supplement_<host>.jsonl so the fold's default
repo_snapshots*.jsonl* glob picks them up next to the collector export.

Usage: supplement_to_parquet.py SUPPLEMENT_DIR EVENTS_OUT SNAPSHOTS_OUT
"""
import json
import pathlib
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from archive_graph_extract import SCHEMA, row_from_event, should_keep_event

CHUNK_ROWS = 2_000_000


def convert(supplement: pathlib.Path, events_out: pathlib.Path, snapshots_out: pathlib.Path) -> int:
    events_out.mkdir(parents=True, exist_ok=True)
    snapshots_out.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sorted(supplement.glob("events-*.jsonl")):
        host = src.name[len("events-"):-len(".jsonl")]
        rows, part = [], 0

        def flush():
            nonlocal rows, part
            if rows:
                dst = events_out / f"events-supplement-{host}-{part:03d}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), dst,
                               compression="zstd", compression_level=6)
                part += 1
                rows = []

        with src.open() as fh:
            for line in fh:
                ev = json.loads(line)
                if should_keep_event(ev):
                    rows.append(row_from_event(ev, source_id="repo_supplement"))
                    total += 1
                    if len(rows) >= CHUNK_ROWS:
                        flush()
        flush()
        snaps = supplement / f"snapshots-{host}.jsonl"
        if snaps.exists():
            shutil.copyfile(snaps, snapshots_out / f"repo_snapshots_supplement_{host}.jsonl")
    return total


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    n = convert(*(pathlib.Path(a) for a in sys.argv[1:]))
    print(json.dumps({"rows": n}))
