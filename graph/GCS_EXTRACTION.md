# Extracting GCE capture segments from Cloud Storage

The GCE collector's sealed segments 3671 and up live in
`gs://catbird-436600-jetstream-buffer-20261020/sources/gce-selfhost-001/buffer-admission-20261020/segments/`.
Segments 0–3670 (plus 3673–3674) were already extracted on the Mac from Drive into
`~/catbird-lake-local/gce-archive-graph/events/`. Together they cover the whole
GCE capture. On 2026-10-22 the bucket held 2,743 contiguous segments (3671–6413,
740 GB).

## Run it inside us-west1

Reading 740 GB out of Google Cloud costs about $0.12/GB (~$89). Reading it from a
VM in us-west1 is free. Run the extraction on a Spot VM in that region and move
only the Parquet output (about 12.6 MB per segment, ~35 GB total) off the VM.

The VM needs: `gcloud` with read access to the bucket, Python 3 with `pyarrow`,
Node 20+, and a copy of
`catbird-community-engine-20261018/tools/community-engine/backfill/` (the decoder
and its `node_modules`).

```bash
python3 graph/archive_graph_extract.py \
  --gcs-prefix gs://catbird-436600-jetstream-buffer-20261020/sources/gce-selfhost-001/buffer-admission-20261020 \
  --min-index 3671 --max-index 4999 \
  --decoder /opt/backfill/decode_segments.mjs \
  --work /data/extract-a --batch 12 --parallel 4 --once
```

`--once` processes one batch. Drop it to keep going until no new segments appear.
Each `--work` directory has its own ledger, so a restart resumes where it stopped.

## Parallelism

The decoder requires increasing sequence numbers within one run, so a single
process works through its range in index order. The fold orders records by repo
rev, not by segment order, so separate processes can take disjoint index ranges
(for example 3671–4999, 5000–5999, 6000–) in separate `--work` directories.
Measured on the Mac: ~8 s to decode one segment (~83k events/s) after a ~12 s
download.

## Collections

The default keeps the graph collections the fold and pack metrics read. For
InterestedIn and Influence, add engagement:
`--collections app.bsky.graph.follow,...,app.bsky.feed.like,app.bsky.feed.repost`.
For likes and reposts, `subject` holds the liked post's AT URI. Engagement output
is far larger than graph output, so use a separate `--work` directory.

## Verification

Every download is checked three ways before decoding: the receipt names the same
segment, the receipt's SHA-256 matches the object name, and the local file's size
and SHA-256 match. A segment with two receipted payloads stops the run instead of
choosing one.
