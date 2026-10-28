#!/bin/bash
# S5 (viii) -> plc.tsv (GCE pass) and the M2 inputs for eval_temporal.py.
# Inputs: work/plc/*.tsv (genesis ops), work/listing_pds.tsv.zst (per-PDS listing; survivors = active),
#         work/didweb_first.tsv (did:web first-record TIDs), S2a test_follows.parquet.
set -euo pipefail
W=$HOME/catbird-lake-local/s5-sweep-20261028/work
O=$HOME/catbird-lake-local/s5-sweep-20261028
L=$HOME/catbird-lake-local
cd "$W"
CH="clickhouse local --max_memory_usage 12000000000"
GEN="file('plc/2026*.tsv', TSV, 'did String, created String, nullified UInt8, typ String')"
LST="(SELECT did FROM file('listing_pds.tsv.zst', TSV, 'did String, active String, status String, nh UInt32, n_active UInt32, conflict UInt8, host String') WHERE active = 'true')"

# plc.tsv: one genesis per did:plc created in [Aug 14, Sep 10), non-nullified, active in the Oct 28 per-PDS listing.
$CH -q "
SELECT g.did, min(g.created) AS created FROM $GEN AS g
WHERE g.nullified = 0 AND g.did IN $LST
GROUP BY g.did ORDER BY g.did
INTO OUTFILE '$O/plc.tsv.tmp' TRUNCATE FORMAT TSV"
mv "$O/plc.tsv.tmp" "$O/plc.tsv"

# Cohort: PLC genesis in [2026-08-15, 2026-09-09) UTC (S1: created in [08-15, 09-08]); did:web by
# first-record TID in the same window. Survivors only (active in the Oct 28 per-PDS listing).
$CH -q "
SELECT did, created_us, if(cityHash64(concat(did, 'm2')) % 2 = 1, 'test', 'val') AS split, source FROM (
  SELECT did, toInt64(toUnixTimestamp64Micro(parseDateTime64BestEffort(created, 6, 'UTC'))) AS created_us, 'plc_genesis' AS source
  FROM file('$O/plc.tsv', TSV, 'did String, created String')
  UNION ALL
  SELECT did, toInt64(first_us), 'didweb_first_record' FROM file('didweb_first.tsv', TSV, 'did String, first_us Int64, n_collections UInt32')
  WHERE first_us > 0 AND did IN $LST
)
WHERE created_us >= 1786752000000000 AND created_us < 1788912000000000
ORDER BY did
INTO OUTFILE '$O/m2_cohort.parquet.tmp' TRUNCATE FORMAT Parquet"
mv "$O/m2_cohort.parquet.tmp" "$O/m2_cohort.parquet"
$CH -q "SELECT did, created_us FROM file('$O/m2_cohort.parquet', Parquet) ORDER BY did INTO OUTFILE '$O/m2_cohort.tsv.tmp' TRUNCATE FORMAT TSV"
mv "$O/m2_cohort.tsv.tmp" "$O/m2_cohort.tsv"

# Follows to day 30 from S2a's test follows (TID >= 2026-08-16T00:00Z only; see the coverage note).
$CH -q "
SELECT f.did AS did, f.subject AS subject, toInt64(f.c_us) AS created_us, toUInt16(f.burst_b) AS burst, toUInt8(f.live_snap) AS live
FROM file('$L/s2-prep-20261028/out/s2a/test/test_follows.parquet', Parquet) AS f
INNER JOIN (SELECT did, created_us FROM file('$O/m2_cohort.parquet', Parquet)) AS c ON c.did = f.did
WHERE toInt64(f.c_us) < c.created_us + 30 * 86400000000
ORDER BY did, created_us
INTO OUTFILE '$O/m2_follows.parquet.tmp' TRUNCATE FORMAT Parquet SETTINGS output_format_parquet_compression_method = 'zstd'"
mv "$O/m2_follows.parquet.tmp" "$O/m2_follows.parquet"

# Coverage of day 1 (test follows begin at 2026-08-16T00:00Z = 1786838400000000 us).
$CH -q "
SELECT count() AS cohort, countIf(split='test') AS test, countIf(split='val') AS val,
  countIf(source='didweb_first_record') AS didweb,
  countIf(created_us < 1786838400000000) AS created_aug15,
  countIf(created_us + 86400000000 > 1786838400000000 AND created_us < 1786838400000000) AS day1_partly_before_coverage,
  round(sumIf(least(1786838400000000 - created_us, 86400000000), created_us < 1786838400000000) / 86400000000, 1) AS day1_user_days_uncovered,
  min(created_us), max(created_us)
FROM file('$O/m2_cohort.parquet', Parquet) FORMAT TSVWithNames" > m2_cohort_stats.tsv
$CH -q "
SELECT count() AS follows, uniqExact(f.did) AS users_with_follows,
  countIf(f.created_us < c.created_us + 86400000000) AS day1_follows,
  uniqExactIf(f.did, f.created_us < c.created_us + 86400000000) AS users_with_day1,
  countIf(f.created_us < c.created_us) AS before_creation,
  countIf(f.burst >= 10) AS burst_ge10, countIf(f.live = 1) AS live
FROM file('$O/m2_follows.parquet', Parquet) AS f
INNER JOIN (SELECT did, created_us FROM file('$O/m2_cohort.parquet', Parquet)) AS c ON c.did = f.did FORMAT TSVWithNames" > m2_follows_stats.tsv
cat m2_cohort_stats.tsv m2_follows_stats.tsv
