#!/bin/bash
# S5 (vii) step A: dedupe the listRepos shards into listing.tsv.zst (did, active, status) and compute
# the "our data" DIDs absent from the listing. Read-only on inputs; writes only under $W.
set -euo pipefail
W=$HOME/catbird-lake-local/s5-sweep-20261028/work
L=$HOME/catbird-lake-local
cd "$W"
CH="clickhouse local --max_memory_usage 12000000000"
# 1. dedupe (shards overlap by at most one page at each boundary)
$CH -q "
SELECT did, any(a) AS active, any(s) AS status, uniqExact((a, s)) AS nstates
FROM file('listrepos/shard*.tsv.zst', TSV, 'did String, a String, s String')
GROUP BY did ORDER BY did
INTO OUTFILE 'listing.tsv.zst' TRUNCATE FORMAT TSV"
$CH -q "SELECT count(), uniqExact(did), countIf(nstates > 1), countIf(startsWith(did,'did:web:')), countIf(startsWith(did,'did:plc:'))
FROM file('listing.tsv.zst', TSV, 'did String, active String, status String, nstates UInt8') FORMAT TSVWithNames" > listing_counts.tsv
$CH -q "SELECT active, status, count() FROM file('listing.tsv.zst', TSV, 'did String, active String, status String, nstates UInt8') GROUP BY ALL ORDER BY 3 DESC FORMAT TSVWithNames" > listing_status_classes.tsv
# 2. "our data" DIDs: producers (full fold, train_T), supplement accounts, S2a test-follow users and subjects
$CH -q "
SELECT did, groupUniqArray(src) AS srcs FROM (
  SELECT did, arrayJoin(JSONExtract(replaceAll(s, '''', '\"'), 'Array(String)')) AS src FROM file('ourdids.tsv', TSV, 'did String, s String')
  UNION ALL SELECT DISTINCT did, 'test_follows_did' FROM file('$L/s2-prep-20261028/out/s2a/test/test_follows.parquet', Parquet)
  UNION ALL SELECT DISTINCT subject, 'test_follows_subject' FROM file('$L/s2-prep-20261028/out/s2a/test/test_follows.parquet', Parquet)
) GROUP BY did ORDER BY did INTO OUTFILE 'ourdata_all.tsv.zst' TRUNCATE FORMAT TSV"
$CH -q "
SELECT o.did, o.srcs FROM file('ourdata_all.tsv.zst', TSV, 'did String, srcs Array(String)') AS o
LEFT ANTI JOIN file('listing.tsv.zst', TSV, 'did String, active String, status String, nstates UInt8') AS l ON l.did = o.did
ORDER BY o.did INTO OUTFILE 'ourdata_absent.tsv' TRUNCATE FORMAT TSV"
$CH -q "SELECT arrayJoin(srcs) s, count() FROM file('ourdata_all.tsv.zst', TSV, 'did String, srcs Array(String)') GROUP BY s ORDER BY s FORMAT TSV" > ourdata_all_counts.tsv
$CH -q "SELECT arrayJoin(srcs) s, count() FROM file('ourdata_absent.tsv', TSV, 'did String, srcs Array(String)') GROUP BY s ORDER BY s FORMAT TSV" > ourdata_absent_counts.tsv
# 3. deterministic 1,000-DID sample of the absent set for getRepoStatus
$CH -q "SELECT did FROM file('ourdata_absent.tsv', TSV, 'did String, srcs Array(String)') ORDER BY cityHash64(concat(did, 's5p')) LIMIT 1000 FORMAT TSV" > absent_sample.txt
cat listing_counts.tsv listing_status_classes.tsv; echo all; cat ourdata_all_counts.tsv; echo absent; cat ourdata_absent_counts.tsv
