#!/bin/bash
# S5 (vii-b) step A: dedupe the per-PDS listRepos output (one row per DID) and find our-data DIDs
# absent from it; writes a deterministic 1,000-DID completeness sample.
set -euo pipefail
W="${1:-$HOME/catbird-lake-local/s5-sweep-20261028/work}"
cd "$W"
CH="clickhouse local --max_memory_usage 16000000000"
SRC="file('pdslist/*.tsv.zst', TSV, 'did String, a String, s String')"
$CH -q "
SELECT did, if(n_active > 0, 'true', 'false') AS active, if(n_active > 0, '', st) AS status, nh, n_active, conflict, host FROM (
  SELECT did, countIf(a = 'true') AS n_active, uniqExact(_file) AS nh,
         anyIf(s, a = 'false' AND s != '') AS st0, if(st0 = '', 'none', st0) AS st,
         toUInt8(uniqExact(a) > 1) AS conflict,
         replaceRegexpOne(if(n_active > 0, anyIf(_file, a = 'true'), any(_file)), '\\.tsv\\.zst$', '') AS host
  FROM $SRC WHERE startsWith(did, 'did:') GROUP BY did)
ORDER BY did INTO OUTFILE 'listing_pds.tsv.zst' TRUNCATE FORMAT TSV"
L2="file('listing_pds.tsv.zst', TSV, 'did String, active String, status String, nh UInt32, n_active UInt32, conflict UInt8, host String')"
$CH -q "SELECT count() AS dids, countIf(nh > 1) AS multi_host, countIf(conflict = 1) AS active_conflict, countIf(n_active > 1) AS active_on_2plus,
  countIf(startsWith(did,'did:web:')) AS did_web FROM $L2 FORMAT TSVWithNames" > listing_pds_counts.tsv
$CH -q "SELECT active, status, count() FROM $L2 GROUP BY ALL ORDER BY 3 DESC FORMAT TSVWithNames" > listing_pds_status_classes.tsv
$CH -q "
SELECT o.did, o.srcs FROM file('ourdata_all.tsv.zst', TSV, 'did String, srcs Array(String)') AS o
LEFT ANTI JOIN $L2 AS l ON l.did = o.did ORDER BY o.did INTO OUTFILE 'ourdata_absent_pds.tsv' TRUNCATE FORMAT TSV"
$CH -q "SELECT arrayJoin(srcs) s, count() FROM file('ourdata_absent_pds.tsv', TSV, 'did String, srcs Array(String)') GROUP BY s ORDER BY s FORMAT TSV" > ourdata_absent_pds_counts.tsv
$CH -q "
SELECT arrayJoin(o.srcs) s, l.active, l.status, count() FROM file('ourdata_all.tsv.zst', TSV, 'did String, srcs Array(String)') AS o
INNER JOIN $L2 AS l ON l.did = o.did GROUP BY ALL ORDER BY 1, 4 DESC FORMAT TSV" > ourdata_listed_pds_classes.tsv
$CH -q "SELECT did FROM file('ourdata_absent_pds.tsv', TSV, 'did String, srcs Array(String)') ORDER BY cityHash64(concat(did, 's5p')) LIMIT 1000 FORMAT TSV" > absent_sample_pds.txt
cat listing_pds_counts.tsv listing_pds_status_classes.tsv; echo absent; cat ourdata_absent_pds_counts.tsv; echo listed; cat ourdata_listed_pds_classes.tsv
