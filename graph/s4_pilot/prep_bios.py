"""S4 producer bios: latest non-empty profile description per producer, minus the bio exclusions.

  prep_bios.py BIOS_RAW_PARQUET BIO_EXCLUDE_TXT PRODUCERS_TSV OUT_PARQUET

BIOS_RAW_PARQUET is claude_labels/extract_profiles.py run over every producers.tsv DID against the
cand1 eval input manifest. The bio text is the `description` field (display names are not embedded).
BIO_EXCLUDE_TXT (from prep_corpus.py) holds user-intents opt-outs and likely-minors-home producers;
hide-from-recs producers keep their bios (S1 amendment A1). Output is keyed by producer pid.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW, EXCL, PRODUCERS, OUT = sys.argv[1:5]
raw = pq.read_table(RAW, columns=["did", "description"]).to_pandas()
excl = {l.strip() for l in open(EXCL) if l.strip()}
prod = pd.read_csv(PRODUCERS, sep="\t", header=None, names=["pid", "did", "followers"], dtype={"did": str})
pid = dict(zip(prod.did, prod.pid))
raw["text"] = raw.description.fillna("").str.strip()
counts = {"producers": len(prod), "profiles_found": len(raw), "nonempty_bios": int((raw.text != "").sum())}
ne = raw[raw.text != ""]
counts["nonempty_bios_excluded"] = int(ne.did.isin(excl).sum())
keep = ne[~ne.did.isin(excl)].copy()
keep["pid"] = keep.did.map(pid).astype(np.int64)
keep = keep.sort_values("pid")[["pid", "did", "text"]].reset_index(drop=True)
counts["bios_to_embed"] = len(keep)
pq.write_table(pa.Table.from_pandas(keep, preserve_index=False), OUT)
counts["out_sha256"] = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
counts["raw_sha256"] = hashlib.sha256(open(RAW, "rb").read()).hexdigest()
json.dump(counts, open(OUT + ".receipt.json", "w"), indent=1)
print(json.dumps(counts, indent=1))
