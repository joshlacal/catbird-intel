"""Mac side of the S12m image gate: build the D9 bundle the VM applies (print-only upload).

  gate_bundle.py OUT_DIR --labels LABELS_TSV --watermark ISO [--extra-exclude FILE ...] [--upload RUN_ID [--execute]]

Reuses C4's resolver (graph/c4images/gate.py, `load_did_exclusions`) with the plan §5.3 pilot-scope exclusion
OFF (A2 lists the D9 rules only): purge, hide-from-recs, S5 account labels, account labels in force in our
Bluesky labeler replay, Claude adult/likely-minors profiles (Haiku or Sonnet, plus adult_creator), and
KnownFor membership at any rank in an adult/likely-minors community (A2: likely-minors stays excluded for
images). LABELS_TSV is `c4images/labels.py` output after the pre-fetch catch-up tail (`label_tail.sh`), and
--watermark is the time up to which that replay + tail is complete; the VM refuses to fetch when it is older
than --max-label-lag-h (default 36 h).

Only the union of DIDs goes up (no reason codes: which account was flagged adult or likely-minors stays on the
Mac), plus the record-labelled post URIs, to gs://catbird-436600-plan-30d/inputs/<RUN_ID>/gate/ (30-day
lifecycle). Counts by reason are written to OUT_DIR/gate_reasons.json, which stays local.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from datetime import datetime

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mmlib as L  # noqa: E402,F401  (puts c4images on sys.path)
import gate as c4gate  # noqa: E402
from synth import write_bundle  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--watermark", required=True, help="ISO time the label replay + tail is complete up to")
    ap.add_argument("--extra-exclude", action="append", default=[])
    ap.add_argument("--upload")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    wm = datetime.fromisoformat(a.watermark.replace("Z", "+00:00"))
    excl = c4gate.load_did_exclusions(account_labels_tsv=a.labels, scope=False)
    extra = set()
    for f in a.extra_exclude:
        extra |= c4gate._lines(f)
    union = set().union(*excl.sets.values()) | extra
    labelled = c4gate.record_labelled(a.labels)
    out = pathlib.Path(a.out_dir)
    man = write_bundle(out, union, labelled, wm)
    reasons = {"did_sets": excl.sizes(), "extra_exclude": len(extra), "union": len(union), "labelled_uris": len(labelled),
               "inputs_sha256": {p: L.sha256_file(p) for p in [a.labels, *a.extra_exclude]}}
    (out / "gate_reasons.json").write_text(json.dumps(reasons, indent=1))
    print(json.dumps({"manifest": man, "reasons": reasons["did_sets"]}, indent=1))
    if a.upload:
        dst = f"gs://catbird-436600-plan-30d/inputs/{a.upload}/gate/"
        cmd = ["gcloud", "storage", "cp"] + [str(out / n) for n in ("gate_manifest.json", "excluded_dids.txt",
                                                                     "labelled_uris.txt")] + [dst]
        print("+", shlex.join(cmd))
        if a.execute:
            subprocess.run(cmd, check=True)
        else:
            print("# dry run: add --execute to upload")


if __name__ == "__main__":
    main()
