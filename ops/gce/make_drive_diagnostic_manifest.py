#!/usr/bin/env python3
"""Write the exact segment-3677 diagnostic manifest without any cloud calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from gcs_drive_diagnostic import build_diagnostic_manifest, load_diagnostic_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", required=True,
                        help="Verified Drive about.user.permissionId for the existing OAuth account")
    parser.add_argument("--output", type=Path, required=True,
                        help="New manifest path; an existing file is never replaced")
    parser.add_argument("--release-id", default=None,
                        help="Optional unique lowercase UUID hex; generated when omitted")
    args = parser.parse_args(argv)
    release_id = args.release_id or uuid.uuid4().hex
    manifest = build_diagnostic_manifest(release_id=release_id, account_id=args.account_id)
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    _, digest = load_diagnostic_manifest(args.output, release_id=release_id,
                                         account_id=args.account_id)
    assert digest == hashlib.sha256(raw).hexdigest()
    print(json.dumps({"release_id": release_id, "manifest_sha256": digest,
                      "manifest_path": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
