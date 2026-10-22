"""Read sealed Jetstream segments from the GCE capture's Cloud Storage buffer.

The buffer stores each segment as
  <prefix>/segments/<group>/<seg_name>.jss/<sha256>.jss
  <prefix>/segments/<group>/<seg_name>.jss/<sha256>.jss.g<generation>.receipt.json
(plus a .local-release.json marker that is ignored). The object name carries the
payload SHA-256 and the receipt name carries the payload generation, so a pair is
identified exactly; a segment with more than one receipted payload is refused.

Downloads land in the decoder's layout: <dest>/<seg_name>.jss and
<dest>/<seg_name>.jss.receipt.json. Run this next to the bucket (us-west1);
reading it from outside Google Cloud is billed egress.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess

_OBJECT_RE = re.compile(
    r"/segments/(?P<group>[0-9a-f]{8,})/(?P<seg>seg_[0-9a-z]{10}\.jss)/"
    r"(?P<sha>[0-9a-f]{64})\.jss(?:\.g(?P<gen>[0-9]+)\.receipt\.json)?$"
)


def _gcloud(*args: str) -> str:
    r = subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcloud storage {args[0]} failed: {(r.stderr or r.stdout)[-300:]}")
    return r.stdout


def parse_listing(listing: str) -> dict[int, dict]:
    """Map segment index -> segment dict from `gcloud storage ls -l` output."""
    payloads: dict[tuple[str, str, str], dict] = {}
    receipts: dict[tuple[str, str, str], list[str]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[2].startswith("gs://"):
            continue
        size, uri = int(parts[0]), parts[2]
        m = _OBJECT_RE.search(uri)
        if not m:
            continue
        key = (m["group"], m["seg"], m["sha"])
        if m["gen"] is None:
            payloads[key] = {"uri": uri, "size": size}
        else:
            receipts.setdefault(key, []).append(uri)

    out: dict[int, dict] = {}
    for (group, seg, sha), payload in payloads.items():
        found = receipts.get((group, seg, sha), [])
        if not found:
            continue  # payload still uploading or its receipt is missing
        if len(found) > 1:
            raise RuntimeError(f"{seg} has {len(found)} receipts for payload {sha}")
        index = int(seg[4:-4], 36)
        if index in out:
            raise RuntimeError(f"{seg} has more than one receipted payload")
        out[index] = {
            "group": group,
            "sha256": sha,
            "jss": {"name": seg, "size": payload["size"], "uri": payload["uri"]},
            "receipt": {"uri": found[0]},
        }
    return out


def completed_segments(prefix: str) -> dict[int, dict]:
    # A ** wildcard lists the bucket flat; -r walks every pseudo-directory and is far slower.
    return parse_listing(_gcloud("ls", "-l", prefix.rstrip("/") + "/segments/**"))


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_segment(seg: dict, dest_dir: pathlib.Path) -> pathlib.Path:
    """Fetch one payload and its receipt, verifying size and SHA-256 both ways."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = seg["jss"]["name"]
    target = dest_dir / name
    receipt = json.loads(_gcloud("cat", seg["receipt"]["uri"]))
    if (receipt.get("name") != name or receipt.get("sha256") != seg["sha256"]
            or int(receipt.get("bytes", -1)) != seg["jss"]["size"]):
        raise RuntimeError(f"receipt does not match {seg['jss']['uri']}")
    if not (target.exists() and target.stat().st_size == seg["jss"]["size"] and _sha256(target) == seg["sha256"]):
        target.unlink(missing_ok=True)
        _gcloud("cp", seg["jss"]["uri"], str(target))
        if target.stat().st_size != seg["jss"]["size"] or _sha256(target) != seg["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded bytes do not match {seg['jss']['uri']}")
    (dest_dir / (name + ".receipt.json")).write_text(json.dumps(receipt, indent=2))
    return target
