"""Read S5 (i)'s live getProfiles sweep into a per-DID status, hashing every raw AppView response.

  load(S5_DIR) -> Live

S5_DIR is the S5 sweep directory (default ~/catbird-lake-local/s5-sweep-20261028). The reader is
format-tolerant on purpose, because S6 was written while S5 (i) was still running: it walks every
*.jsonl / *.jsonl.gz / *.json file except receipt.json, and in each JSON record finds
  - detailed profile views (dicts with a "did:" `did`, a `handle` and `followersCount`/`postsCount`,
    i.e. getProfiles' ProfileViewDetailed, not the basic author views inside getAuthorFeed), at any
    depth, including inside a raw response body stored as a JSON string;
  - requested DIDs: string lists under actors / dids / requested / request_dids / batch, `actors=`
    in a URL, or the top-level `did` of a record that has a `profile` key (null when not returned).
S5 (i)'s rows look like {"batch": n, "rows": [{"did", "found": true, "handle", "postsCount",
"followersCount", "labels", ...} | {"did", "found": false}]}; a record with an `error` key is a failed
call and leaves its DIDs unchecked. Files whose first line has no profile markers are skipped.
A DID requested but never returned is "missing" (taken down, deactivated, deleted or not found).
A DID never requested has no live check and is never shown. Every raw line consumed is sha256-hashed
for the S6 receipt, and each file's sha256 is recorded too.

Status per DID: ok, missing, no_unauth (`!no-unauthenticated`), labelled (any other non-negated label,
self or moderation), invalid_handle (`handle.invalid`), or unchecked.
"""
import dataclasses
import gzip
import hashlib
import json
import os
import pathlib
import re
import urllib.parse

REQ_KEYS = ("actors", "dids", "requested", "request_dids", "batch", "requested_dids")
ACTORS_RE = re.compile(r"actors=([^&\s\"']+)")


@dataclasses.dataclass
class Live:
    present: bool
    receipt: dict
    receipt_sha256: str | None
    profiles: dict  # did -> profile dict (last seen wins)
    requested: set
    files: list  # [{path, sha256, bytes, lines, response_records, profiles}]
    response_hashes: list  # sha256 of each raw record that carried >=1 profile or request
    controls: dict  # name -> bool, from the receipt
    errored_only: set = dataclasses.field(default_factory=set)  # DIDs only seen in failed calls

    def status(self, did):
        p = self.profiles.get(did)
        if p is None:
            return "missing" if did in self.requested else "unchecked"
        labels = [l.get("val", "") for l in (p.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
        if "!no-unauthenticated" in labels:
            return "no_unauth"
        if labels:
            return "labelled"
        if p.get("handle") == "handle.invalid":
            return "invalid_handle"
        return "ok"

    def labels(self, did):
        p = self.profiles.get(did) or {}
        return sorted({l.get("val", "") for l in (p.get("labels") or []) if isinstance(l, dict) and not l.get("neg")})


def _walk(obj, profiles, requested, depth=0):
    if depth > 8:
        return
    if isinstance(obj, dict):
        d = obj.get("did")
        if isinstance(d, str) and d.startswith("did:") and "handle" in obj and (
                "followersCount" in obj or "postsCount" in obj) and obj.get("found", True) is not False:
            profiles.append(obj)
            return
        if isinstance(d, str) and d.startswith("did:") and obj.get("found") is False:
            requested.add(d)  # S5 row: requested, not returned by getProfiles
            return
        for k, v in obj.items():
            if k in REQ_KEYS and isinstance(v, list):
                requested.update(x for x in v if isinstance(x, str) and x.startswith("did:"))
            elif isinstance(v, str):
                if k in ("url", "request", "query", "path") and "actors=" in v:
                    requested.update(urllib.parse.unquote(m) for m in ACTORS_RE.findall(v))
                elif v[:1] in "{[" and ('"profiles"' in v or '"handle"' in v):
                    try:
                        _walk(json.loads(v), profiles, requested, depth + 1)
                    except ValueError:
                        pass
            else:
                _walk(v, profiles, requested, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, profiles, requested, depth + 1)


def _controls(receipt):
    """Booleans under any key containing 'control' (e.g. {"controls": {"labelled": true, ...}})."""
    out = {}

    def rec(o, prefix):
        if isinstance(o, dict):
            for k, v in o.items():
                name = f"{prefix}.{k}" if prefix else k
                if "control" in prefix.lower() or "control" in k.lower():
                    if isinstance(v, bool):
                        out[name] = v
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, bool):
                                out[f"{name}.{kk}"] = vv
                            elif isinstance(vv, dict) and isinstance(vv.get("caught", vv.get("pass")), bool):
                                out[f"{name}.{kk}"] = vv.get("caught", vv.get("pass"))
                rec(v, name)

    rec(receipt, "")
    return out


def _lines(f):
    opener = gzip.open if f.name.endswith(".gz") else open
    with opener(f, "rb") as fp:
        if f.name.endswith(".json"):
            yield fp.read()
        else:
            yield from fp


def load(s5_dir):
    root = pathlib.Path(s5_dir).expanduser()
    rpath = root / "receipt.json"
    if not rpath.exists():
        return Live(False, {}, None, {}, set(), [], [], {})
    raw = rpath.read_bytes()
    receipt = json.loads(raw)
    profiles, requested, errored, files, hashes = {}, set(), set(), [], []
    found = sorted(pathlib.Path(dp) / n for dp, _dn, fn in os.walk(root, followlinks=True) for n in fn)
    for f in found:
        if not f.is_file() or f.name == "receipt.json" or "_trash" in f.parts:
            continue
        if not (f.name.endswith(".jsonl") or f.name.endswith(".jsonl.gz") or f.name.endswith(".json")):
            continue
        head = next(_lines(f), b"")[:65536]
        if not any(k in head for k in (b'"found"', b'"profiles"', b'"followersCount"')):
            continue  # not a getProfiles record file (e.g. author feeds, repo status, PLC pages)
        info = {"path": str(f), "lines": 0, "response_records": 0, "error_records": 0, "profiles": 0, "not_found": 0}
        for line in _lines(f):
            if not line.strip():
                continue
            info["lines"] += 1
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and "error" in obj:
                info["error_records"] += 1  # a failed call proves nothing about its DIDs: they stay unchecked
                errored.update(x for x in obj.get("dids", []) if isinstance(x, str))
                continue
            ps, rq = [], set()
            _walk(obj, ps, rq)
            if ps or rq:
                info["response_records"] += 1
                hashes.append(hashlib.sha256(line.rstrip(b"\n")).hexdigest())
            for p in ps:
                profiles[p["did"]] = p
                rq.add(p["did"])
            info["profiles"] += len(ps)
            info["not_found"] += len(rq) - len({p["did"] for p in ps})
            requested |= rq
        info["sha256"] = sha256_file(f)
        info["bytes"] = f.stat().st_size
        files.append(info)
    live = Live(True, receipt, hashlib.sha256(raw).hexdigest(), profiles, requested, files, hashes, _controls(receipt))
    live.errored_only = errored - requested
    return live


def sha256_file(f):
    h = hashlib.sha256()
    with open(f, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
