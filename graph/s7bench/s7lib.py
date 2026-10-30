"""Pure helpers for S7 (whole-network plan §3.1 row S7): Colab EG2 benchmark + GCS round trip.

Used on the Mac (prep, driver, tests) and on the Colab VM (bench.py, colab_job.py, roundtrip_job.py), so this
file imports only the standard library and numpy (lazily). Tested in test_s7lib.py.
"""
import hashlib
import json
import re
import time

# --- sampling -------------------------------------------------------------------------------------------------


def hash64(salt, keys):
    """Deterministic 64-bit order keys: first 8 bytes of blake2b(salt|key), the same rule as S4's prep_corpus."""
    return [int.from_bytes(hashlib.blake2b(f"{salt}|{k}".encode(), digest_size=8).digest(), "big") for k in keys]


def draw_smallest(keys, n, salt):
    """Indices of the n keys with the smallest salted hash, in hash order (ties impossible in practice; broken
    by index for determinism)."""
    h = hash64(salt, keys)
    order = sorted(range(len(keys)), key=lambda i: (h[i], i))
    return order[:n]


def draw_one_per_group(groups, keys, salt_group, salt_item, exclude=frozenset()):
    """Slab-style draw: groups (producers) in salted-hash order, and for each group its single item with the
    smallest salted hash among items not in `exclude`. Returns [(group, item_index)] for every group that has an
    eligible item, in group-hash order. The caller takes the first N."""
    best = {}
    hi = hash64(salt_item, keys)
    for i, (g, k) in enumerate(zip(groups, keys)):
        if k in exclude:
            continue
        cur = best.get(g)
        if cur is None or (hi[i], i) < cur[0]:
            best[g] = ((hi[i], i), i)
    gs = sorted(best)
    hg = dict(zip(gs, hash64(salt_group, gs)))
    gs.sort(key=lambda g: (hg[g], g))
    return [(g, best[g][1]) for g in gs]


# --- slab text format ------------------------------------------------------------------------------------------


def slab_text(text, alts):
    """O8's slab row text, as graph/vmpass/sql_p2.py builds it: the post text followed by ' [image: <alt>]' for
    every non-empty alt text, in order. The prompt prefix is NOT stored: the encoder prepends it
    (`task: clustering | query: `), exactly as S12 will."""
    return text + "".join(f" [image: {a}]" for a in (alts or []) if a != "")


def alts_from_record(record):
    """Alt texts of a post record's media, in the order graph/post_extract.py stores them: images (also inside
    recordWithMedia), or the single video alt. Missing alts are '' (so counts line up), dropped by slab_text."""
    emb = (record or {}).get("embed") or {}
    t = emb.get("$type", "")
    if t == "app.bsky.embed.recordWithMedia":
        emb = emb.get("media") or {}
        t = emb.get("$type", "")
    if t == "app.bsky.embed.images":
        return [str(i.get("alt") or "") for i in emb.get("images") or []]
    if t == "app.bsky.embed.video":
        return [str(emb.get("alt") or "")]
    return []


# --- secrets hygiene -------------------------------------------------------------------------------------------

_URL = re.compile(r"(https?://[^\s?#'\"<>]+)\?[^\s'\"<>]*")
_TOKENS = [
    (re.compile(r"ya29\.[0-9A-Za-z._\-]+"), "ya29.<redacted>"),                 # Google OAuth access tokens
    (re.compile(r"1//[0-9A-Za-z_\-]{20,}"), "1//<redacted>"),                    # Google refresh tokens
    (re.compile(r"\b4/[0-9A-Za-z_\-]{20,}"), "4/<redacted>"),                    # OAuth authorization codes
    (re.compile(r"(?i)\b(access_token|refresh_token|id_token|token|code|key|sig|signature|authorization)"
                r"(\"?\s*[:=]\s*\"?)(bearer\s+)?[^\s\"'&,}]{8,}"), r"\1\2<redacted>"),
]


def redact(s):
    """Strip query strings from URLs (Colab proxy/error URLs carry tokens there) and anything token-shaped."""
    s = _URL.sub(r"\1?<redacted>", s)
    for rx, rep in _TOKENS:
        s = rx.sub(rep, s)
    return s


# --- units ----------------------------------------------------------------------------------------------------

# Colab units/hour (handoff §1; the same table C5 uses). The driver also measures the live rate from `colab usage`.
RATES = {"T4": 1.19, "L4": 1.71, "A100": 5.4, "H100": None, "RTX PRO 6000": 8.71}


def gpu_rate(name):
    """(gpu, units/h) for a torch device name, or (None, None) when unknown."""
    for k, v in RATES.items():
        if v is not None and k in (name or "").upper():
            return ("G4" if k == "RTX PRO 6000" else k), v
    return None, None


class UnitGuard:
    """Projects Colab units from wall time since VM allocation; rate = units/hour of the GPU actually assigned."""

    def __init__(self, cap, rate, preamble_min=0.0, clock=time.time):
        self.cap, self.rate, self.clock = cap, rate, clock
        self.t0 = clock() - preamble_min * 60

    def spent(self):
        return (self.clock() - self.t0) / 3600 * self.rate

    def allows(self, next_seconds):
        return self.spent() + next_seconds / 3600 * self.rate <= self.cap

    def state(self):
        return {"units_spent_est": round(self.spent(), 3), "cap": self.cap, "rate_per_hour": self.rate}


def units_per_10m(posts_per_s, rate, inefficiency=1.0):
    """Units to embed 10M posts at a measured rate, with the arithmetic as a string."""
    hours = 1e7 / posts_per_s / 3600
    units = hours * rate * inefficiency
    eff = f" x {inefficiency:g}" if inefficiency != 1.0 else ""
    return units, f"1e7 / {posts_per_s:,.0f} posts/s = {hours * 3600:,.0f} s = {hours:.3f} h x {rate:g} units/h{eff} = {units:.2f} units"


def budget_seconds(cap_units, rate, spent_units=0.0):
    """Wall seconds a session may run before its projected spend reaches cap_units."""
    return max(0.0, (cap_units - spent_units) / rate * 3600)


def parse_usage(text):
    """`colab usage` output -> {balance, rate_per_hour, active}. Missing fields are None."""
    def grab(rx, cast):
        m = re.search(rx, text)
        return cast(m.group(1)) if m else None
    return {"balance": grab(r"Current balance:\s*([0-9.]+)", float),
            "rate_per_hour": grab(r"Usage rate:\s*([0-9.]+)", float),
            "active": grab(r"Active assignments:\s*([0-9]+)", int)}


# --- parity ---------------------------------------------------------------------------------------------------


def cosine_stats(a, b, dim=None):
    """Row-wise cosine between two (n, d) arrays (optionally truncated to `dim` first and renormalised, the 256-d
    storage rule): median, p01, min, and the count below 0.99."""
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if dim:
        a, b = a[:, :dim], b[:, :dim]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    c = (a * b).sum(axis=1) / np.maximum(na * nb, 1e-30)
    return {"n": int(len(c)), "median": float(np.median(c)), "p01": float(np.quantile(c, 0.01)),
            "min": float(c.min()), "below_0_99": int((c < 0.99).sum())}


# --- GCS round-trip guards ------------------------------------------------------------------------------------

RT_BUCKET = "catbird-436600-plan-30d"
RT_PREFIX = "colab-test/"


def rt_allowed(bucket, name):
    """R-S7 confines the Colab credential's use to plan-30d/colab-test/ in code (the credential itself is not
    prefix-scoped). Rejects traversal and anything outside the prefix."""
    return (bucket == RT_BUCKET and name.startswith(RT_PREFIX) and ".." not in name.split("/")
            and "//" not in name and len(name) > len(RT_PREFIX))


def scopes_from_tokeninfo(info):
    """Sorted scope list from an oauth2 tokeninfo response (space-separated `scope`)."""
    return sorted((info or {}).get("scope", "").split())


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def manifest(paths_by_rel):
    """{rel: sha256} for the files the VM must verify before running anything."""
    return {rel: sha256_file(p) for rel, p in sorted(paths_by_rel.items())}


def dumps(o):
    return json.dumps(o, indent=1, sort_keys=False, default=str)


def randbytes_to(stream, n, sha_path, block=1 << 24):
    """Write n cryptographically random bytes (the synthetic GCS test object: no personal data) to stream, and
    their sha256 to sha_path once all are written."""
    import os
    h = hashlib.sha256()
    left = n
    while left:
        b = os.urandom(min(left, block))
        h.update(b)
        stream.write(b)
        left -= len(b)
    stream.flush()
    with open(sha_path, "w") as f:
        f.write(h.hexdigest() + "\n")
    return h.hexdigest()


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["redact"]:          # filter: run_colab.sh pipes every colab CLI output through this
        for line in sys.stdin:
            sys.stdout.write(redact(line))
            sys.stdout.flush()
    elif sys.argv[1:2] == ["randbytes"]:     # randbytes N SHA_PATH > stdout
        randbytes_to(sys.stdout.buffer, int(sys.argv[2]), sys.argv[3])
    else:
        sys.exit("usage: s7lib.py redact | randbytes N SHA_PATH")
