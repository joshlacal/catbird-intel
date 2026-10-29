"""One S12m work unit: gate -> fetch (in memory) -> decode -> encode -> one output table.

A unit is a slice of input rows. `prepare` applies the D9 gate and lists the thumbnails to fetch; the fetch runs
on the polite fetcher (in a background thread, overlapping the previous unit's GPU work); `encode_unit` embeds
every post with text once as text (the primary vector for text posts, the text-only companion for image posts,
kept so the report-only G-img comparison of A2 can be computed), and every post whose images all came back as
ONE interleaved forward pass. Image bytes live only in the fetch result dict and the decoded PIL objects of
one batch; both are dropped before the next batch. Nothing in this module writes a file.
"""
from __future__ import annotations

import collections
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mmlib as L
from encoder import decode


@dataclass
class Unit:
    name: str
    uris: list
    texts: list
    pids: list
    n_cids: list
    urls: list                     # per post: list of thumbnail URLs to fetch ([] = none)
    reasons: list                  # per post: gate reason or None
    counts: collections.Counter = field(default_factory=collections.Counter)

    def fetch_items(self):
        return [((i, j), u) for i, us in enumerate(self.urls) for j, u in enumerate(us)]


def prepare(name, rows, bundle, now, min_age=L.MIN_AGE):
    """rows: dicts with uri (or did+rkey), text, image_cids, self_labels, p."""
    u = Unit(name, [], [], [], [], [], [])
    for r in rows:
        if r.get("uri"):
            did, rkey = L.did_rkey(r["uri"])
            uri = r["uri"]
        else:
            did, rkey = r["did"], r["rkey"]
            uri = L.post_uri(did, rkey)
        cids = [c for c in (r.get("image_cids") or []) if c]
        urls, reason = L.gate_images(did, rkey, cids, r.get("self_labels"), bundle, now, min_age)
        u.uris.append(uri)
        u.texts.append(r.get("text") or "")
        u.pids.append(r.get("p"))
        u.n_cids.append(len(cids))
        u.urls.append(urls)
        u.reasons.append(reason)
        u.counts["posts"] += 1
        if cids:
            u.counts["posts_with_cids"] += 1
            u.counts[f"gate_{reason or 'pass'}"] += 1
    u.counts["images_to_fetch"] = sum(len(x) for x in u.urls)
    return u


def _batches(idx, bs):
    for s in range(0, len(idx), bs):
        yield idx[s:s + bs]


def encode_unit(unit, fetched, enc, *, text_bs=256, mm_bs=16, decode_threads=6, enc_threads=1, on_oom=None):
    """fetched: {(post, image): (bytes | None, log)}. Returns (columns dict for mmlib.output_schema, counts)."""
    n = len(unit.uris)
    c = unit.counts
    has_text = [bool(t.strip()) for t in unit.texts]
    tidx = [i for i in range(n) if has_text[i]]
    tvec = {}
    if tidx:
        v = L.to_stored(enc.encode_text([unit.texts[i] for i in tidx], bs=text_bs))
        tvec = dict(zip(tidx, L.pack(v)))

    reasons = list(unit.reasons)
    mm_ready = []
    for i, us in enumerate(unit.urls):
        if not us:
            continue
        logs = [fetched.get((i, j), (None, {"skip": "not_fetched"})) for j in range(len(us))]
        bad = next((lg["skip"] for b, lg in logs if b is None), None)
        if bad:
            reasons[i] = bad
            c[f"fetch_{bad}"] += 1
        else:
            mm_ready.append(i)

    mvec = {}
    mm_ready.sort(key=lambda i: len(unit.urls[i]))
    lock = threading.Lock()

    def run_batch(b, pool):
        """Decode and embed one batch; on CUDA OOM, halve it and run the halves in turn."""
        imgs, ok = {}, []
        for i, res in zip(b, pool.map(lambda i: _decode_post(i, unit, fetched), b)):
            if isinstance(res, str):
                reasons[i] = res
                with lock:
                    c[f"fetch_{res}"] += 1
            else:
                imgs[i] = res
                ok.append(i)
        try:
            v = L.to_stored(enc.encode_mm([(unit.texts[i], imgs[i]) for i in ok], bs=len(ok) or 1))
        except Exception as e:
            if "out of memory" not in str(e).lower() or len(b) == 1:
                raise
            if on_oom:
                on_oom()
            with lock:
                c["mm_oom_halvings"] += 1
            imgs.clear()
            for half in _batches(b, max(1, len(b) // 2)):
                run_batch(half, pool)
            return
        finally:
            imgs.clear()   # drop decoded pixels
        with lock:
            mvec.update(zip(ok, L.pack(v)))
            for i in ok:   # drop this post's bytes as soon as it is embedded
                for j in range(len(unit.urls[i])):
                    fetched.pop((i, j), None)

    # enc_threads > 1 overlaps one batch's CPU work (decode, image processor) with another's forward pass on the
    # same model. Batches are the same lists of posts either way, so each post's input is unchanged.
    batches = list(_batches(mm_ready, mm_bs))
    with ThreadPoolExecutor(decode_threads * enc_threads) as pool:
        if enc_threads <= 1:
            for b in batches:
                run_batch(b, pool)
        else:
            it = iter(batches)

            def worker():
                while True:
                    with lock:
                        b = next(it, None)
                    if b is None:
                        return
                    run_batch(b, pool)

            with ThreadPoolExecutor(enc_threads, thread_name_prefix="enc") as ew:
                for f in [ew.submit(worker) for _ in range(enc_threads)]:
                    f.result()

    cols = {k: [] for k in L.output_schema().names}
    for i in range(n):
        if i in mvec:
            mod = L.TEXT_IMAGE if has_text[i] else L.IMAGE_ONLY
            vec, vtext, nimg, skip = mvec[i], tvec.get(i), len(unit.urls[i]), None
        elif has_text[i]:
            mod = L.TEXT if unit.n_cids[i] == 0 else L.TEXT_FALLBACK
            vec, vtext, nimg, skip = tvec[i], None, 0, reasons[i]
        else:
            mod, vec, vtext, nimg, skip = L.NONE, None, None, 0, reasons[i] or ("no_text" if unit.n_cids[i] == 0 else None)
        c[f"modality_{L.MODALITY_NAMES[mod]}"] += 1
        c["images_embedded"] += nimg
        cols["uri"].append(unit.uris[i])
        cols["p"].append(unit.pids[i])
        cols["modality"].append(mod)
        cols["n_image_cids"].append(min(unit.n_cids[i], 127))
        cols["n_images"].append(nimg)
        cols["soft_tokens"].append(enc.soft_tokens if nimg else 0)
        cols["image_skip"].append(skip)
        cols["vec"].append(vec)
        cols["vec_text"].append(vtext)
    fetched.clear()
    assert all(v is None or len(v) == 2 * L.DIM for v in cols["vec"])
    return cols, c


def _decode_post(i, unit, fetched):
    out = []
    for j in range(len(unit.urls[i])):
        b, _lg = fetched[(i, j)]
        try:
            out.append(decode(b))
        except Exception as e:
            return f"decode_{type(e).__name__}"
    return out


def to_table(cols):
    import pyarrow as pa
    return pa.Table.from_pydict(cols, schema=L.output_schema())
