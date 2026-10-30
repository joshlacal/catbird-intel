"""S7g CDN rate ramp: how fast cdn.bsky.app serves feed thumbnails of mostly OLD posts (cache misses).

Steps of increasing target rate (default 25 -> 50 -> 100 -> 200 req/s), ~90 s each (all shortened equally, to >= 30 s,
when there are too few URLs: S7g has ~21k, so ~57 s each), every request a thumbnail URL
not requested before in this bench (the gate-passing image posts the main fetch did not use, in the bench's hash
order). Bodies are measured and dropped at once (`discard_bodies`); nothing is kept or written.

The ramp stops at the first of:
  * any 429 (the fetcher opens its circuit on the first one; the rest of the step is never sent);
  * more than 1% 5xx answers in a step (checked as answers arrive: the step is cut once its 5xx count passes 1% of
    its planned size, and again at the end on the actual count);
  * p95 latency above 3x the first step's p95;
  * too few fresh URLs left for a >= 30 s step (reported, not a CDN verdict).
Per step it records achieved req/s, statuses, the `cdn-cache` header counts (HIT / MISS / ...), HIT and MISS
shares, and p50/p95/p99 latency. The Mac HEAD probe of 2026-10-28 (429 at a 400 target / 277 achieved; clean at
200 / 157 achieved) ran on ~90% cache HITs; this ramp exists to measure the miss path S12m will mostly see.
"""
from __future__ import annotations

import math
import time

DEFAULT_STEPS = (25, 50, 100, 200)
STEP_S = 90.0
MIN_STEP_S = 30.0
MAX_5XX_SHARE = 0.01
P95_FACTOR = 3.0


def plan(n_fresh, steps=DEFAULT_STEPS, step_s=STEP_S, min_step_s=MIN_STEP_S):
    """[(rate, n_requests, seconds)] that fits in n_fresh URLs. When there are too few URLs for step_s at every
    rate, every step is shortened equally (never below min_step_s), so the top rate still gets measured; if even
    that does not fit, the steps that cannot get min_step_s are dropped from the top."""
    if sum(r * step_s for r in steps) > n_fresh:
        step_s = max(min_step_s, n_fresh / sum(steps))
    out, left = [], n_fresh
    for r in steps:
        want = int(round(r * step_s))
        n = min(want, left)
        if n < r * min_step_s:
            break
        out.append((r, n, round(n / r, 1)))
        left -= n
    return out


def in_flight_for(rate, latency_s=1.0):
    """Enough concurrent requests to sustain `rate` at `latency_s` per answer (Little's law), at least 32."""
    return max(32, min(512, math.ceil(rate * latency_s * 1.5)))


def share(counter, *keys):
    n = sum(counter.values())
    return round(sum(counter.get(k, 0) for k in keys) / n, 4) if n else None


def verdict(step, first_p95, max_5xx_share=MAX_5XX_SHARE, p95_factor=P95_FACTOR):
    """None if the ramp may go on after this step, else the reason it stops."""
    st, opened = step["statuses"], step.get("circuit_open", "")
    if st.get(429) or st.get("429") or "429" in opened:
        return "429"
    n = step["requests"] or 0
    n5 = sum(v for k, v in st.items() if str(k).isdigit() and int(k) >= 500)
    if (n and n5 / n > max_5xx_share) or "5xx" in opened:
        return f"5xx: {n5} of {n} answers (> {max_5xx_share:.0%})"
    p95 = step.get("latency_ms_p95")
    if first_p95 and p95 and p95 > p95_factor * first_p95:
        return f"p95 {p95} ms > {p95_factor} x first step's {first_p95} ms"
    return None


def run(items, make_fetcher, steps=DEFAULT_STEPS, step_s=STEP_S, min_step_s=MIN_STEP_S, log=print):
    """items: [(key, url)] of fresh thumbnail URLs. make_fetcher(rate, in_flight, max_5xx) -> AsyncFetcher with
    discard_bodies=True and stop_on_429=True. Returns the ramp report (no bytes)."""
    from fetcher import CircuitOpen
    p = plan(len(items), steps, step_s, min_step_s)
    rep = {"targets": list(steps), "step_s": step_s, "fresh_urls": len(items), "plan": p, "steps": [],
           "stopped": None, "max_clean_target_rps": None, "achieved_rps_at_max_clean": None}
    if len(p) < len(steps):
        rep["short_of_urls"] = f"{len(items)} fresh URLs fit {len(p)} of {len(steps)} steps"
    pos, first_p95 = 0, None
    for rate, n, secs in p:
        chunk = items[pos:pos + n]
        pos += n
        f = make_fetcher(rate, in_flight_for(rate), int(MAX_5XX_SHARE * n))
        t0 = time.time()
        res = {"target_rps": rate, "planned_requests": n, "planned_s": secs}
        try:
            f.fetch_sync(chunk)
        except CircuitOpen as e:
            res["circuit_open"] = str(e)
        finally:
            f.close()
        s = f.stats.summary()
        res.update(seconds=round(time.time() - t0, 1), requests=s["requests"], achieved_rps=s["achieved_rps"],
                   statuses={str(k): v for k, v in s["statuses"].items()}, cdn_cache=s["cdn_cache"],
                   hit_share=share(f.stats.cache, "HIT"), miss_share=share(f.stats.cache, "MISS", "EXPIRED"),
                   latency_ms_p50=s["latency_ms_p50"], latency_ms_p95=s["latency_ms_p95"],
                   latency_ms_p99=s["latency_ms_p99"], ok=s["ok"], bytes=s["bytes"], skips=s["skips"])
        if first_p95 is None:
            first_p95 = res["latency_ms_p95"]
        why = verdict(res, first_p95)
        res["verdict"] = why or "clean"
        rep["steps"].append(res)
        log("[ramp]", {k: v for k, v in res.items() if k not in ("skips",)})
        if why:
            rep["stopped"] = {"at_target_rps": rate, "reason": why}
            break
        rep["max_clean_target_rps"], rep["achieved_rps_at_max_clean"] = rate, res["achieved_rps"]
    if rep["stopped"] is None and len(p) < len(steps):
        rep["stopped"] = {"at_target_rps": steps[len(p)] if len(p) < len(steps) else None, "reason": "out_of_fresh_urls"}
    return rep
