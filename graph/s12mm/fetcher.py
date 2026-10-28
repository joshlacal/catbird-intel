"""Polite concurrent CDN thumbnail fetcher for S12m: bytes in memory only, every non-200 is skip-and-log.

asyncio drives the schedule: a global token bucket (default 50 requests/s, the plan §5.3 etiquette cap), a
bound on requests in flight, and a global pause after any 429/5xx (Retry-After honoured up to 60 s, else 5 s).
Each blocking GET runs in a worker thread on a thread-local keep-alive HTTPS connection (stdlib http.client),
so one VM uses one IP and a handful of TCP connections. No retries of any response; a connection the server
closed before answering is reopened once (that is not a retry of a CDN answer). Redirects are never followed.
A circuit breaker trips when, over the last `window` answers, 429s exceed 5% or 5xx/network errors exceed
20%; the caller then stops the run rather than silently degrading posts to text.

URL policy: only `https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}`, except that a rehearsal may
point `base` at a loopback HTTP server (127.0.0.1 / ::1 / localhost); nothing else is ever requested.
Nothing here opens a file. Callers get {key: (bytes | None, log)}; logs never carry the body.
"""
from __future__ import annotations

import asyncio
import collections
import http.client
import ipaddress
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

CDN_PREFIX = "https://cdn.bsky.app/img/feed_thumbnail/plain/"
USER_AGENT = "catbird-community-research/0.2 (+https://catbird.blue; contact joshlacal@gmail.com)"
ETIQUETTE_RPS = 50.0
MAX_BYTES = 2 * 1024 * 1024
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass
class Resp:
    status: int
    content_type: str = ""
    body: bytes = b""
    retry_after: str = ""


class KeepAliveTransport:
    """GET over a per-thread persistent connection to one origin. `base` is scheme://host[:port]."""

    def __init__(self, base, timeout=20.0):
        u = urllib.parse.urlsplit(base)
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port
        self.timeout = timeout
        self._tls = threading.local()

    def _conn(self, fresh=False):
        c = getattr(self._tls, "c", None)
        if c is None or fresh:
            if c is not None:
                c.close()
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            c = cls(self.host, self.port, timeout=self.timeout)
            self._tls.c = c
        return c

    def __call__(self, path, headers) -> Resp:
        for attempt in (0, 1):
            c = self._conn(fresh=attempt == 1)
            try:
                c.request("GET", path, headers=headers)
                r = c.getresponse()
                body = r.read(MAX_BYTES + 1)
                if r.length:            # oversize body: drop the connection rather than drain it
                    c.close()
                    self._tls.c = None
                return Resp(r.status, r.getheader("content-type", "") or "", body, r.getheader("retry-after", "") or "")
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                if attempt == 1:
                    raise
            except Exception:
                self._tls.c = None
                raise
        raise RuntimeError("unreachable")


@dataclass
class Stats:
    requests: int = 0
    ok: int = 0
    bytes: int = 0
    statuses: collections.Counter = field(default_factory=collections.Counter)
    skips: collections.Counter = field(default_factory=collections.Counter)
    pauses: int = 0
    latency_ms: list = field(default_factory=list)
    t_first: float | None = None
    t_last: float | None = None

    def rate(self):
        if not self.t_first or self.t_last is None or self.t_last <= self.t_first:
            return None
        return (self.requests - 1) / (self.t_last - self.t_first) if self.requests > 1 else None

    def summary(self):
        lat = sorted(self.latency_ms)
        q = lambda f: round(lat[min(len(lat) - 1, int(f * len(lat)))], 1) if lat else None  # noqa: E731
        return {"requests": self.requests, "ok": self.ok, "bytes": self.bytes, "achieved_rps": self.rate(),
                "statuses": dict(self.statuses), "skips": dict(self.skips), "pauses": self.pauses,
                "latency_ms_p50": q(0.5), "latency_ms_p95": q(0.95), "latency_ms_p99": q(0.99)}


class CircuitOpen(RuntimeError):
    pass


class AsyncFetcher:
    def __init__(self, *, rps=ETIQUETTE_RPS, max_in_flight=32, base="https://cdn.bsky.app", rehearsal=False,
                 allow_above_etiquette=False, user_agent=USER_AGENT, transport=None, window=2000,
                 clock=time.monotonic):
        if not 0 < rps <= (200.0 if allow_above_etiquette else ETIQUETTE_RPS):
            raise ValueError(f"rps {rps} outside (0, {ETIQUETTE_RPS}] (above needs allow_above_etiquette)")
        if "contact" not in user_agent:
            raise ValueError("User-Agent must be descriptive and carry a contact")
        host = urllib.parse.urlsplit(base).hostname or ""
        if rehearsal:
            if host not in LOOPBACK and not _is_loopback(host):
                raise ValueError("rehearsal base must be a loopback address")
        elif base != "https://cdn.bsky.app":
            raise ValueError("only https://cdn.bsky.app outside a rehearsal")
        self.base, self.rehearsal = base.rstrip("/"), rehearsal
        self.rps, self.max_in_flight, self.user_agent, self.clock = rps, max_in_flight, user_agent, clock
        self.transport = transport or KeepAliveTransport(self.base)
        self.pool = ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="cdn")
        self.stats = Stats()
        self._recent = collections.deque(maxlen=window)
        self._next = 0.0
        self._pause_until = 0.0
        self.tripped: str | None = None

    def close(self):
        self.pool.shutdown(wait=True)

    def to_request_path(self, url: str) -> str:
        """Maps the canonical CDN URL to the path requested; refuses anything else."""
        if not url.startswith(CDN_PREFIX):
            raise ValueError(f"refusing non-thumbnail URL {url!r}")
        return url[len("https://cdn.bsky.app"):]

    async def _slot(self):
        while True:
            now = self.clock()
            wait = max(self._next, self._pause_until) - now
            if wait <= 0:
                self._next = max(now, self._next) + 1.0 / self.rps
                return
            await asyncio.sleep(wait)

    def _record(self, status_kind):
        self._recent.append(status_kind)
        n = len(self._recent)
        if n >= min(200, self._recent.maxlen):
            c = collections.Counter(self._recent)
            if c["429"] / n > 0.05:
                self.tripped = f"429 share {c['429'] / n:.3f} over last {n}"
            elif c["err"] / n > 0.20:
                self.tripped = f"5xx/network share {c['err'] / n:.3f} over last {n}"

    async def _one(self, sem, key, url):
        try:
            path = self.to_request_path(url)
        except ValueError:
            self.stats.skips["not_feed_thumbnail_url"] += 1
            return key, None, {"status": None, "skip": "not_feed_thumbnail_url"}
        async with sem:
            if self.tripped:
                return key, None, {"status": None, "skip": "circuit_open"}
            await self._slot()
            t0 = self.clock()
            self.stats.t_first = self.stats.t_first or t0
            loop = asyncio.get_running_loop()
            hdr = {"User-Agent": self.user_agent, "Accept": "image/*", "Host": "cdn.bsky.app"}
            try:
                r = await loop.run_in_executor(self.pool, self.transport, path, hdr)
            except Exception as e:  # network error: skip, never retried
                r = None
                err = type(e).__name__
            self.stats.t_last = self.clock()
            self.stats.requests += 1
            ms = (self.stats.t_last - t0) * 1000
            self.stats.latency_ms.append(ms)
            if r is None:
                self.stats.statuses["net_error"] += 1
                self._record("err")
                self.stats.skips[f"net_{err}"] += 1
                return key, None, {"status": None, "skip": f"net_{err}", "ms": round(ms, 1)}
            self.stats.statuses[r.status] += 1
            log = {"status": r.status, "content_type": r.content_type, "bytes": len(r.body), "ms": round(ms, 1)}
            if r.status == 429 or r.status >= 500:
                ra = r.retry_after.strip()
                delay = min(60.0, float(ra)) if ra.isdigit() else 5.0
                self._pause_until = max(self._pause_until, self.clock() + delay)
                self.stats.pauses += 1
                self._record("429" if r.status == 429 else "err")
            else:
                self._record("ok")
            skip = None
            if r.status != 200:
                skip = f"cdn_{r.status}"
            elif not r.content_type.startswith("image/"):
                skip = "cdn_not_image"
            elif len(r.body) > MAX_BYTES:
                skip = "cdn_oversize"
            elif not r.body:
                skip = "cdn_empty"
            if skip:
                self.stats.skips[skip] += 1
                return key, None, {**log, "skip": skip}
            self.stats.ok += 1
            self.stats.bytes += len(r.body)
            return key, r.body, {**log, "skip": None}

    async def fetch(self, items):
        """items: iterable of (key, url). Returns {key: (bytes | None, log)}. Raises CircuitOpen if tripped."""
        sem = asyncio.Semaphore(self.max_in_flight)
        res = await asyncio.gather(*(self._one(sem, k, u) for k, u in items))
        if self.tripped:
            raise CircuitOpen(self.tripped)
        return {k: (b, log) for k, b, log in res}

    def fetch_sync(self, items):
        return asyncio.run(self.fetch(list(items)))


def _is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
