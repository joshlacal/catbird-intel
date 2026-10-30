"""Bluesky CDN thumbnail fetcher for C4: bytes in memory only, every non-200 is a skip.

The CDN (`cdn.bsky.app`, Bluesky's image service) applies takedowns and blob blocks, so under the
amended D9 policy it is the live check: a post whose thumbnail does not come back 200 with an image
content type is skipped and logged, never retried into success. Etiquette (plan §5.3): descriptive
User-Agent with contact, a global request-rate cap (default 10/s, at most 50/s), a pause after any
429/5xx (honouring Retry-After up to 60 s), no IP rotation, redirects never followed, bodies capped.

The HTTP transport is injected: `transport(method, url, headers, timeout) -> Response`. `UrllibTransport`
is the real one; tests pass a fake. Nothing here writes to disk; callers get `bytes` or None plus a
log record without the body.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from gate import CDN_HOST

USER_AGENT = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
MAX_BYTES = 2 * 1024 * 1024
MAX_RPS = 50.0


@dataclass
class Response:
    status: int
    headers: dict = field(default_factory=dict)
    body: bytes = b""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class UrllibTransport:
    """Real transport: HTTPS only, no redirects followed (a 3xx comes back as its status), body read
    to at most MAX_BYTES + 1 so an oversize body is detected without buffering it whole."""
    def __init__(self):
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __call__(self, method, url, headers, timeout):
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with self._opener.open(req, timeout=timeout) as r:
                body = r.read(MAX_BYTES + 1) if method == "GET" else b""
                return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, body)
        except urllib.error.HTTPError as e:
            return Response(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, b"")


class CdnFetcher:
    def __init__(self, transport, *, user_agent=USER_AGENT, max_rps=10.0, timeout=20.0,
                 clock=time.monotonic, sleep=time.sleep):
        if not 0 < max_rps <= MAX_RPS:
            raise ValueError(f"max_rps must be in (0, {MAX_RPS}]")
        if "contact" not in user_agent:
            raise ValueError("User-Agent must be descriptive and carry a contact")
        self.transport, self.user_agent, self.timeout = transport, user_agent, timeout
        self.interval, self.clock, self.sleep = 1.0 / max_rps, clock, sleep
        self._next = 0.0
        self.requests = 0
        self.statuses: dict[int, int] = {}

    def _pace(self):
        now = self.clock()
        if now < self._next:
            self.sleep(self._next - now)
            now = self._next
        self._next = now + self.interval

    def _backoff(self, resp):
        ra = (resp.headers.get("retry-after") or "").strip()
        delay = min(60.0, float(ra)) if ra.isdigit() else 5.0
        self._next = max(self._next, self.clock() + delay)

    def get(self, url: str, method: str = "GET"):
        """(bytes | None, log record). bytes only for 200 + image/* + within MAX_BYTES."""
        if not url.startswith(f"https://{CDN_HOST}/img/feed_thumbnail/plain/"):
            return None, {"url": url, "status": None, "skip": "not_feed_thumbnail_url"}
        self._pace()
        resp = self.transport(method, url, {"User-Agent": self.user_agent, "Accept": "image/*"}, self.timeout)
        self.requests += 1
        self.statuses[resp.status] = self.statuses.get(resp.status, 0) + 1
        log = {"url": url, "status": resp.status, "content_type": resp.headers.get("content-type", ""),
               "bytes": len(resp.body)}
        if resp.status == 429 or resp.status >= 500:
            self._backoff(resp)
        if resp.status != 200:
            return None, {**log, "skip": f"cdn_{resp.status}"}
        if not log["content_type"].startswith("image/"):
            return None, {**log, "skip": "cdn_not_image"}
        if method == "GET" and len(resp.body) > MAX_BYTES:
            return None, {**log, "skip": "cdn_oversize"}
        if method == "GET" and not resp.body:
            return None, {**log, "skip": "cdn_empty"}
        return (resp.body if method == "GET" else b""), {**log, "skip": None}
