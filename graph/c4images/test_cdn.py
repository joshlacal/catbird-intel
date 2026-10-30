"""CDN fetcher with a fake transport: non-200 is always a skip; pacing, backoff and URL guard."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cdn  # noqa: E402

URL = "https://cdn.bsky.app/img/feed_thumbnail/plain/did:plc:" + "a" * 24 + "/bafkreiaaaaaaaaaaaaaaaaaaaa"


class Fake:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, method, url, headers, timeout):
        self.calls.append((method, url, headers))
        return self.responses.pop(0)


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def fetcher(*responses, rps=10.0):
    clock = Clock()
    return cdn.CdnFetcher(Fake(*responses), max_rps=rps, clock=clock, sleep=clock.sleep), clock


def test_200_image_returns_bytes_and_sends_user_agent():
    f, _ = fetcher(cdn.Response(200, {"content-type": "image/jpeg"}, b"\xff\xd8\xffdata"))
    data, log = f.get(URL)
    assert data == b"\xff\xd8\xffdata" and log["skip"] is None and "body" not in log
    method, url, headers = f.transport.calls[0]
    assert method == "GET" and "contact" in headers["User-Agent"]


@pytest.mark.parametrize("resp,reason", [
    (cdn.Response(404), "cdn_404"), (cdn.Response(410), "cdn_410"), (cdn.Response(301, {"location": "x"}), "cdn_301"),
    (cdn.Response(451), "cdn_451"), (cdn.Response(200, {"content-type": "text/html"}, b"<html>"), "cdn_not_image"),
    (cdn.Response(200, {"content-type": "image/jpeg"}, b"x" * (cdn.MAX_BYTES + 1)), "cdn_oversize"),
    (cdn.Response(200, {"content-type": "image/jpeg"}, b""), "cdn_empty")])
def test_every_non_clean_response_is_a_skip(resp, reason):
    f, _ = fetcher(resp)
    data, log = f.get(URL)
    assert data is None and log["skip"] == reason


def test_429_and_5xx_skip_and_pause_without_retry():
    f, clock = fetcher(cdn.Response(429, {"retry-after": "30"}), cdn.Response(503), cdn.Response(200, {"content-type": "image/png"}, b"ok"))
    assert f.get(URL)[1]["skip"] == "cdn_429"
    f.get(URL)
    assert clock.t >= 30
    t_before = clock.t
    assert f.get(URL)[0] == b"ok"
    assert clock.t - t_before >= 5  # 5 s default pause after the 503
    assert len(f.transport.calls) == 3


def test_rate_cap_paces_requests():
    ok = cdn.Response(200, {"content-type": "image/jpeg"}, b"x")
    f, clock = fetcher(*[ok] * 5, rps=10)
    for _ in range(5):
        f.get(URL)
    assert clock.t == pytest.approx(0.4)


def test_refuses_other_urls_and_bad_settings():
    f, _ = fetcher()
    for bad in ("https://cdn.bsky.app/img/avatar/plain/did:plc:x/bafy", "https://evil.test/img/feed_thumbnail/plain/x",
                "http://cdn.bsky.app/img/feed_thumbnail/plain/x"):
        assert f.get(bad)[1]["skip"] == "not_feed_thumbnail_url"
    assert f.transport.calls == []
    with pytest.raises(ValueError):
        cdn.CdnFetcher(Fake(), max_rps=100)
    with pytest.raises(ValueError):
        cdn.CdnFetcher(Fake(), user_agent="python-urllib")


def test_head_returns_no_body():
    f, _ = fetcher(cdn.Response(200, {"content-type": "image/jpeg"}, b""))
    data, log = f.get(URL, method="HEAD")
    assert data == b"" and log["skip"] is None and f.transport.calls[0][0] == "HEAD"
