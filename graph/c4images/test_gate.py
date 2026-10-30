"""Fetch gate on fixtures (no network): every skip reason, the age rule, the cap and the URL builder."""
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gate  # noqa: E402

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
ALPH = "234567abcdefghijklmnopqrstuvwxyz"
CID = "bafkreidtt757okdivxau7bl5jgwtmbaq62s7s67cd63greeaqbiq33msae"


def tid(dt, clock=7):
    v = (int(dt.timestamp() * 1e6) << 10) | clock
    return "".join(ALPH[(v >> ((12 - i) * 5)) & 31] for i in range(13))


def did(c):
    return "did:plc:" + c * 24


def post(d, age_days=5, cids=(CID,), clock=7, **kw):
    r = {"did": d, "rkey": tid(NOW - timedelta(days=age_days), clock), "rev": "3m000000000aa", "seq": 1,
         "operation": "create", "reply_parent": "", "self_labels": [], "image_cids": list(cids),
         "text": "a photo", "image_alts": ["alt"] * len(cids)}
    r.update(kw)
    return r


def uri(r):
    return f"at://{r['did']}/app.bsky.feed.post/{r['rkey']}"


def run(rows, excl=None, labelled=(), **kw):
    return gate.gate(rows, excl or gate.DidExclusions({}), set(labelled), NOW, **kw)


def test_tid_roundtrip_and_invalid():
    t = NOW - timedelta(days=3, seconds=1)
    assert abs((gate.tid_time(tid(t)) - t).total_seconds()) < 1e-3
    assert gate.tid_time("self") is None
    assert gate.tid_time("zzzzzzzzzzzzz") is None  # top bit set
    assert gate.tid_time("3lxyz!bcdefgh") is None


def test_thumb_url_format_and_refusals():
    assert gate.thumb_url("did:plc:pa6yviflbsmtv2kqla5q6ujs", CID) == (
        "https://cdn.bsky.app/img/feed_thumbnail/plain/did:plc:pa6yviflbsmtv2kqla5q6ujs/" + CID)
    for d, c in [("did:plc:short", CID), (did("a"), "../avatar/x"), (did("a"), CID.upper()),
                 ("did:web:evil.com/../..", CID), (did("a"), "Qm" + "a" * 44)]:
        with pytest.raises(ValueError):
            gate.thumb_url(d, c)


def test_age_rule_is_three_days():
    old, young = post(did("a"), age_days=3.01), post(did("b"), age_days=2.99)
    fut = post(did("c"), age_days=-1)
    fetch, skips, counts = run([old, young, fut])
    assert [f["uri"] for f in fetch] == [uri(old)]
    assert {s["uri"]: s["reason"] for s in skips} == {uri(young): "too_young", uri(fut): "bad_tid"}


def test_each_post_level_reason():
    rows = {
        "latest_rev_deleted": post(did("a"), operation="delete"),
        "reply": post(did("b"), reply_parent="at://x/app.bsky.feed.post/1"),
        "no_image_cids": post(did("c"), cids=()),
        "post_self_label": post(did("d"), self_labels=["porn"]),
        "record_label_replay": post(did("e")),
        "bad_tid": post(did("f"), rkey="self"),
    }
    fetch, skips, counts = run(list(rows.values()), labelled={uri(rows["record_label_replay"])})
    assert fetch == []
    assert {s["uri"]: s["reason"] for s in skips} == {uri(r): k for k, r in rows.items()}


def test_did_reasons_in_priority_order():
    d = did("a")
    excl = gate.DidExclusions({"claude_profile": {d}, "purge": {d}, "scope_community": {d}})
    _f, skips, _c = run([post(d)], excl)
    assert skips[0]["reason"] == "purge"
    for reason in gate.DID_REASONS:
        _f, skips, _c = run([post(d)], gate.DidExclusions({reason: {d}}))
        assert skips[0]["reason"] == reason


def test_latest_revision_wins():
    first = post(did("a"), rev="3m000000000aa", self_labels=["porn"])
    second = dict(first, rev="3m000000000ab", self_labels=[])
    fetch, _s, _c = run([second, first])
    assert len(fetch) == 1
    deleted = dict(first, rev="3m000000000ac", operation="delete", image_cids=[])
    _f, skips, _c = run([first, second, deleted])
    assert skips[0]["reason"] == "latest_rev_deleted"


def test_producer_cap_eight_images_whole_posts_deterministic():
    d = did("a")
    rows = [post(d, clock=i, cids=[CID[:-1] + ALPH[i]] * 3) for i in range(5)]  # 5 posts x 3 images
    fetch, skips, counts = run(rows)
    assert counts["fetch_images"] == 6 and counts["fetch_posts"] == 2  # 3+3 <= 8 < 9
    assert counts["producer_cap"] == 3
    again, _s, _c = run(list(reversed(rows)))
    assert [f["uri"] for f in fetch] == [f["uri"] for f in again]


def test_producer_allowlist_avatar_cids_and_alts():
    a, b = post(did("a"), cids=[CID, "bafkreiaaaaaaaaaaaaaaaaaaaa"], image_alts=["first", "second"]), post(did("b"))
    fetch, skips, counts = run([a, b], producers={did("a")}, avatar_banner_cids=frozenset({CID}))
    assert {s["reason"] for s in skips} == {"not_selected_producer"}
    assert [(f["cid"], f["alt"]) for f in fetch] == [("bafkreiaaaaaaaaaaaaaaaaaaaa", "second")]
    assert counts["avatar_or_banner_cid"] == 1
    assert all(f["url"].startswith("https://cdn.bsky.app/img/feed_thumbnail/plain/") for f in fetch)


def test_misaligned_alts_are_not_attached():
    r = post(did("a"), cids=[CID], image_alts=["dropped-image alt", "kept"])
    fetch, _s, _c = run([r])
    assert fetch[0]["alt"] == ""


def test_loaders_on_fixture_files(tmp_path):
    a, b, c, e, f, g, h = (did(x) for x in "abcefgh")
    (tmp_path / "purge.txt").write_text(a + "\n")
    (tmp_path / "hfr.tsv").write_text(b + "\textra\n")
    pq.write_table(pa.Table.from_pylist([
        {"did": c, "found": True, "mod_labels": [], "self_labels": ["!no-unauthenticated"], "adult_graphic": False,
         "hard_label": False, "any_mod_label": False, "no_unauthenticated": True, "repo_status": "active"},
        {"did": h, "found": True, "mod_labels": [], "self_labels": [], "adult_graphic": False, "hard_label": False,
         "any_mod_label": False, "no_unauthenticated": False, "repo_status": "active"}]), tmp_path / "s5.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"did": e, "haiku_account_type": "person", "haiku_sensitivity": [], "sonnet_account_type": "person",
         "sonnet_sensitivity": ["likely_minors"]},
        {"did": h, "haiku_account_type": "person", "haiku_sensitivity": ["politics"],
         "sonnet_account_type": "person", "sonnet_sensitivity": []}]), tmp_path / "prof.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"community": 1, "community_type": "fandom", "sensitivity": ["adult"]},
        {"community": 2, "community_type": "roleplay", "sensitivity": []},
        {"community": 3, "community_type": "topical_scene", "sensitivity": []}]), tmp_path / "comm.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"did": f, "community": 1, "rank": 2}, {"did": g, "community": 2, "rank": 0},
        {"did": h, "community": 3, "rank": 0}, {"did": h, "community": 2, "rank": 1}]), tmp_path / "kf.parquet")
    (tmp_path / "labels.tsv").write_text(
        f"{h}\taccount\t{h}\tspam\tsrc\t2024\t\n{h}\trecord\tat://{h}/app.bsky.feed.post/x\tporn\tsrc\t2024\t\n")
    paths = gate.Paths(purge=tmp_path / "purge.txt", hide_from_recs=tmp_path / "hfr.tsv",
                       s5_profiles=tmp_path / "s5.parquet", profile_labels=tmp_path / "prof.parquet",
                       community_labels=tmp_path / "comm.parquet", known_for=tmp_path / "kf.parquet")
    ex = gate.load_did_exclusions(paths, account_labels_tsv=None)
    assert [ex.reason(x) for x in (a, b, c, e, f, g, h)] == [
        "purge", "hide_from_recs", "account_label_s5", "claude_profile", "claude_community", "scope_community", None]
    assert gate.load_did_exclusions(paths, scope=False).reason(g) is None
    ex2 = gate.load_did_exclusions(paths, account_labels_tsv=tmp_path / "labels.tsv")
    assert ex2.reason(h) == "account_label_replay"
    assert gate.record_labelled(tmp_path / "labels.tsv") == {f"at://{h}/app.bsky.feed.post/x"}


def test_cli_end_to_end(tmp_path, monkeypatch):
    rows = [post(did("a")), post(did("b"), age_days=1)]
    schema = pa.schema([(k, pa.list_(pa.string())) if isinstance(v, list) else (k, pa.int64() if k == "seq" else pa.string())
                        for k, v in rows[0].items()])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp_path / "c.parquet")
    (tmp_path / "l.tsv").write_text("")
    monkeypatch.setattr(gate, "load_did_exclusions", lambda **kw: gate.DidExclusions({}))
    gate.main([str(tmp_path / "c.parquet"), str(tmp_path / "out"), "--record-labels", str(tmp_path / "l.tsv"),
               "--now", NOW.isoformat()])
    t = pq.read_table(tmp_path / "out/fetch_list.parquet").to_pylist()
    assert [r["did"] for r in t] == [did("a")]
    assert pq.read_table(tmp_path / "out/skips.parquet").to_pylist()[0]["reason"] == "too_young"
