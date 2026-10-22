"""Synthetic inputs for graph/fold_follows.sql with known expected follows.

The shapes follow the deployed collector (jetstream v0.4.4):
- a first repo download stamps every record with the download's commit rev and
  emits no sync event; the collector records that rev as backfill.rev;
- a re-download or resync emits a sync event at its rev before the records;
- bootstrap-live events merged into the main sequence keep their rev but get
  the merge time as their envelope time.
Cases A-L reproduce the 2026-10-22 review fixtures; M and N add the stop-window
follow and sync-reset paths.
"""
import datetime as dt
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

ALPHABET = "234567abcdefghijklmnopqrstuvwxyz"
AUG = "2026-08-04T06:00:00Z"
OCT = "2026-10-19T22:00:00Z"
OLD = "2026-05-01T00:00:00.000Z"
HOST = "morel.us-east.host.bsky.network"
LOW_HOST = "amanita.us-east.host.bsky.network"

EVENT_SCHEMA = pa.schema([
    ("source_id", pa.string()), ("seq", pa.uint64()), ("time", pa.string()), ("kind", pa.string()),
    ("did", pa.string()), ("collection", pa.string()), ("rkey", pa.string()), ("rev", pa.string()),
    ("operation", pa.string()), ("created_at", pa.string()), ("subject", pa.string()),
    ("account_active", pa.bool_()), ("account_status", pa.string()),
])
NETCUP_FOLLOWS = pa.schema([
    ("did", pa.string()), ("rkey", pa.string()), ("rev", pa.uint64()), ("is_deleted", pa.uint8()),
    ("rkey_date", pa.date32()), ("seq", pa.uint64()), ("indexed_at", pa.timestamp("ms", tz="UTC")),
    ("subject", pa.string()),
])
NETCUP_DELETIONS = pa.schema([
    ("collection", pa.string()), ("did", pa.string()), ("rkey", pa.string()), ("rev", pa.uint64()),
    ("seq", pa.uint64()), ("indexed_at", pa.timestamp("ms", tz="UTC")),
])
NETCUP_ACCOUNTS = pa.schema([
    ("did", pa.string()), ("seq", pa.uint64()), ("indexed_at", pa.timestamp("ms", tz="UTC")),
    ("active", pa.uint8()), ("status", pa.string()),
])


def ts(text):
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def tid_u64(text, clock=0):
    return (int(ts(text).timestamp() * 1_000_000) << 10) | clock


def tid(text, clock=0):
    v = tid_u64(text, clock)
    return "".join(ALPHABET[(v >> ((12 - i) * 5)) & 31] for i in range(13))


def jetstream_time(text):
    return ts(text).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Fixture:
    def __init__(self):
        self.events, self.snapshots = [], []
        self.netcup_follows, self.netcup_deletions, self.netcup_accounts = [], [], []
        self._seq = 0

    def event(self, src, time, did, kind="commit", rkey="", rev="", op="create", subject="", active=None, seq=None):
        self._seq += 1
        self.events.append(dict(
            source_id=src, seq=self._seq if seq is None else seq, time=jetstream_time(time), kind=kind, did=did,
            collection="app.bsky.graph.follow" if kind == "commit" else "", rkey=rkey, rev=rev,
            operation=op if kind == "commit" else "", created_at=OLD if kind == "commit" else "",
            subject=subject, account_active=active, account_status="" if active in (None, True) else "deactivated",
        ))

    def download(self, did, when, follows, host=HOST, record=True):
        """A completed first download: records at one rev, no sync event."""
        rev = tid(when)
        for rkey, subject in follows:
            self.event("gce_main", when, did, rkey=rkey, rev=rev, subject=subject)
        if record:
            self.snapshots.append(dict(did=did, status="complete", snapshot_rev=rev,
                                       completed_at=jetstream_time(when), latest_rev=rev, host=host, active=1))

    def august(self, did, follows):
        for rkey, subject in follows:
            self.event("public_archive", AUG, did, rkey=rkey, rev=tid(AUG), subject=subject)

    def roster(self, did, status, host=HOST, active=1):
        self.snapshots.append(dict(did=did, status=status, snapshot_rev="", completed_at="",
                                   latest_rev="", host=host, active=active))

    def write(self, root: pathlib.Path):
        (root / "events").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.events, schema=EVENT_SCHEMA), root / "events" / "events-0.parquet")
        (root / "snapshots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.snapshots))
        day = root / "netcup" / "2026-10-20"
        day.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(self.netcup_follows, schema=NETCUP_FOLLOWS), day / "follows.parquet")
        pq.write_table(pa.Table.from_pylist(self.netcup_deletions, schema=NETCUP_DELETIONS), day / "deletions.parquet")
        pq.write_table(pa.Table.from_pylist(self.netcup_accounts, schema=NETCUP_ACCOUNTS), day / "account_status.parquet")


def build():
    """Return (fixture, expected follows per follower DID, expected inactive DIDs)."""
    f = Fixture()
    expected = {}

    # A: August follow absent from the October snapshot is superseded.
    f.august("did:plc:a", [("fa1", "did:plc:pa1"), ("fa2", "did:plc:pa2")])
    f.download("did:plc:a", OCT, [("fa2", "did:plc:pa2"), ("fa3", "did:plc:pa3")])
    expected["did:plc:a"] = {"did:plc:pa2", "did:plc:pa3"}

    # B: August-only account keeps its follows, reported events_only.
    f.august("did:plc:b", [("fb1", "did:plc:pb1")])
    expected["did:plc:b"] = {"did:plc:pb1"}

    # C: live delete after the snapshot applies.
    f.download("did:plc:c", OCT, [("fc1", "did:plc:pc1"), ("fc2", "did:plc:pc2")])
    f.event("gce_main", "2026-10-22T16:00:00Z", "did:plc:c", rkey="fc1", rev=tid("2026-10-22T16:00:00Z"), op="delete")
    expected["did:plc:c"] = {"did:plc:pc2"}

    # D: deactivated after the snapshot; drops as follower and as producer (x follows d).
    f.download("did:plc:d", OCT, [("fd1", "did:plc:pd1"), ("fd2", "did:plc:pd2")])
    f.event("gce_main", "2026-10-18T16:00:00Z", "did:plc:d", kind="account", active=False)
    f.download("did:plc:x", OCT, [("fx1", "did:plc:d"), ("fx2", "did:plc:px2")])
    expected["did:plc:x"] = {"did:plc:px2"}

    # E: delete with a higher rev but a lower seq from another namespace still wins.
    f.download("did:plc:e", OCT, [("fe1", "did:plc:pe1"), ("fe2", "did:plc:pe2")])
    f.event("gce_bootstrap_live", "2026-10-20T02:00:00Z", "did:plc:e", rkey="fe1",
            rev=tid("2026-10-20T02:00:00Z"), op="delete", seq=1)
    expected["did:plc:e"] = {"did:plc:pe2"}

    # F: netcup stop-window delete with a rev applies.
    f.download("did:plc:f", OCT, [("ff1", "did:plc:pf1"), ("ff2", "did:plc:pf2")])
    f.netcup_deletions.append(dict(collection="app.bsky.graph.follow", did="did:plc:f", rkey="ff1",
                                   rev=tid_u64("2026-10-20T12:00:00Z"), seq=7, indexed_at=ts("2026-10-20T12:00:00Z")))
    expected["did:plc:f"] = {"did:plc:pf2"}

    # F2: netcup delete with rev 0 orders by its indexed time and still applies.
    f.download("did:plc:f2", OCT, [("fg1", "did:plc:pf21"), ("fg2", "did:plc:pf22")])
    f.netcup_deletions.append(dict(collection="app.bsky.graph.follow", did="did:plc:f2", rkey="fg1",
                                   rev=0, seq=8, indexed_at=ts("2026-10-20T12:00:00Z")))
    expected["did:plc:f2"] = {"did:plc:pf22"}

    # G: live follow buffered in bootstrap-live, merged into main with the merge time.
    f.download("did:plc:g", OCT, [("fh1", "did:plc:pg1"), ("fh2", "did:plc:pg2")])
    rg = tid("2026-10-20T03:00:00Z")
    f.event("gce_bootstrap_live", "2026-10-20T03:00:00Z", "did:plc:g", rkey="fh3", rev=rg, subject="did:plc:pg3")
    f.event("gce_main", "2026-10-22T13:00:00Z", "did:plc:g", rkey="fh3", rev=rg, subject="did:plc:pg3")
    expected["did:plc:g"] = {"did:plc:pg1", "did:plc:pg2", "did:plc:pg3"}

    # H: August-only account with a merged live follow keeps all three.
    f.august("did:plc:h", [("fi1", "did:plc:ph1"), ("fi2", "did:plc:ph2")])
    f.event("gce_main", "2026-10-22T13:00:01Z", "did:plc:h", rkey="fi3",
            rev=tid("2026-10-20T04:00:00Z"), subject="did:plc:ph3")
    expected["did:plc:h"] = {"did:plc:ph1", "did:plc:ph2", "did:plc:ph3"}

    # I: live follow created before UTC midnight and witnessed after it.
    f.download("did:plc:i", OCT, [("fj1", "did:plc:pi1"), ("fj2", "did:plc:pi2")])
    f.event("gce_main", "2026-10-23T00:00:00.400000Z", "did:plc:i", rkey="fj3",
            rev=tid("2026-10-22T23:59:59.900000Z"), subject="did:plc:pi3")
    expected["did:plc:i"] = {"did:plc:pi1", "did:plc:pi2", "did:plc:pi3"}

    # K: netcup deactivation then reactivation in the same hour leaves the account active.
    f.download("did:plc:k", OCT, [("fk1", "did:plc:pk1")])
    f.netcup_accounts += [dict(did="did:plc:k", seq=9, indexed_at=ts("2026-10-21T10:05:00Z"), active=0, status="deactivated"),
                          dict(did="did:plc:k", seq=10, indexed_at=ts("2026-10-21T10:30:00Z"), active=1, status="")]
    expected["did:plc:k"] = {"did:plc:pk1"}
    # K2: the same sequence seen only in archive account events.
    f.download("did:plc:k2", OCT, [("fk2", "did:plc:pk2")])
    f.event("gce_main", "2026-10-21T10:05:00Z", "did:plc:k2", kind="account", active=False)
    f.event("gce_main", "2026-10-21T10:30:00Z", "did:plc:k2", kind="account", active=True)
    expected["did:plc:k2"] = {"did:plc:pk2"}
    # K3: netcup deactivation alone makes the account inactive.
    f.download("did:plc:k3", OCT, [("fk3", "did:plc:pk3")])
    f.netcup_accounts.append(dict(did="did:plc:k3", seq=11, indexed_at=ts("2026-10-21T11:00:00Z"), active=0, status="deactivated"))

    # L: October snapshot with no follow records supersedes the August follow.
    f.august("did:plc:l", [("fl1", "did:plc:pl1")])
    f.download("did:plc:l", OCT, [])

    # M: follow created during the GCE stop window reaches the fold from netcup.
    f.download("did:plc:m", OCT, [("fm1", "did:plc:pm1")])
    f.netcup_follows.append(dict(did="did:plc:m", rkey="fm2", rev=tid_u64("2026-10-20T15:00:00Z"), is_deleted=0,
                                 rkey_date=dt.date(2026, 10, 20), seq=12, indexed_at=ts("2026-10-20T15:00:00Z"),
                                 subject="did:plc:pm2"))
    expected["did:plc:m"] = {"did:plc:pm1", "did:plc:pm2"}

    # O: netcup's follows table keeps the subject on its deleted row; the delete still wins.
    f.download("did:plc:o", OCT, [("fo1", "did:plc:po1"), ("fo2", "did:plc:po2")])
    f.netcup_follows.append(dict(did="did:plc:o", rkey="fo1", rev=tid_u64("2026-10-20T16:00:00Z"), is_deleted=1,
                                 rkey_date=dt.date(2026, 10, 20), seq=13, indexed_at=ts("2026-10-20T16:00:00Z"),
                                 subject="did:plc:po1"))
    expected["did:plc:o"] = {"did:plc:po2"}

    # N: a resync (sync event) replaces August state without a recorded download.
    f.august("did:plc:n", [("fn1", "did:plc:pn1"), ("fn2", "did:plc:pn2")])
    rn = tid("2026-10-21T09:00:00Z")
    f.event("gce_main", "2026-10-21T09:00:00Z", "did:plc:n", kind="sync", rev=rn)
    f.event("gce_main", "2026-10-21T09:00:00Z", "did:plc:n", rkey="fn2", rev=rn, subject="did:plc:pn2")
    expected["did:plc:n"] = {"did:plc:pn2"}

    # Roster-only accounts: one failed download on a poorly covered host, one unavailable.
    f.download("did:plc:low1", OCT, [("fo1", "did:plc:pa2")], host=LOW_HOST)
    expected["did:plc:low1"] = {"did:plc:pa2"}
    for i in range(9):
        f.roster(f"did:plc:lowmiss{i}", "failed", host=LOW_HOST)
    f.roster("did:plc:gone", "unavailable", active=0)

    inactive = {"did:plc:d", "did:plc:k3", "did:plc:gone"}
    return f, expected, inactive
