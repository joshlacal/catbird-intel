"""Mac-side ledger for S12m on Colab: which work units are done, and how many Colab units the run has spent.

The Mac is the source of truth. A Colab VM, its disk and everything uploaded to it vanish when a session ends
(after 12-24 h, on a drop, or at `colab stop`), so nothing is "done" until its vectors are on the Mac:

  RUN_DIR/ledger/<unit>.json      written (atomically) only after the unit's output parquet was downloaded, its
                                  sha256 matched the VM's own ledger record, its row count matched the input
                                  manifest and it passed mmlib.audit_output_table on the Mac.
  RUN_DIR/vectors/units/u-<unit>-00000.parquet   the verified vectors (moved in before the marker is written).
  RUN_DIR/sessions.tsv            one line per session event: created / stopped / dropped / stopped(trap) ...
                                  The driver only ever stops sessions recorded here for this run.
  RUN_DIR/spend/<session>.json    t_created, t_last_seen (heartbeat), t_stopped, units/h, `colab usage` balances.
  RUN_DIR/ledger/claims/<unit>.json   a lease {owner, expires}: parallel sessions (--sessions N) each take disjoint
                                  units by claiming them under an flock'd lock; a lease that expires (a dropped
                                  session, a killed driver) puts its units back in the pool, and a lane that sees
                                  its session drop releases its leases at once. Marking a unit done removes its lease.

Spend = sum over sessions of (t_stopped, else t_last_seen + grace) - t_created, x units/h. The guard uses the
larger of that and the account balance drop since the run's first session (account-wide, so it over-counts when
another Colab job runs at the same time: conservative). Stdlib only, except `verify_unit` (pyarrow, numpy).
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time

GRACE_S = 900
_THREAD_LOCKS: dict = {}


def atomic_write(path, text):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def sha256_file(p, block=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def parse_usage(text):
    """`colab usage` output -> {balance, rate_per_hour, active} (None when a field is missing)."""
    def grab(rx, cast):
        m = re.search(rx, text or "")
        return cast(m.group(1)) if m else None
    return {"balance": grab(r"Current balance:\s*([0-9.]+)", float),
            "rate_per_hour": grab(r"Usage rate:\s*([0-9.]+)", float),
            "active": grab(r"Active assignments:\s*([0-9]+)", int)}


def output_name(unit):
    """job.py names a one-file unit's output after the file stem plus its slice number."""
    return f"u-{unit}-00000"


class RunLedger:
    def __init__(self, run_dir, clock=time.time):
        self.dir = pathlib.Path(run_dir)
        self.clock = clock
        for d in ("ledger", "vectors/units", "vectors/receipt", "spend", "incoming", "logs"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.tsv = self.dir / "sessions.tsv"
        (self.dir / "ledger" / "claims").mkdir(exist_ok=True)
        self._tlock = _THREAD_LOCKS.setdefault(str(self.dir.resolve()), threading.RLock())

    # --- leases -----------------------------------------------------------------------------------------------
    class _Locked:
        def __init__(self, led):
            self.led = led

        def __enter__(self):
            self.led._tlock.acquire()
            self.fh = open(self.led.dir / "ledger" / "claims.lock", "a")
            fcntl.flock(self.fh, fcntl.LOCK_EX)     # also excludes a second driver process on this run dir
            return self

        def __exit__(self, *exc):
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.led._tlock.release()

    def _claim_path(self, unit):
        return self.dir / "ledger" / "claims" / f"{unit}.json"

    def claims(self, now=None):
        """{unit: lease} of leases that have not expired."""
        now = self.clock() if now is None else now
        out = {}
        for p in (self.dir / "ledger" / "claims").glob("*.json"):
            try:
                c = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if c.get("expires", 0) > now:
                out[p.stem] = c
        return out

    def claim(self, owner, manifest, n, ttl_s):
        """Lease up to n units that are neither done nor under a live lease (anyone's, including owner's own
        in-flight batches), in manifest order."""
        with self._Locked(self):
            now, done, live = self.clock(), self.done(), self.claims()
            got = []
            for u in manifest["units"]:
                if len(got) >= n:
                    break
                if u["unit"] in done or u["unit"] in live:
                    continue
                atomic_write(self._claim_path(u["unit"]), json.dumps({"owner": owner, "t": now, "expires": now + ttl_s}))
                got.append(u)
            return got

    def renew(self, owner, units, ttl_s):
        with self._Locked(self):
            now = self.clock()
            for u in units:
                p = self._claim_path(u)
                try:
                    c = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if c.get("owner") == owner:
                    atomic_write(p, json.dumps({**c, "expires": now + ttl_s}))

    def release(self, owner, units=None):
        """Drop owner's leases (all of them, or those on `units`)."""
        with self._Locked(self):
            want = None if units is None else set(units)
            n = 0
            for p in (self.dir / "ledger" / "claims").glob("*.json"):
                if want is not None and p.stem not in want:
                    continue
                try:
                    c = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if c.get("owner") == owner:
                    p.unlink(missing_ok=True)
                    n += 1
            return n

    def unclaimed_todo(self, manifest):
        live = self.claims()
        return [u for u in self.todo(manifest) if u["unit"] not in live]

    # --- units ------------------------------------------------------------------------------------------------
    def done(self) -> dict:
        out = {}
        for p in sorted((self.dir / "ledger").glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue   # a marker is written atomically; anything unreadable is not a marker
        return out

    def is_done(self, unit) -> bool:
        return (self.dir / "ledger" / f"{unit}.json").exists()

    def todo(self, manifest, exclude=()):
        d, ex = self.done(), set(exclude)
        return [u for u in manifest["units"] if u["unit"] not in d and u["unit"] not in ex]

    def mark(self, unit, rec):
        for k in ("sha256", "rows", "file"):
            if k not in rec:
                raise ValueError(f"ledger record for {unit} lacks {k}")
        atomic_write(self.dir / "ledger" / f"{unit}.json", json.dumps({"unit": unit, **rec}, sort_keys=True))

    def verify_unit(self, unit_meta, vm_rec, local_file, audit=None):
        """Check a downloaded output against the VM's record and the input manifest; on success move it into
        vectors/units/ and write the marker. Returns (ok, reason)."""
        local_file = pathlib.Path(local_file)
        if not local_file.exists():
            return False, "missing"
        got = sha256_file(local_file)
        if got != vm_rec.get("sha256"):
            return False, f"sha256 {got[:12]} != VM {str(vm_rec.get('sha256'))[:12]}"
        if vm_rec.get("rows") != unit_meta["rows"]:
            return False, f"rows {vm_rec.get('rows')} != manifest {unit_meta['rows']}"
        if audit is not None:
            chk = audit(local_file)
            if not chk.get("ok") or chk.get("rows") != unit_meta["rows"]:
                return False, f"audit failed: {chk}"
        dst = self.dir / "vectors" / "units" / f"{output_name(unit_meta['unit'])}.parquet"
        shutil.move(str(local_file), dst)
        self.mark(unit_meta["unit"], {"sha256": got, "rows": unit_meta["rows"], "file": str(dst.relative_to(self.dir)),
                                      "input_sha256": unit_meta.get("sha256"), "vm": vm_rec,
                                      "verified_at": self.clock()})
        self._claim_path(unit_meta["unit"]).unlink(missing_ok=True)
        return True, None

    # --- sessions and spend -------------------------------------------------------------------------------------
    def record(self, session, event, gpu="-", note=""):
        with open(self.tsv, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.clock()))}\t{session}\t{gpu}\t{event}\t{note}\n")

    def sessions(self):
        """{session: [events]} from sessions.tsv, in order."""
        out = {}
        if self.tsv.exists():
            for ln in self.tsv.read_text().splitlines():
                parts = ln.split("\t")
                if len(parts) >= 4:
                    out.setdefault(parts[1], []).append(parts[3])
        return out

    def open_sessions(self):
        """Sessions this run created whose end was never recorded (a driver crash or kill -9)."""
        return [s for s, ev in self.sessions().items() if "created" in ev and not any(
            e.startswith(("stopped", "dropped", "ended")) for e in ev)]

    def spend_path(self, session):
        return self.dir / "spend" / f"{session}.json"

    def spend_update(self, session, **fields):
        p = self.spend_path(session)
        cur = json.loads(p.read_text()) if p.exists() else {"session": session}
        cur.update(fields)
        atomic_write(p, json.dumps(cur, indent=1, sort_keys=True))
        return cur

    def spend_records(self):
        return [json.loads(p.read_text()) for p in sorted((self.dir / "spend").glob("*.json"))]

    def session_units(self, rec, now=None):
        now = self.clock() if now is None else now
        t0 = rec.get("t_created")
        if t0 is None:
            return 0.0
        end = rec.get("t_stopped")
        if end is None:
            seen = rec.get("t_last_seen", t0)
            end = min(now, seen + GRACE_S) if rec.get("closed_unobserved") else now
        return max(0.0, end - t0) / 3600 * float(rec.get("rate", 0.0))

    def spent_units(self, now=None, balance_now=None):
        """(guard value, details): max(wall estimate, balance drop since the first session)."""
        recs = self.spend_records()
        wall = sum(self.session_units(r, now) for r in recs)
        b0 = next((r["balance_before"] for r in sorted(recs, key=lambda r: r.get("t_created", 0))
                   if r.get("balance_before") is not None), None)
        drop = round(b0 - balance_now, 3) if (b0 is not None and balance_now is not None) else None
        guard = max(wall, drop or 0.0)
        return guard, {"wall_estimate_units": round(wall, 3), "balance_drop_units": drop, "guard_units": round(guard, 3),
                       "sessions": len(recs)}
