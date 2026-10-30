"""S7 GCS round trip on the Colab VM, run by `colab exec -f roundtrip_job.py` AFTER Josh has run `colab auth` on
the same session (run_colab.sh prints the command and waits). Never by hand on the Mac except as the local rehearsal.

Uses the credential `colab auth` left on the VM (google.colab.auth.authenticate_user with USE_AUTH_EPHEM=0, i.e.
`gcloud auth login --enable-gdrive-access --no-launch-browser`, copied to /content/.adc/adc.json). That credential is
Josh's, with whatever scopes he granted; the restriction to gs://catbird-436600-plan-30d/colab-test/ is enforced HERE
(s7lib.rt_allowed), not by the credential. Steps:
  1. record the scopes actually granted (oauth2 tokeninfo, token sent in a POST body, never printed or logged);
  2. read the synthetic random-bytes object (no personal data) end to end, hashing it, and compare sha256 + size with
     what the Mac recorded when it wrote the object;
  3. write one small JSON object to colab-test/ (if_generation_match=0: never overwrites), read it back, compare.
Env: S7_RT_OBJECT (name under colab-test/), S7_RT_SHA256, S7_RT_BYTES, S7_SESSION.
Local rehearsal: S7_RT_LOCAL_DIR=<dir standing in for the bucket>, S7_OUT=<dir>; no network, no credential.
Writes $S7_OUT/roundtrip.json (default /content/s7_out) and appends it to /content/s7_rt.tar.
"""
import datetime
import hashlib
import json
import os
import sys
import tarfile
import time

LOCAL = os.environ.get("S7_RT_LOCAL_DIR")
ROOT = os.environ.get("S7_ROOT", "/content/s7") if LOCAL else "/content/s7"
OUT = os.environ.get("S7_OUT", "/content/s7_out") if LOCAL else "/content/s7_out"
sys.path.insert(0, ROOT)
import s7lib  # noqa: E402

ADC = "/content/.adc/adc.json"
PROJECT = "catbird-436600"


def say(*a):
    print("[s7rt]", s7lib.redact(" ".join(str(x) for x in a)), flush=True)


class HashSink:
    def __init__(self):
        self.h, self.n = hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return len(b)


class LocalStore:
    """Rehearsal stand-in for one bucket: a local directory."""

    def __init__(self, root):
        self.root = root

    def read_into(self, name, sink):
        with open(os.path.join(self.root, name), "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                sink.write(b)

    def write_new(self, name, data):
        p = os.path.join(self.root, name)
        if os.path.exists(p):
            raise FileExistsError(name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)

    def read_bytes(self, name):
        return open(os.path.join(self.root, name), "rb").read()


class GcsStore:
    def __init__(self, creds):
        from google.cloud import storage
        self.b = storage.Client(project=PROJECT, credentials=creds).bucket(s7lib.RT_BUCKET)

    def read_into(self, name, sink):
        self.b.blob(name).download_to_file(sink, checksum="crc32c")

    def write_new(self, name, data):
        self.b.blob(name).upload_from_string(data, content_type="application/json", if_generation_match=0)

    def read_bytes(self, name):
        return self.b.blob(name).download_as_bytes()


def credentials(rec):
    if os.path.exists(ADC) and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ADC
    rec["adc_file_present"] = os.path.exists(ADC)
    import google.auth
    import google.auth.transport.requests
    import requests
    creds, proj = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    r = requests.post("https://oauth2.googleapis.com/tokeninfo", data={"access_token": creds.token}, timeout=30)
    info = r.json() if r.ok else {}
    rec["credential"] = {"type": type(creds).__name__, "adc_project": proj, "tokeninfo_http": r.status_code,
                         "scopes": s7lib.scopes_from_tokeninfo(info), "email": info.get("email"),
                         "expires_in_s": info.get("expires_in"), "oauth_client_azp": info.get("azp")}
    return creds


def zone():
    try:
        import requests
        r = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/zone",
                         headers={"Metadata-Flavor": "Google"}, timeout=2)
        return r.text.rsplit("/", 1)[-1] if r.ok else f"http {r.status_code}"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def main():
    os.makedirs(OUT, exist_ok=True)
    obj = os.environ["S7_RT_OBJECT"]
    want_sha, want_n = os.environ["S7_RT_SHA256"], int(os.environ["S7_RT_BYTES"])
    session = os.environ.get("S7_SESSION", "local")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    back = f"{s7lib.RT_PREFIX}roundtrip-{session}-{stamp}.json"
    rec = {"bucket": s7lib.RT_BUCKET, "read_object": obj, "write_object": back, "rehearsal": bool(LOCAL),
           "vm_zone": None if LOCAL else zone(), "ok_read": False, "ok_write": False}
    try:
        for name in (obj, back):
            if not s7lib.rt_allowed(s7lib.RT_BUCKET, name):
                raise SystemExit(f"refusing {name}: outside gs://{s7lib.RT_BUCKET}/{s7lib.RT_PREFIX}")
        store = LocalStore(LOCAL) if LOCAL else GcsStore(credentials(rec))
        if not LOCAL:
            say("scopes granted:", " ".join(rec["credential"]["scopes"]))
        sink, t = HashSink(), time.time()
        store.read_into(obj, sink)
        secs = time.time() - t
        rec["read"] = {"bytes": sink.n, "seconds": round(secs, 2), "mb_per_s": round(sink.n / 1e6 / max(secs, 1e-9), 1),
                       "sha256": sink.h.hexdigest(), "sha256_expected": want_sha, "bytes_expected": want_n}
        rec["ok_read"] = sink.h.hexdigest() == want_sha and sink.n == want_n
        say("read", json.dumps(rec["read"]))
        payload = json.dumps({"s7_roundtrip": True, "session": session, "utc": stamp, "read_sha256": rec["read"]["sha256"],
                              "read_bytes": sink.n, "read_seconds": rec["read"]["seconds"]}, sort_keys=True).encode()
        store.write_new(back, payload)
        rec["ok_write"] = store.read_bytes(back) == payload
        rec["write"] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        say("write", back, rec["ok_write"])
    except SystemExit as e:
        rec["error"] = str(e)
    except Exception as e:  # recorded, redacted; the driver decides on the fallback
        rec["error"] = s7lib.redact(f"{type(e).__name__}: {e}")[:2000]
    rec["ok"] = rec["ok_read"] and rec["ok_write"]
    json.dump(rec, open(os.path.join(OUT, "roundtrip.json"), "w"), indent=1)
    tar = os.path.join(os.path.dirname(OUT), "s7_rt.tar")
    with tarfile.open(tar, "w") as tf:
        tf.add(os.path.join(OUT, "roundtrip.json"), arcname="roundtrip.json")
    say("TAR", tar, s7lib.sha256_file(tar))
    say("ROUNDTRIP_OK" if rec["ok"] else "ROUNDTRIP_FAILED", rec.get("error", ""))


main()
