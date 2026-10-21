# Continuous retained copy supervisor — separate review draft

The existing bounded runner remains the first live path. This draft supervisor
starts another bounded snapshot after a clean completed batch. It also refreshes
the snapshot and evidence when the runner stops **only** because evidence aged
past five minutes. Any Drive 403/429, unresolved attempt, nonterminal pair,
cooldown, account mismatch, expired exclusion, disk gate, or unknown exit stops
the loop. It never retries an ambiguous pair, deletes a file, or changes the
collector. Capture may report `uploading`; that is a normal GCS state.

## Standing owner envelope

The sole GCE owner must obtain an explicit standing exclusion covering the
intended continuous interval, author a root-owned mode-0600 JSON file, review
its exact bytes and SHA-256, and pin that digest in the service invocation.
The file is an
attestation for writers the VM cannot observe, **not** a proof that remote
browser, Colab, or Netcup state is mechanically monitored. Required fields:

```json
{
  "schema": "catbird.drive.retained-standing-exclusion.v1",
  "owner_reference": "OWNER_REVIEW_REFERENCE",
  "issued_at": 0,
  "account_identity_verified_at": 0,
  "account_id": "VERIFIED_PERMISSION_ID",
  "writer_id": "SOLE_RETAINED_WRITER_ID",
  "authority_db": "/data/gcs-drive-retained-authority/authority.sqlite",
  "oauth_file_sha256": "SHA256_OF_EXISTING_OAUTH_FILE",
  "drive_root": "EXISTING_APP_OWNED_DRIVE_ROOT_ID",
  "bucket": "EXISTING_GCS_BUFFER_BUCKET",
  "manual_colab_hold_until": 0,
  "netcup_nonwriter_hold_until": 0,
  "other_writer_hold_until": 0
}
```

The owner must resolve these placeholders from reviewed deployed configuration
and new scope-specific facts; the timestamps must cover each next 20-minute
pair. The effective exclusion ends at the earliest of the three writer hold
deadlines and one hour after `issued_at`, even if another field claims a later
hold. It maps that earliest limit to a monotonic deadline at launch and passes
the stricter limit, including capture STOP, to the child watchdog. A backward
wall-clock correction cannot extend any reviewed hold. An ordinary expiry
before a Drive child starts is reported as `standing_exclusion_approaching`;
a child timeout while it is running remains an uncertain stop requiring review.
A renewed assertion requires the owner
to stop this supervisor, verify the external and local writer state again,
review new exact envelope bytes, and restart with the new SHA-256 pin. Editing
the file under a running supervisor fails the old pin; it is not a silent
renewal. The sole owner plans a 15-minute heartbeat to recheck external
Netcup and local writer facts; it may renew the one-hour assertion only from
truthful fresh checks, then stop and restart with a new pinned digest. This
heartbeat is an operational owner duty, not a check performed by this code.
If access changes or another writer appears, the owner stops Drive
copying while GCS capture continues. Josh's new "No other uploads" answer is
a current human attestation, not a mechanical future check. His earlier
manual/Colab pause was attached to a bounded test. The
Netcup check found no `RCLONE_REMOTE` at that observation, but its timer was
still enabled. Neither fact alone authorizes or proves a multi-day exclusion.
If the owner cannot establish and commit exact hold end times, this supervisor
must not be deployed. Root ownership and a SHA pin prevent accidental local
replacement; they do not establish the truth of attested remote facts.
The OAuth SHA binds the same existing credential bytes to the account ID that
was independently checked earlier. Each one-pair worker still does a paced
Drive `about.get` check of the actual permission ID before sending media.
If the credential file changes, or any hold expires, the supervisor stops.
No broader OAuth scope is requested.

Immediately after each GCS snapshot, the supervisor checks the exact envelope
bytes and OAuth fingerprint again, observes inactive/disabled legacy GCE
offloader and drain units, scans `/proc` for known local Drive writers, and
reads the shared SQLite stop gates. It then creates new immutable per-pair
evidence files that bind the manifest digest, account, writer, observed time,
owner envelope digest, and local observations. No truth value comes solely
from a command-line flag. The VM cannot itself verify the future manual/Colab
hold or a later Netcup configuration change, so the owner must own and renew
those standing assertions; without them there is no continuous release.

The supervisor uses one canonical ledger-adjacent lock. Its work root is the
reviewed `/data/gcs-drive-retained-continuous` path on the capture filesystem.
It inventories all retained supervisor artifacts before and during a snapshot,
with a 4 GiB cumulative cache ceiling, a 16 MiB reservation for the at-most
nine receipt downloads, and the same 100 GiB shared free-space floor. A single
monotonic STOP is mapped at launch; each cycle checks both clocks so a backward
wall correction cannot lengthen the run. Each bounded snapshot
contains at most eight pairs, at most 2,684,354,560 reserved bytes, and one
20-minute maximum child per pair. The runner retains its 32 GiB cumulative
local-stage ceiling, 100 GiB shared capture restart floor, 0.55-second
account-wide Drive request pacing, separate 32 MiB cumulative GCS mapping
partition, and 2026-10-24 10:10 UTC capture STOP. JSONL status records append
under the supervisor work root; each batch also has its own append-only runner
status file. Empty current-live snapshots are polled at most once per minute.
The supervisor automatically takes another snapshot after a clean batch or an
evidence-age stop with a clean ledger. It does not require capture to be idle.

## Remaining release gates

- This supervisor is an offline prototype. Its 18 synthetic tests pass on Mac;
  independently review source and run target Linux process/parent-death and
  monitored-snapshot smokes before deployment.
  The current bounded runner rollout proceeds separately.
- The current selector scans at most 5,000 receipts from the namespace start.
  It will stop after that bound even when early receipts are complete. Add a
  separately reviewed generation-pinned cursor/page protocol before claiming
  full 2.5 TiB catch-up. A current-live listing is not historical or
  soft-deleted coverage proof.
- The root owner's standing envelope is still required. No Mac-side agent
  may create it from guessed remote-writer or account facts.
- All original GCS generations, local staging, mappings, SQLite ledgers,
  journals, and checkpoints remain retained. Reclaiming any local generated
  cache or exact GCS generation is a separate reviewed retention decision;
  no cleanup refunds the main buffer's cumulative 2.5 TiB admission cap.
