# Generated retained-stage cache: review packet only

This packet inspects only downloaded duplicate files in
`/data/gcs-drive-retained-<run>/stage/<pair>/`. It does not alter the collector,
GCS originals, Drive files, immutable mappings, checkpoints, manifests,
journals, SQLite authority, or status logs. `LOCAL_CACHE_RELEASE_ENABLED=False`
is a hard stop in the shipped module; no cleanup is authorized or deployed.

## What can be inventoried now

`gcs_drive_retained_cache_release.py` has a local-only dry-run CLI. On the
owner's host, with a **closed pair's exact launch manifest** and the existing
read-only SQLite account/writer identity, the owner can call:

```text
python3 gcs_drive_retained_cache_release.py \
  --manifest <exact-retained-pair-manifest> \
  --run-id <32-hex-run-id> \
  --account-id <pinned-drive-permission-id> \
  --writer-id <existing-ledger-writer-id> \
  --authority-db /data/gcs-drive-retained-authority/authority.sqlite \
  --data-root /data
```

It hashes both exact generated stage files through `O_NOFOLLOW` descriptors,
checks same-device single-link inode/size/time identity, matches the immutable
source manifest and terminal journal, and reads the closed authority pair and
zero-unresolved account state. It appends and fsyncs a per-run JSONL evidence
record at `/data/catbird-retained-cache-evidence/review-<run>.jsonl`, outside
the runner's `gcs-drive-retained-*` staging scan. The output is
always `eligible_for_approval=false` with `remote_readback_unavailable` because
this CLI performs **no** cloud call. Thus it supplies exact local paths, bytes,
inodes, and hashes for review, but it does not authorize release.

The seven-pair independent reports already demonstrate 7/7 metadata checks
and complete **payload** Drive media readback. Those reports say receipt bodies
were **not** independently read back; they also do not bind the current local
stage inode. They are useful provisional context, not a release inventory.
Do not infer eligibility or exact reclaimable bytes from the reports alone.

## Remote qualification still needed

A separately reviewed read-only `RemoteVerifier` adapter must, for each pair:

1. Match the same Drive permission/account identity.
2. Read the exact immutable GCS mapping **generation and body**, checking its
   canonical bytes, SHA-256, source generations, Drive IDs, and historic
   `retained_gcs=true`, `retained_local=true`, `archive_qualified=false` fields.
3. Stream **both** exact pinned GCS payload and receipt generations to EOF and
   match size, SHA-256, MD5.
4. Stream **both** exact Drive payload and receipt IDs to EOF, match size,
   SHA-256, MD5, account, parent folder, and untrashed state. Preserve the
   shared account request pacing and bounded read costs.

`inspect_cache(..., verifier=adapter)` then returns
`eligible_for_approval=true` only if all local, ledger, mapping, GCS and Drive
checks agree. That value means *candidate for Josh's decision*, not deletion
authority. No production adapter is included, so this packet cannot yet
produce a remote-qualified live inventory. A separate owner-controlled adapter
and an independently authenticated approval record are the exact remaining
implementation gates.

## Future release transaction, currently disabled

`release_after_approval` contains a separately reviewable transaction behind
the hard stop. Before enabling it, the sole owner must supply a hash-pinned
inventory and a trusted `ApprovalVerifier` that authenticates Josh's actual
approval reference, approved inventory SHA-256, and exact two paths. The
`approved_by: Josh` string in JSON is **not** authentication. The owner's
review must verify no noncooperating process can mutate the stage tree.

The transaction takes the retained worker's existing nonblocking
`overflow.lock`, authenticates approval, and performs a fresh full remote
readback with a five-minute elapsed-time limit **before** reserving the account
ledger. The reviewed remote adapter must itself have bounded request timeouts;
Python cannot safely interrupt a hung network callback. The transaction then
takes SQLite `BEGIN IMMEDIATE` on the shared authority database. Every current
retained admission and request mutation uses that database, including a direct
single-pair worker that does not hold the bounded runner's `.runner.lock`.
While holding the database writer reservation, it rejects any prior release
intent for that run, rechecks the closed pair, whole-account unresolved and
nonterminal state, and journal digest, pins the stage directory descriptor,
and writes and
fsyncs an append-only start record. For each exact file it writes and fsyncs
an intent, rechecks the ledger and journal again, reopens relative to the pinned
directory with `O_NOFOLLOW`, verifies
device/inode/link count/size/times and full SHA-256+MD5, then removes only that
basename relative to the directory descriptor. It fsyncs the stage directory
and writes a per-file removed or uncertain result. An uncertain outcome stops
without automatic replay. It never removes the stage directory or any
authority, mapping, source, checkpoint, journal, manifest, or unique collector
file. POSIX does not offer an atomic unlink-by-name-and-inode primitive; the
flock covers cooperating retained/recovery processes, and the owner must
exclude other stage mutators before any future enablement. The SQLite
reservation covers cooperating writers to this one canonical ledger; the owner
must separately exclude any account writer outside it. The evidence directory
is private to the runtime owner, and its creation and append records are
fsynced. A stale remote preflight is never treated as current admission proof:
any run still active or unresolved when the account reservation is acquired
blocks release.

The immutable mapping's `retained_local=true` remains a truthful
**creation-time receipt**. A later local cache release would be a separate
append-only event with its own evidence. GCS originals and Drive copies stay
retained. Reclaiming generated local bytes does not refund the active GCS
buffer's cumulative 2.5 TiB admission cap or the additive 32 MiB mapping
partition. GCS payload/receipt deletion is a different decision and is absent
from this code.

## Review status

Synthetic offline tests exercise local and remote agreement, symlink/hardlink
and digest mismatch refusal, durable intent followed by a mocked uncertain
unlink, no automatic replay, retained-worker lock exclusion, and the shipped
hard stop. They use no credentials, cloud calls, real transfers, or file
deletion. Code must remain disabled until Josh explicitly approves a concrete
hash-pinned inventory under the repository's Red-tier deletion rule and the
owner separately reviews the remote/approval adapters and live process
exclusion. Neither approval nor adapter is present in this packet.
