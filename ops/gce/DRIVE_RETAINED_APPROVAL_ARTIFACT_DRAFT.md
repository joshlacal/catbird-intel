# Local cache approval artifact (offline review)

This packet supplies the `ApprovalVerifier` interface required by the separate
cache-release v2 draft. It creates or verifies only a local JSON file. It never
calls GCS or Drive, enables deletion, or changes OAuth credentials. The cache
release gate remains `LOCAL_CACHE_RELEASE_ENABLED = False`.

## Operator boundary

Only the sole capture owner may call `create_owner_artifact_after_human_yes`,
and only after Josh actually approves the reviewed generated-local-cache
policy in this conversation. The code cannot authenticate that conversation:
the owner supplies its exact authorization reference and timestamp as an
operator attestation. An `approved_by` string is not proof of consent. Pin the
SHA-256 of the approved policy and each freshly remote-qualified exact
inventory in the owner decision record.

The owner must first reach a clean maintenance boundary: no active Drive
controller, child, or account claim; all runs closed; zero unresolved attempts
and zero cooldown. Preserve GCS and Drive copies, collector data, journals,
mappings, authority ledgers, and checkpoints. No cleanup is authorized by this
packet.

## Exact owner procedure after human authorization

1. At a clean boundary, obtain a newly remote-qualified inventory for one
   pair. It must include full Drive payload **and receipt** media readback,
   exact GCS generations, immutable mapping, and local inode/hash checks.
   The creator rejects local-only inventories that are not marked eligible,
   GCS-retained, Drive-media-verified and still present.
2. Record the human authorization message reference and timestamp and the
   SHA-256 of the policy Josh approved. Run the following only as root, with
   `/data/catbird-retained-cache-evidence` already root-owned mode `0700`:

   ```python
   from gcs_drive_retained_approval_artifact import (
       OwnerPinnedApproval, create_owner_artifact_after_human_yes)

   path, approval_sha256 = create_owner_artifact_after_human_yes(
       inventory=qualified_inventory,
       authorization_reference=trusted_message_reference,
       authorization_epoch=trusted_message_epoch,
       policy_sha256=approved_policy_sha256,
       valid_until=approved_expiry_epoch,
   )
   # Record path and approval_sha256 in the sole owner's append-only decision log.
   verifier = OwnerPinnedApproval(
       path=path, expected_sha256=approval_sha256,
       inventory=qualified_inventory,
       authorization_reference=trusted_message_reference,
       authorization_epoch=trusted_message_epoch,
       expected_policy_sha256=approved_policy_sha256,
   )
   approval = verifier.read_approval()
   verifier.verify(approval)
   ```

   Creation uses exclusive `O_EXCL | O_NOFOLLOW`, root owner, mode `0600`,
   exact canonical JSON, and file and directory `fsync`. The name is
   `approval-<run>-<unique attempt>.json`. It never replaces or deletes a
   prior artifact; a later reviewed attempt gets a new name, while the exact
   selected path and hash remain pinned for that release. If creation returns
   an uncertain error after opening a file, inspect the evidence directory
   before a new attempt; never assume the earlier file was absent.
3. A future separately reviewed release caller passes that exact `approval`
   and `verifier` to the disabled v2 backend. The backend must continue to
   compare the live inventory hash, path, pair, and scope, hold its account
   admission reservation, and recheck ledger/journal and inode/hash before
   each action. It must also call this verifier again immediately before
   **each** removal to enforce expiry after long remote readbacks; the current
   v2 backend calls it only once. This is a required integration change
   before activation. Do not flip its hard gate in this packet.

The verifier requires a root-owned `0700` evidence directory and a root-owned
`0600` single-link regular artifact, exact pinned bytes, exact inventory and
policy hashes, both stage paths beneath the expected run and pair, a bounded
generated-local-only scope, both cloud copies retained, and an expiry no later
than the existing 2026-10-24 10:10 UTC capture STOP. It rejects an artifact
issued before the owner-attested human authorization time. Repeated checks in
one process also compare monotonic elapsed time so a backward wall correction
cannot extend validity. The owner must use a fresh trusted clock before
maintenance; this verifier cannot prove wall time across a reboot or by
itself enforce the active supervisor's monotonic STOP.

## Limits

This local record binds the owner's reviewed decision; it is not a signature
from Josh or an independent way to verify a chat message. Human consent and
the policy scope must be established by the sole owner. A future release path
must check expiry again at each file action. Without the remote verifier,
the v2 expiry integration, and an explicit yes, leave all
local caches in place and let Drive copying stop at its stage ceiling while
GCS capture continues.
