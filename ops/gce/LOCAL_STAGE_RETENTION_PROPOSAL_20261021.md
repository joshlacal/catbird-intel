# Generated transfer-cache retention proposal

The collector is capturing Bluesky into Google Cloud Storage (GCS), while a separate controller copies pinned archive pairs into Drive for later analysis. Each successful transfer currently leaves a temporary payload and receipt on the VM as well as both cloud copies. Those temporary downloads will eventually reach the 32 GiB transfer-stage ceiling, even though capture can continue independently. This proposal is to reclaim only verified generated download copies during clean maintenance windows; it is not authorized yet.

Governing [capture plan](../../../docs/execution-plans/2026-10-21-network-capture-resume.md) and [bounded-window decision](../../../docs/DECISIONS.md#2026-10-21--network-capture--renew-bounded-drive-writer-windows-after-clean-expiry). The workspace [file-safety rule](../../../AGENTS.md#file-safety) says “Don't delete files,” and its Red-tier policy requires approval for deleting user data. Moving the files elsewhere on the same VM would not reclaim filesystem space, so actual removal needs Josh's explicit exception.

## Concrete inventory

The initial reviewed set is **16 files, 2,182,402,035 bytes**, from eight closed retained-copy runs corresponding to source indices 3671–3676 and 3678–3679. It excludes diagnostic 3677. Every selected payload previously passed a complete independent Drive SHA-256 and MD5 readback; payload/receipt source generations, immutable mappings, exact Drive IDs and local hashes passed 31 checks per pair. Receipt bodies were verified locally and by Drive metadata, but were not independently downloaded from Drive in that verification pass.

The subsequent local-only inventory rehashed all sixteen files and recorded exact absolute paths, byte lengths, device/inode identities, modification times, single-link status, and matching manifest/journal/mapping digests. Its receipt is `~/catbird-lake-local/graph-build/out/capture-resume-20261021/retained-local-cache-inventory-20261021T1138Z.json`, SHA-256 `6fe6b9ac3973ff7169bd4fbf15a555a24429b97d5e7fd09d761b4c619ce9a4d7`. No files were removed and no cloud calls occurred during that inventory. The inventory is a concrete proposal, not fresh deletion eligibility.

## Proposed standing permission

Allow the sole capture owner to remove the exact generated payload and receipt files beneath `/data/gcs-drive-retained-<run-id>/stage/<pair-key>/` after each pair meets the following conditions. This would cover the initial inventory and future pairs admitted by the same reviewed retained-copy pipeline, until the unchanged October 24 capture STOP. It does not authorize GCS or Drive deletion, other local directories, or a new spending envelope.

1. Let the controller stop cleanly between transfer windows. Require no active transfer or account claim, all retained runs closed, zero unresolved requests and no cooldown. Do not interrupt an in-flight request to make a cleanup window.
2. Keep the sole-owner maintenance hold throughout remote verification. After those reads, hold a short shared SQLite admission exclusion that every transfer admission honors during final account, pair, journal, inode and hash checks and removal. Any drift stops the operation. Recheck the approval expiry immediately before each file removal.
3. Fetch the exact immutable GCS mapping and verify both pinned GCS source generations still exist. Verify the exact Drive account, folder and file identities and read back each payload and receipt with matching length and digests. Retain both cloud copies.
4. Recheck the two local files against their exact path, hash, device/inode and single-link inventory. Refuse symlinks, unexpected files, unknown names, changed bytes or partial downloads.
5. Bind an approval artifact to that exact inventory and Josh's standing permission. Append durable intent and result records. Preserve original mapping and transfer receipts as creation-time evidence; append the local-cache release state separately. Stop on uncertain removal instead of replaying it blindly.

Preserve all collector data, the authoritative database, bootstrap live files, checkpoints, manifests, transfer journals, immutable mapping records, upload-charge ledgers and release evidence. This removes only expendable download copies; it does not refund upload charges, GCS admission accounting or the 2.5 TiB cumulative capture cap. Filesystem free-space change must be measured separately because capture can allocate new bytes concurrently.

## Current boundary

The GCS collector remains active. The Drive controller stopped on an unresolved HTTP 500 receipt request and cannot enter cleanup maintenance until that fault is reconciled. The cleanup implementation is still under independent review and its release gate remains disabled. No cleanup permission has been granted. Until both review and explicit permission are complete, all copies stay retained; reaching the 32 GiB stage ceiling pauses Drive transfers while GCS capture continues within its existing controls.
