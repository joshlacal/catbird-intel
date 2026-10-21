# Retained cache read-only remote verifier packet

This add-only packet implements the v2 cache inspector's `RemoteVerifier` contract.
It verifies both payload and receipt from their exact GCS generations and exact
Drive file IDs. It performs no upload, GCS mutation, Drive mutation, local file
release, OAuth scope change, or ledger charge. It does **not** enable the
cache-release path (`LOCAL_CACHE_RELEASE_ENABLED` remains false).

Run only at a clean maintenance boundary after the sole Drive writer has
stopped. The owner's normal account, process, and authority preflight remains
required. The adapter reads a single pinned account/root/folder and does not
list Drive siblings; `drive.file` visibility cannot qualify the archive shard.

## One exact review

`review_exact_cache(...)` is the owner-callable library entry point. It runs the
existing dry-run inspector first, reads `index_generation` from that exact run
journal after matching its hash and pair key, constructs `LiveReadOnlyVerifier`,
then calls `inspect_cache` again with the verifier. Pass `evidence_path` in the
dedicated root-owned evidence directory outside `/data/gcs-drive-retained-*`
to append the result. An eligible record is evidence for a
future **generated-local-stage-only** decision, not permission to remove a
file. The v2 inspector still checks closed pair/account authority, source
generations, immutable mapping, full GCS and Drive media, local inode/hash,
and journal stability. A later release must revalidate all of these under its
account-wide admission reservation and exact approval artifact.

The adapter accepts injectable bucket, auth, and HTTP transports so the
offline tests cannot access cloud services. Production defaults use the VM's
existing GCS service identity and the existing private Drive OAuth refresh
grant. Its token refresh uses a fixed OAuth host, rejects redirects, and
requests no new scope. The Drive transport allows only exact HTTPS GETs to
`www.googleapis.com`, rejects redirects, 403/404/429, other non-200 replies,
unexpected encoding, partial media responses, wrong account, parent, IDs,
sizes, observed role-specific MIME types (payload octet-stream, receipt
application/json), and full-stream
hashes. It sends at least 0.60 seconds
between its own Drive GET dispatches. It does not create upload-attempt ledger
rows for read-only requests or infer a fresh quota window.

The adapter's limits per pair are 10 Drive GETs (normally 7), 8 top-level GCS
SDK operations (normally 6), 300 MiB per media file, 1.2 GiB across all four
media streams, 90 seconds per media stream, and 240 seconds for the session.
Each network operation has a 30-second timeout and retries are disabled.
GCS `blob.open` may make several generation-pinned range requests within one
top-level operation; the SDK does not expose an exact range-request counter to
this adapter. A bounded external process-group watchdog is required for any
live invocation because an unexpectedly blocked library callback cannot be
preempted by the monotonic checks alone. The owner must review the target
installed `google-cloud-storage` version's `blob.open` behavior before use.

Source: [Google Cloud Storage Python Blob reference](https://docs.cloud.google.com/python/docs/reference/storage/latest/google.cloud.storage.blob.Blob), [Drive files.get reference](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/get).

## Offline check

Run `python3 -W error::ResourceWarning -m unittest -v
test_gcs_drive_retained_cache_remote` with only synthetic fake transports.
No credential file or network access is required.
