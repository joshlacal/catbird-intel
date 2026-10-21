# Retained receipt recovery transport (offline review)

This add-only module supplies request syntax for the one pinned 1,093-byte
receipt after a final media PUT returned HTTP 500. It has no CLI, OAuth
loader, retry loop, or cloud client. The controller must inject the existing
`SafeDrive.request` bound to the shared retained authority, under a reviewed
exclusive writer lease, so **every** zero-body status query, resumable start,
and media PUT has a new durable attempt and obeys request pacing. The original
HTTP 500 attempt and its full 1,093-byte charge remain unchanged.

The first step validates the original private session against its durably
recorded SHA-256 fingerprint, then makes exactly one empty PUT to it with
`Content-Range: bytes */1093`. Results:

| Result | Transport output | Controller action |
| --- | --- | --- |
| 200/201 with pinned file ID | `complete` | Independently read exact-ID metadata and entire receipt media; verify account, parent, size, SHA-256 and MD5 before any authority reconciliation. |
| 308 with valid `Range` | `partial` and next offset | Review the saved evidence before sending one exact unsent suffix to the **same** session. No automatic loop. |
| 308 without `Range` | `partial`, offset zero | Review before sending the full receipt once to the same session. |
| 404 or 410 | `expired` session only | Recheck the exact pinned file ID. If absent and the run is still authorized, a separate reviewed step may start one new session with the **same** pre-generated receipt ID. A 409 remains uncertain pending exact-ID verification. |
| Other 4xx, 5xx, timeout, malformed response | error | Keep the pair blocked; no media is sent. |

`resume` checks the private session fingerprint and reads and hashes the entire locally retained receipt through
`O_NOFOLLOW`, confirms inode and size did not change, and sends exactly
the pinned SHA-256 and MD5, then sends exactly `bytes offset-1092/1093` once. A 308 response stops with its acknowledged
offset; it does not initiate another PUT. `restart_same_id` sends only pinned
receipt metadata and returns the private session URL to the caller; the caller
must durably save it in the existing private journal **before** any media PUT.
Neither method touches the payload, folder, source objects, or earlier attempt.

The implementation has no authority to perform the status query by itself.
Before invocation, the owner must verify the exact manifest, GCS generation,
local receipt bytes, Drive account/root/folder/IDs, old attempt identity,
remaining charged-byte reservation, current lease and absolute capture STOP.
The controller must preserve append-only forensic evidence and keep raw HTTP
500 and any later non-2xx attempts charged. A later full Drive media readback,
not metadata or a status code alone, is needed to complete the pair.

Google documents the empty status PUT, 308 Range semantics, 200/201 completion,
404 session expiry, and safe retry with a pre-generated file ID in
[Upload file data](https://developers.google.com/workspace/drive/api/guides/manage-uploads).
