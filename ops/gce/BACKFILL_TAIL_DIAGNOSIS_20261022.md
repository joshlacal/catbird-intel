# Jetstream Self-Host Backfill Tail Diagnosis & Stop-Extension Proposal (`2026-10-22`)

## Cold-Reader Summary

We are running a self-hosted Bluesky Jetstream `v0.4.4` collector (`commit d4db8c08eb887dd148cd7c1d5f4a9fb327e15f06`) on Google Compute Engine (`jetstream-selfhost-test`, an `n2-highmem-8` Spot VM in `us-west1-b`) that downloads full AT Protocol repositories from Personal Data Servers (PDSs) and streams sealed archive segments into Google Cloud Storage (`gs://catbird-436600-jetstream-buffer-20261020`) via a local supervisor (`catbird-gcs-buffer.service`, governed by a hard `2026-10-24T10:10:00Z` power-off deadline and a `2.5 TiB` / `2,748,779,069,440`-byte admission cap). Between `~22:00Z` on Oct 21 and `12:49:15Z` on Oct 22, `bootstrap` (initial direct-PDS enumeration and download) appeared stuck at `~35k repos/hr` with `72 in flight` hosts because 71 of those 72 hosts were `status: "banned"` PDS entries left permanently in `State: "pending"` in Pebble metadata, leaving only **1 active host** (`jellybaby.us-east.host.bsky.network`) running under `--backfill-host-workers-max=2` while 62 of 64 global download slots sat idle. At **`2026-10-22T12:49:24Z`**, `jellybaby.us-east` finished its bootstrap attempt and the collector cut over to Phase **`merging`** (the cutover phase that replays buffered firehose segments, downloads repos left in `StatusPending` by earlier restarts via `runPendingRepoRetryPass`, and re-scans `listRepos` tails via `runMergeDiscovery`), jumping from **`~35k repos/hr` to `672,482 repos/hr` (`186.8 repos/sec`)**. We also resolved the apparent "4.41M pending/unattempted" gap (`3,169,794` discovered accounts have `active: false`—deactivated, suspended, or taken down—which `atmos` and `jetstream` intentionally never fetch via `getRepo`), proved why 18 `*.host.bsky.network` shards stopped early during `bootstrap` (`listRepos: xrpc 429` combined with a 1-minute fixed host backoff against a 5-minute PDS rate-limit window), and identified three `steady_state` (`RunFailedRepoRetry`) bottlenecks that require a non-destructive container flag update (`--failed-repo-retry-workers=64 --failed-repo-retry-interval=10m`) once `merging` finishes at `~2026-10-22T15:10Z` so the remaining `~6.86M` `StatusFailed` + unenumerated active repos finish well before the existing `2026-10-24T10:10:00Z` deadline without needing a deadline extension.

---

## 1. Live Collector Telemetry & Real-Time Cutover (`bootstrap` -> `merging`)

### 1.1 Exact Container, Supervisor, and Disk State on `jetstream-selfhost-test`

- **Container (`docker ps --no-trunc` at `2026-10-22T13:11:39Z`):**
  ```text
  552acc76374fd22e4374abaa37d6f258052a1f443c040ae506ef502dd44663be
  sha256:bb779a4831afab96eea1cf399469619772ca8eed69db6f3a5202ae858d393745
  "/usr/local/bin/jetstream serve --data-dir=/data --backfill-global-downloads=64 --backfill-host-workers-max=2 --backfill-max-active-hosts=32 --backfill-async-flush-workers=8 --failed-repo-retry-host-workers=1 --pending-repo-pass-workers=32 --shutdown-timeout=60s --compaction-interval=0"
  Up 18 hours (started 2026-10-21T19:07:03Z) | Container IP: 172.17.0.2:8080
  ```
- **GCS Buffer Supervisor (`systemctl status catbird-gcs-buffer.service` at `2026-10-22T13:33:32Z`):**
  ```json
  {"reason": "ready", "updated_at_unix": 1792676012.425015, "pending_sealed": 0, "run_id": "6a833795ab4144058ee53cc1e1d8f9e9", "current_index": null, "local_segments": 3, "cloud_inventory_bytes": 593331446173, "cumulative_admitted_bytes": 593600102384, "admission_accounting": "cumulative_observed_generations_no_refund", "max_bytes": 2748779069440, "durable_pairs": 2198, "local_releases": 2196, "source_epoch": "buffer-admission-20261020", "epoch_semantics": "admission_namespace_only"}
  ```
- **Disk Usage (`df -h /data` at `2026-10-22T13:33:31Z`):**
  ```text
  Filesystem      Size  Used Avail Use% Mounted on
  /dev/sda1       197G   42G  147G  23% /
  ```

### 1.2 Why `bootstrap` Throughput Collapsed to `~35k repos/hr` and What the "72 in flight" Hosts Actually Were

1. **Why `/status?tab=summary` showed `72 in flight` (at `11:27Z`) and `71 in flight` (after `12:49:15Z`):**
   - In `atmos` `v0.7.3` ([`backfill/engine.go:315-345`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/atmos_v073_engine.go#L315-L345)), `recordHosts` calls `e.opts.Store.OnHost(ctx, info)` for **every** validated host returned by `com.atproto.sync.listHosts` *before* checking `if info.RelayStatus == "banned" && !e.opts.IncludeBannedHosts.ValOr(false) { continue }` at line 343.
   - `Store.OnHost` ([`internal/ingest/backfill/store.go:1421-1458`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/store.go#L1421-L1458)) creates a `pdshost/<hostname>` key in Pebble with `State: HostStatePending` (`"pending"`). Because `engine.go:343` skips banned hosts without transitioning their `pdshost/*` state to `"drained"` or `"exhausted"`, all **71 relay-banned hosts** (`9,488` total `accountCount` in `com.atproto.sync.listHosts`) remain in `State: "pending"` forever.
   - In [`internal/status/collect.go:184-191`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/jetstream_status_collect.go#L184-L191) (`collectBackfillFleet`), `stats.HostsInFlight` is **not** a gauge of active worker goroutines; it counts every `pdshost/*` row whose `State` is neither `"drained"` nor `"exhausted"`:
     ```go
     switch h.State {
     case backfill.HostStateDrained:
         stats.HostsDrained++
     case backfill.HostStateExhausted:
         stats.HostsExhausted++
     default:
         stats.HostsInFlight++
     }
     ```
   - Therefore, at `11:27Z`, **71 of the 72 "in flight" hosts were the 71 skipped `status: "banned"` hosts**, and **only 1 host (`jellybaby.us-east.host.bsky.network`) was actually running**!
2. **Why `jellybaby.us-east.host.bsky.network` ran alone for 7.35 hours (`05:28:16Z` -> `12:49:15Z`):**
   - By `2026-10-22T05:28:16Z` (when `brittlegill.us-west.host.bsky.network` exhausted), 88 of the 89 active `*.host.bsky.network` PDS shards had already reached terminal `bootstrap` states (`71` `drained`, `17` `exhausted`), leaving only `jellybaby.us-east.host.bsky.network` (`1,143,602` accounts on the shard).
   - Because `--backfill-host-workers-max=2` limits each PDS host to at most **2 concurrent `getRepo` workers**, `jellybaby.us-east` could only use **2 of the 64 `--backfill-global-downloads` slots** (leaving 62 global slots idle), downloading at `~9.8 repos/sec` (`~35,300 repos/hr`) until `12:49:15Z`.

### 1.3 Exact Cutover Timeline (`docker logs js`) and 5-Sample Live Time Series (`S1`–`S5`)

From `docker logs js` on `jetstream-selfhost-test`:
```json
{"time":"2026-10-22T12:49:15.821810022Z","level":"WARN","msg":"PDS host exhausted","component":"backfill/run","host":"jellybaby.us-east.host.bsky.network","attempts":2,"err":"backfill: host jellybaby.us-east.host.bsky.network listRepos: xrpc 429"}
{"time":"2026-10-22T12:49:24.648549184Z","level":"INFO","msg":"PDS-direct backfill drained","component":"backfill/run","limited":false}
{"time":"2026-10-22T12:49:24.648568739Z","level":"INFO","msg":"cutover begin","component":"orchestrator"}
{"time":"2026-10-22T13:02:23.621605848Z","level":"INFO","msg":"starting failed repo retry pass","component":"orchestrator","workers":32,"host_workers":1,"pending_pass":true}
{"time":"2026-10-22T13:05:58.814368075Z","level":"INFO","msg":"failed repo retry pass collected candidates","component":"orchestrator","candidates":1246960,"hosts":484}
```

| Metric | Prior (`11:27Z`) | `S1` (`12:57:02Z`) | `S2` (`13:05:25Z`) | `S3` (`13:11:58Z`) | `S4` (`13:16:42Z`) | `S5` (`13:33:31Z`) | Delta `S2 -> S5` (`27.54m` of `pending_pass`) |
|---|---|---|---|---|---|---|---|
| **Phase / Sub-step** | `bootstrap` (1 host) | `merging` (`live_segments` replay) | `merging` (`scanDue` Pebble scan) | `merging` (`pending_pass` dl) | `merging` (`pending_pass` dl) | `merging` (`pending_pass` dl) | Active `pending_pass` (`workers=32, host_workers=1`) |
| **Discovered (`counts.Total`)** | `42,089,129` | `42,154,129` | `42,154,129` | `42,154,129` | `42,154,129` | `42,154,129` | `+0` (`listRepos` paused until `runMergeDiscovery`) |
| **Downloaded (`Complete`)** | `33,876,683` | `33,932,665` | `33,932,665` | `33,974,360` | `34,022,017` | **`34,210,261`** | **`+277,596` (`+604,784/hr` overall; `+671,625/hr` in `S4->S5`)** |
| **Pending (`StatusPending`)** | `1,395,263` | `1,395,267` | `1,395,267` | `1,353,131` | `1,305,001` | **`1,116,517`** | **`-278,750` (`-672,482/hr` in `S4->S5`)** |
| **Active Pending Remaining** | `1,246,956` | `1,246,960` | `1,246,960` | `1,204,824` | `1,156,694` | **`968,210`** | **`22.35%` of active pending queue drained in `27.5m`** |
| **Errored (`StatusFailed`)** | `3,811,972` | `3,811,972` | `3,811,972` | `3,812,410` | `3,812,877` | **`3,813,113`** | **`+1,141` (`0.41%` overall; `0.125%` in `S4->S5`)** |
| **Unavailable (`StatusUnavailable`)** | `36` | `36` | `36` | `39` | `45` | **`49`** | `+13` |
| **Inferred `StatusNotStarted`** | `3,005,175` | `3,014,189` | `3,014,189` | `3,014,189` | `3,014,189` | **`3,014,189`** | `+0` (inactive accounts; see §1.4) |
| **Archive Events** | `~801M` | `864,313,947` | `900,558,616` | `945,729,264` | `987,222,699` | **`1,151,117,959`** | **`+250,559,343` events (`+9.75M events/min` in `S4->S5`)** |
| **GCS `durable_pairs` / Admitted** | `2,088` / `563.87 GB` | `~2,102` | `~2,125` | `2,136` / `576.94 GB` | `2,150` / `580.70 GB` | **`2,198` / `593.60 GB`** | **`+73` segments (`2.85 segs/min`, `+16.66 GB` in `22m`)** |

### 1.4 Resolution of the "4.41M Pending/Unattempted" Mystery (`StatusNotStarted` vs `StatusPending` vs `Active`)

Summing all `3,349` host rows in `/status?tab=hosts` at cutover (`S1`, [`parsed_hosts_s1.json`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/parsed_hosts_s1.json)):
- `Sum(Repos)` (`counts.Total` / `Discovered`): **`42,154,129`**
- `Sum(Active)` (`rs.Active == true`): **`38,984,335`**
- `Sum(Repos - Active)` (`rs.Active == false`, i.e., deactivated, suspended, or taken-down accounts returned by `com.atproto.sync.listRepos`): **`3,169,794`**
- `Sum(Complete)`: **`33,932,665`**
- `Sum(Failed)`: **`3,811,972`**
- `Sum(Unavailable)`: **`36`**
- `Sum(Repos - Complete - Failed - Unavailable)`: **`4,409,456`**, which decomposes into:
  - `1,395,267` `StatusPending` (`counts.Pending`)
  - `3,014,189` `StatusNotStarted` (`counts.Discovered`)

**Why `3,014,189` rows sit in `StatusNotStarted` and `148,307` `StatusPending` rows were skipped by `runPendingRepoRetryPass`:**
1. In `atmos` `v0.7.3` ([`backfill/engine.go:624-654`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/atmos_v073_engine.go#L624-L654)), `reconcile` calls `e.opts.Store.OnDiscover` for **every** new DID on a `listRepos` page (including `entry.Active == false`), writing `Backfill.Status = StatusNotStarted` ([`internal/ingest/backfill/store.go:1173`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/store.go#L1173)).
2. Immediately afterward, `dispatchable(entry, rec)` ([`engine.go:644`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/atmos_v073_engine.go#L644)) checks `entry.Active && rec.State != StateComplete`. When `entry.Active == false`, `atmos` **never dispatches `getRepo`** and leaves the row in `StatusNotStarted` (or `StatusPending` if an already-enumerated batch was re-scanned after a container restart via `deferInterruptedBootstrapRepos`, [`store.go:311-323`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/store.go#L311-L323)).
3. In `runPendingRepoRetryPass` ([`internal/ingest/backfill/retry.go:411`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/retry.go#L411)), `scanDue` filters out inactive accounts:
   ```go
   if !eligibleStatus(rs.Backfill.Status) || !rs.Active {
       continue
   }
   ```
   Out of `1,395,267` `StatusPending` rows in Pebble at `13:02:23Z`, `scanDue` collected **`1,246,960` active pending candidates** (`1,395,267 - 1,246,960 = 148,307` inactive `StatusPending` rows). Together with the `3,014,189` `StatusNotStarted` rows, `3,014,189 + 148,307 = 3,162,496` rows are `active: false` accounts (matching `Sum(Repos - Active) = 3,169,794` within `7,298` accounts whose `active` flag flipped during bootstrap). **None of the `3,014,189` `StatusNotStarted` rows or `148,307` inactive `StatusPending` rows are downloadable work.**

### 1.5 Per-Host Table: All 90 `*.host.bsky.network` Shards Ranked by Remaining Active Work (`S5`, `2026-10-22T13:33:31Z`)

- **Fleet Totals Across All 3,351 Host Buckets in Pebble at `S5` (`13:33:31Z`) + Relay Roster (`6,638` hosts):**
  - **90 `*.host.bsky.network` shards (`27` `us-east`, `62` `us-west`, `spaces-alpha`, `vlpds`):** `41,761,157` enumerated (`38,662,112` active, `3,099,045` inactive), `33,962,030` complete (`+190,945` in `S1->S5`), `888,083` active pending remaining, `3,811,751` `StatusFailed`, and `3,701,018` unenumerated shortfall vs `relay_floor` (`~3,034,530` estimated active unenumerated) -> **`7,734,364` total remaining active repos (`888,083` in current `pending_pass` + `6,846,281` for `steady_state` retry)**.
  - **3,261 Custom PDS host buckets in Pebble:** `392,972` enumerated (`322,223` active, `70,749` inactive), `248,231` complete (`+86,651` in `S1->S5`), `80,127` active pending remaining, `1,362` `StatusFailed`.

| Rank | Host (`*.host.bsky.network`) | Bootstrap State (`Exhausted UTC`) | Relay Floor (`listHosts`) | Enumerated (`Repos`) | Active | Inactive (`NotStarted`/Inact) | Complete at `S5` (`S1->S5` `27.5m`) | `pending_pass` Rate | Active Pending (`S5`) | Errored (`Failed` `S5`) | Unenumerated Shortfall (`~Active`) | **Total Remaining Active Work** | Latest Error Class |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `chalciporus.us-west.host.bsky.network` | exhausted (2026-10-21T19:53:38Z) | 759,421 | 184,972 | 149,451 | 35,521 | 141,897 (+2,701) | 5,885/h | 4,784 | 2,770 | 574,449 (~464,135 act) | **471,689** | `car` |
| 2 | `stropharia.us-west.host.bsky.network` | exhausted (2026-10-22T00:47:10Z) | 1,140,022 | 644,883 | 518,697 | 126,186 | 507,861 (+2,855) | 6,220/h | 8,191 | 2,645 | 495,139 (~398,254 act) | **409,090** | `http_429` |
| 3 | `hygrophorus.us-west.host.bsky.network` | exhausted (2026-10-21T21:59:03Z) | 580,743 | 244,980 | 206,830 | 38,150 | 199,231 (+2,656) | 5,786/h | 4,947 | 2,652 | 335,763 (~283,476 act) | **291,075** | `http_5xx` |
| 4 | `discina.us-west.host.bsky.network` | exhausted (2026-10-22T01:46:32Z) | 758,058 | 414,946 | 334,444 | 80,502 | 324,927 (+2,671) | 5,819/h | 4,682 | 4,835 | 343,112 (~276,546 act) | **286,063** | `http_5xx` |
| 5 | `auriporia.us-west.host.bsky.network` | exhausted (2026-10-21T21:06:39Z) | 759,930 | 449,951 | 362,860 | 87,091 | 351,126 (+2,637) | 5,745/h | 8,974 | 2,760 | 309,979 (~249,981 act) | **261,715** | `car` |
| 6 | `jellybaby.us-east.host.bsky.network` | exhausted (2026-10-22T12:49:15Z) | 1,143,602 | 829,832 | 670,950 | 158,882 | 664,726 (+2,507) | 5,462/h | 3,556 | 2,668 | 313,770 (~253,695 act) | **259,919** | `http_5xx` |
| 7 | `phellinus.us-west.host.bsky.network` | exhausted (2026-10-22T00:44:42Z) | 760,646 | 519,942 | 422,921 | 97,021 | 414,505 (+2,735) | 5,959/h | 5,617 | 2,798 | 240,704 (~195,789 act) | **204,204** | `http_429` |
| 8 | `rhizopogon.us-west.host.bsky.network` | exhausted (2026-10-21T21:56:03Z) | 760,211 | 524,937 | 426,844 | 98,093 | 419,896 (+2,704) | 5,891/h | 4,144 | 2,803 | 235,274 (~191,309 act) | **198,256** | `car` |
| 9 | `leccinum.us-west.host.bsky.network` | exhausted (2026-10-22T00:06:23Z) | 583,154 | 409,956 | 346,870 | 63,086 | 334,121 (+2,646) | 5,765/h | 8,925 | 3,824 | 173,198 (~146,545 act) | **159,294** | `http_5xx` |
| 10 | `entoloma.us-west.host.bsky.network` | exhausted (2026-10-21T20:13:17Z) | 581,079 | 409,961 | 347,350 | 62,611 | 333,644 (+2,668) | 5,813/h | 11,171 | 2,535 | 171,118 (~144,984 act) | **158,690** | `http_429` |
| 11 | `agrocybe.us-west.host.bsky.network` | exhausted (2026-10-21T21:51:30Z) | 583,237 | 409,965 | 347,027 | 62,938 | 337,276 (+2,493) | 5,431/h | 4,096 | 5,655 | 173,272 (~146,671 act) | **156,422** | `http_5xx` |
| 12 | `hebeloma.us-west.host.bsky.network` | exhausted (2026-10-21T22:47:29Z) | 580,930 | 419,968 | 355,406 | 64,562 | 347,038 (+2,610) | 5,686/h | 5,430 | 2,938 | 160,962 (~136,217 act) | **144,585** | `http_5xx` |
| 13 | `pholiota.us-west.host.bsky.network` | exhausted (2026-10-21T23:26:18Z) | 581,792 | 454,967 | 384,951 | 70,016 | 375,639 (+2,651) | 5,776/h | 6,589 | 2,723 | 126,825 (~107,308 act) | **116,620** | `http_5xx` |
| 14 | `puffball.us-east.host.bsky.network` | drained (-) | 214,026 | 494,272 | 483,595 | 10,677 | 372,532 (+1,516) | 3,303/h | 16,911 | 94,151 | 0 (~0 act) | **111,062** | `http_5xx` |
| 15 | `amanita.us-east.host.bsky.network` | drained (-) | 214,694 | 495,003 | 483,734 | 11,269 | 373,220 (+2,020) | 4,401/h | 17,601 | 92,913 | 0 (~0 act) | **110,514** | `http_5xx` |
| 16 | `cordyceps.us-west.host.bsky.network` | drained (-) | 192,165 | 456,708 | 446,356 | 10,352 | 338,756 (+2,441) | 5,318/h | 15,173 | 92,427 | 0 (~0 act) | **107,600** | `http_429` |
| 17 | `enoki.us-east.host.bsky.network` | drained (-) | 217,372 | 494,492 | 483,937 | 10,555 | 376,541 (+1,709) | 3,723/h | 15,142 | 92,254 | 0 (~0 act) | **107,396** | `http_5xx` |
| 18 | `inkcap.us-east.host.bsky.network` | drained (-) | 217,413 | 495,256 | 484,287 | 10,969 | 377,323 (+1,536) | 3,346/h | 16,645 | 90,319 | 0 (~0 act) | **106,964** | `http_5xx` |
| 19 | `shiitake.us-east.host.bsky.network` | drained (-) | 216,446 | 494,350 | 483,196 | 11,154 | 378,369 (+2,096) | 4,566/h | 13,821 | 91,006 | 0 (~0 act) | **104,827** | `http_5xx` |
| 20 | `porcini.us-east.host.bsky.network` | drained (-) | 217,452 | 495,494 | 484,067 | 11,427 | 379,962 (+1,782) | 3,882/h | 12,064 | 92,040 | 0 (~0 act) | **104,104** | `http_5xx` |
| 21 | `lionsmane.us-east.host.bsky.network` | drained (-) | 216,363 | 493,867 | 482,938 | 10,929 | 379,639 (+1,837) | 4,002/h | 16,535 | 86,764 | 0 (~0 act) | **103,299** | `car` |
| 22 | `gomphus.us-west.host.bsky.network` | drained (-) | 187,233 | 459,485 | 448,781 | 10,704 | 347,442 (+2,617) | 5,702/h | 17,840 | 83,499 | 0 (~0 act) | **101,339** | `http_429` |
| 23 | `bracket.us-west.host.bsky.network` | drained (-) | 187,382 | 459,367 | 448,706 | 10,661 | 349,557 (+2,617) | 5,702/h | 16,267 | 82,882 | 0 (~0 act) | **99,149** | `http_429` |
| 24 | `witchesbutter.us-west.host.bsky.network` | drained (-) | 187,113 | 458,376 | 447,751 | 10,625 | 349,156 (+2,673) | 5,824/h | 15,029 | 83,566 | 0 (~0 act) | **98,595** | `http_429` |
| 25 | `milkcap.us-west.host.bsky.network` | drained (-) | 187,250 | 459,514 | 448,772 | 10,742 | 350,323 (+2,713) | 5,911/h | 15,769 | 82,680 | 0 (~0 act) | **98,449** | `http_5xx` |
| 26 | `shimeji.us-east.host.bsky.network` | drained (-) | 216,960 | 495,972 | 484,556 | 11,416 | 386,932 (+1,593) | 3,471/h | 7,818 | 89,804 | 0 (~0 act) | **97,622** | `http_5xx` |
| 27 | `ganoderma.us-west.host.bsky.network` | drained (-) | 186,836 | 453,718 | 442,919 | 10,799 | 345,430 (+2,672) | 5,821/h | 14,261 | 83,228 | 0 (~0 act) | **97,489** | `http_429` |
| 28 | `oyster.us-east.host.bsky.network` | drained (-) | 215,869 | 494,763 | 483,994 | 10,769 | 386,824 (+1,566) | 3,412/h | 8,431 | 88,739 | 0 (~0 act) | **97,170** | `http_5xx` |
| 29 | `matsutake.us-west.host.bsky.network` | drained (-) | 188,029 | 460,027 | 449,258 | 10,769 | 352,623 (+2,641) | 5,754/h | 14,564 | 82,071 | 0 (~0 act) | **96,635** | `http_429` |
| 30 | `morel.us-east.host.bsky.network` | drained (-) | 213,164 | 494,884 | 483,927 | 10,957 | 388,157 (+1,520) | 3,312/h | 3,408 | 92,362 | 0 (~0 act) | **95,770** | `http_5xx` |
| 31 | `stinkhorn.us-west.host.bsky.network` | drained (-) | 186,913 | 458,974 | 448,359 | 10,615 | 352,734 (+2,629) | 5,728/h | 13,518 | 82,107 | 0 (~0 act) | **95,625** | `http_429` |
| 32 | `chanterelle.us-west.host.bsky.network` | drained (-) | 187,503 | 460,010 | 449,289 | 10,721 | 354,563 (+2,656) | 5,786/h | 12,375 | 82,351 | 0 (~0 act) | **94,726** | `http_429` |
| 33 | `button.us-west.host.bsky.network` | drained (-) | 187,025 | 458,155 | 447,399 | 10,756 | 354,722 (+2,636) | 5,743/h | 10,456 | 82,221 | 0 (~0 act) | **92,677** | `http_429` |
| 34 | `shaggymane.us-west.host.bsky.network` | drained (-) | 187,827 | 460,036 | 449,253 | 10,783 | 357,089 (+2,668) | 5,813/h | 9,107 | 83,057 | 0 (~0 act) | **92,164** | `http_429` |
| 35 | `truffle.us-east.host.bsky.network` | drained (-) | 172,921 | 456,410 | 445,854 | 10,556 | 354,482 (+2,480) | 5,403/h | 17,004 | 74,368 | 0 (~0 act) | **91,372** | `http_429` |
| 36 | `cremini.us-west.host.bsky.network` | drained (-) | 186,901 | 458,594 | 447,711 | 10,883 | 357,797 (+2,634) | 5,739/h | 7,898 | 82,016 | 0 (~0 act) | **89,914** | `http_429` |
| 37 | `earthstar.us-east.host.bsky.network` | drained (-) | 172,636 | 457,112 | 446,344 | 10,768 | 361,615 (+1,978) | 4,309/h | 18,170 | 66,559 | 0 (~0 act) | **84,729** | `http_429` |
| 38 | `reishi.us-east.host.bsky.network` | drained (-) | 172,392 | 457,456 | 446,675 | 10,781 | 363,709 (+2,217) | 4,830/h | 15,384 | 67,582 | 0 (~0 act) | **82,966** | `http_429` |
| 39 | `scarletina.us-east.host.bsky.network` | drained (-) | 173,048 | 458,162 | 447,604 | 10,558 | 365,698 (+2,026) | 4,414/h | 16,346 | 65,560 | 0 (~0 act) | **81,906** | `http_429` |
| 40 | `velvetfoot.us-east.host.bsky.network` | drained (-) | 172,309 | 456,714 | 445,805 | 10,909 | 364,644 (+2,018) | 4,397/h | 14,817 | 66,344 | 0 (~0 act) | **81,161** | `http_429` |
| 41 | `meadow.us-east.host.bsky.network` | drained (-) | 173,277 | 457,411 | 446,567 | 10,844 | 366,738 (+2,092) | 4,558/h | 15,144 | 64,685 | 0 (~0 act) | **79,829** | `http_429` |
| 42 | `splitgill.us-east.host.bsky.network` | drained (-) | 171,961 | 456,590 | 445,861 | 10,729 | 366,822 (+2,161) | 4,708/h | 12,907 | 66,131 | 0 (~0 act) | **79,038** | `http_429` |
| 43 | `panthercap.us-east.host.bsky.network` | drained (-) | 172,770 | 456,568 | 445,846 | 10,722 | 367,236 (+1,886) | 4,109/h | 12,073 | 66,537 | 0 (~0 act) | **78,610** | `http_429` |
| 44 | `parasol.us-east.host.bsky.network` | drained (-) | 172,351 | 456,210 | 445,436 | 10,774 | 367,920 (+2,169) | 4,725/h | 11,753 | 65,763 | 0 (~0 act) | **77,516** | `http_429` |
| 45 | `helvella.us-east.host.bsky.network` | drained (-) | 173,345 | 457,513 | 446,589 | 10,924 | 369,991 (+2,163) | 4,712/h | 10,666 | 65,932 | 0 (~0 act) | **76,598** | `http_429` |
| 46 | `yellowfoot.us-west.host.bsky.network` | drained (-) | 155,398 | 499,273 | 481,526 | 17,747 | 416,509 (+2,731) | 5,950/h | 11,182 | 53,835 | 0 (~0 act) | **65,017** | `http_429` |
| 47 | `blusher.us-east.host.bsky.network` | drained (-) | 156,777 | 500,346 | 482,420 | 17,926 | 420,393 (+2,374) | 5,172/h | 13,166 | 48,861 | 0 (~0 act) | **62,027** | `http_429` |
| 48 | `hollowfoot.us-west.host.bsky.network` | drained (-) | 153,466 | 477,010 | 422,550 | 54,460 | 363,729 (+2,652) | 5,778/h | 9,101 | 49,719 | 0 (~0 act) | **58,820** | `http_429` |
| 49 | `psathyrella.us-west.host.bsky.network` | drained (-) | 153,254 | 477,295 | 422,753 | 54,542 | 364,056 (+2,665) | 5,806/h | 9,384 | 49,313 | 0 (~0 act) | **58,697** | `http_429` |
| 50 | `russula.us-west.host.bsky.network` | drained (-) | 136,651 | 492,656 | 480,250 | 12,406 | 424,559 (+2,625) | 5,719/h | 17,507 | 38,182 | 0 (~0 act) | **55,689** | `http_5xx` |
| 51 | `mazegill.us-west.host.bsky.network` | drained (-) | 152,559 | 475,544 | 421,129 | 54,415 | 366,179 (+2,598) | 5,660/h | 8,931 | 46,019 | 0 (~0 act) | **54,950** | `http_429` |
| 52 | `panus.us-west.host.bsky.network` | drained (-) | 152,436 | 476,269 | 421,820 | 54,449 | 368,567 (+2,583) | 5,627/h | 7,122 | 46,131 | 0 (~0 act) | **53,253** | `http_429` |
| 53 | `fuzzyfoot.us-west.host.bsky.network` | drained (-) | 153,050 | 476,747 | 422,123 | 54,624 | 370,284 (+2,666) | 5,808/h | 7,603 | 44,236 | 0 (~0 act) | **51,839** | `http_429` |
| 54 | `scalycap.us-west.host.bsky.network` | drained (-) | 142,115 | 453,935 | 415,426 | 38,509 | 364,806 (+2,761) | 6,015/h | 13,103 | 37,517 | 0 (~0 act) | **50,620** | `http_429` |
| 55 | `agaric.us-west.host.bsky.network` | drained (-) | 129,266 | 494,420 | 482,281 | 12,139 | 435,029 (+2,590) | 5,643/h | 15,961 | 31,289 | 0 (~0 act) | **47,250** | `http_429` |
| 56 | `blewit.us-west.host.bsky.network` | drained (-) | 129,438 | 493,219 | 481,014 | 12,205 | 434,180 (+2,563) | 5,584/h | 16,084 | 30,750 | 0 (~0 act) | **46,834** | `http_5xx` |
| 57 | `hydnum.us-west.host.bsky.network` | drained (-) | 129,239 | 495,002 | 482,588 | 12,414 | 436,318 (+2,631) | 5,732/h | 14,918 | 31,352 | 0 (~0 act) | **46,270** | `http_429` |
| 58 | `maitake.us-west.host.bsky.network` | drained (-) | 129,996 | 495,394 | 482,639 | 12,755 | 436,736 (+2,518) | 5,486/h | 14,561 | 31,342 | 0 (~0 act) | **45,903** | `http_429` |
| 59 | `lepista.us-west.host.bsky.network` | drained (-) | 129,253 | 494,591 | 481,706 | 12,885 | 436,444 (+2,576) | 5,612/h | 13,864 | 31,396 | 0 (~0 act) | **45,260** | `http_5xx` |
| 60 | `oysterling.us-west.host.bsky.network` | drained (-) | 142,231 | 454,442 | 415,635 | 38,807 | 370,894 (+2,730) | 5,948/h | 11,760 | 32,981 | 0 (~0 act) | **44,741** | `http_429` |
| 61 | `conocybe.us-west.host.bsky.network` | drained (-) | 130,070 | 496,525 | 483,815 | 12,710 | 439,182 (+2,708) | 5,900/h | 12,980 | 31,653 | 0 (~0 act) | **44,633** | `http_429` |
| 62 | `verpa.us-west.host.bsky.network` | drained (-) | 129,336 | 494,236 | 481,778 | 12,458 | 438,676 (+2,584) | 5,630/h | 12,115 | 30,987 | 0 (~0 act) | **43,102** | `http_429` |
| 63 | `woodtuft.us-west.host.bsky.network` | drained (-) | 142,259 | 454,316 | 416,097 | 38,219 | 373,459 (+2,777) | 6,050/h | 8,867 | 33,771 | 0 (~0 act) | **42,638** | `http_429` |
| 64 | `boletus.us-west.host.bsky.network` | drained (-) | 129,145 | 494,162 | 481,514 | 12,648 | 439,858 (+2,551) | 5,558/h | 10,441 | 31,215 | 0 (~0 act) | **41,656** | `http_429` |
| 65 | `bankera.us-west.host.bsky.network` | exhausted (2026-10-22T02:57:13Z) | 580,598 | 539,955 | 450,762 | 89,193 | 443,286 (+2,683) | 5,845/h | 5,003 | 2,473 | 40,643 (~33,929 act) | **41,405** | `http_5xx` |
| 66 | `chaga.us-west.host.bsky.network` | drained (-) | 129,419 | 495,560 | 482,703 | 12,857 | 441,654 (+2,595) | 5,654/h | 9,558 | 31,490 | 0 (~0 act) | **41,048** | `http_429` |
| 67 | `rooter.us-west.host.bsky.network` | drained (-) | 130,837 | 468,414 | 416,257 | 52,157 | 379,512 (+2,762) | 6,017/h | 9,361 | 27,384 | 0 (~0 act) | **36,745** | `http_429` |
| 68 | `waxcap.us-west.host.bsky.network` | drained (-) | 131,573 | 470,043 | 418,238 | 51,805 | 382,082 (+2,779) | 6,054/h | 7,641 | 28,515 | 0 (~0 act) | **36,156** | `http_429` |
| 69 | `coral.us-east.host.bsky.network` | drained (-) | 126,835 | 455,470 | 438,067 | 17,403 | 402,086 (+2,370) | 5,163/h | 14,261 | 21,720 | 0 (~0 act) | **35,981** | `http_429` |
| 70 | `mottlegill.us-west.host.bsky.network` | drained (-) | 131,175 | 467,914 | 416,543 | 51,371 | 380,588 (+2,768) | 6,031/h | 7,241 | 28,714 | 0 (~0 act) | **35,955** | `http_429` |
| 71 | `magic.us-west.host.bsky.network` | drained (-) | 126,412 | 453,259 | 436,044 | 17,215 | 400,596 (+2,708) | 5,900/h | 6,314 | 29,134 | 0 (~0 act) | **35,448** | `http_429` |
| 72 | `grisette.us-west.host.bsky.network` | drained (-) | 131,178 | 467,904 | 415,731 | 52,173 | 380,372 (+2,798) | 6,096/h | 9,642 | 25,717 | 0 (~0 act) | **35,359** | `http_429` |
| 73 | `woodear.us-west.host.bsky.network` | drained (-) | 127,662 | 455,114 | 437,740 | 17,374 | 402,826 (+2,720) | 5,926/h | 8,861 | 26,052 | 0 (~0 act) | **34,913** | `http_429` |
| 74 | `pioppino.us-west.host.bsky.network` | drained (-) | 131,070 | 470,249 | 418,250 | 51,999 | 383,706 (+2,780) | 6,057/h | 7,718 | 26,826 | 0 (~0 act) | **34,544** | `http_429` |
| 75 | `goldenear.us-west.host.bsky.network` | drained (-) | 131,261 | 468,596 | 416,219 | 52,377 | 382,389 (+2,772) | 6,039/h | 5,937 | 27,893 | 0 (~0 act) | **33,830** | `http_429` |
| 76 | `lobster.us-east.host.bsky.network` | drained (-) | 126,860 | 455,268 | 437,882 | 17,386 | 404,140 (+2,254) | 4,911/h | 10,440 | 23,302 | 0 (~0 act) | **33,742** | `http_429` |
| 77 | `polypore.us-west.host.bsky.network` | drained (-) | 130,953 | 468,992 | 416,831 | 52,161 | 383,219 (+2,805) | 6,111/h | 6,472 | 27,140 | 0 (~0 act) | **33,612** | `http_429` |
| 78 | `elfcup.us-east.host.bsky.network` | drained (-) | 131,312 | 469,710 | 417,557 | 52,153 | 384,240 (+2,428) | 5,290/h | 10,155 | 23,162 | 0 (~0 act) | **33,317** | `http_429` |
| 79 | `dapperling.us-west.host.bsky.network` | drained (-) | 130,525 | 468,281 | 416,288 | 51,993 | 383,179 (+2,776) | 6,048/h | 5,589 | 27,520 | 0 (~0 act) | **33,109** | `http_429` |
| 80 | `suillus.us-west.host.bsky.network` | drained (-) | 130,970 | 468,155 | 416,308 | 51,847 | 383,424 (+2,794) | 6,087/h | 6,085 | 26,799 | 0 (~0 act) | **32,884** | `http_429` |
| 81 | `mycena.us-west.host.bsky.network` | drained (-) | 130,457 | 467,835 | 416,387 | 51,448 | 387,258 (+2,658) | 5,791/h | 1,943 | 27,186 | 0 (~0 act) | **29,129** | `http_429` |
| 82 | `calocybe.us-west.host.bsky.network` | exhausted (2026-10-22T00:25:58Z) | 581,734 | 574,941 | 480,248 | 94,693 | 466,779 (+2,674) | 5,826/h | 7,677 | 5,792 | 6,793 (~5,674 act) | **19,143** | `car` |
| 83 | `poisonpie.us-west.host.bsky.network` | exhausted (2026-10-21T23:33:03Z) | 426,764 | 429,939 | 410,137 | 19,802 | 393,397 (+2,697) | 5,876/h | 8,791 | 7,949 | 0 (~0 act) | **16,740** | `http_5xx` |
| 84 | `fibercap.us-west.host.bsky.network` | drained (-) | 426,473 | 710,813 | 653,314 | 57,499 | 641,762 (+2,676) | 5,830/h | 7,626 | 3,926 | 0 (~0 act) | **11,552** | `http_429` |
| 85 | `cortinarius.us-west.host.bsky.network` | exhausted (2026-10-21T23:52:00Z) | 582,486 | 589,943 | 492,516 | 97,427 | 481,397 (+2,625) | 5,719/h | 8,604 | 2,515 | 0 (~0 act) | **11,119** | `http_5xx` |
| 86 | `gomphidius.us-west.host.bsky.network` | drained (-) | 581,522 | 613,155 | 509,945 | 103,210 | 499,929 (+2,677) | 5,832/h | 7,392 | 2,624 | 0 (~0 act) | **10,016** | `http_5xx` |
| 87 | `brittlegill.us-west.host.bsky.network` | exhausted (2026-10-22T05:28:16Z) | 426,598 | 694,904 | 638,476 | 56,428 | 628,892 (+2,610) | 5,686/h | 5,740 | 3,844 | 0 (~0 act) | **9,584** | `http_429` |
| 88 | `spaces-alpha.host.bsky.network` | drained (-) | 631 | 614 | 614 | 0 | 614 (+547) | 1,192/h | 0 | 0 | 17 (~17 act) | **17** | `none` |
| 89 | `hedgehog.us-west.host.bsky.network` | drained (-) | 5 | 18 | 18 | 0 | 18 (+12) | 26/h | 0 | 0 | 0 (~0 act) | **0** | `none` |
| 90 | `vlpds.host.bsky.network` | drained (-) | 6 | 6 | 6 | 0 | 6 (+0) | 0/h | 0 | 0 | 0 (~0 act) | **0** | `none` |


### 1.6 Top 25 Custom PDS Hosts by Remaining / Active Work (`S5`, `2026-10-22T13:33:31Z`)

| Rank | Custom PDS Host | Relay Floor | Enumerated (`Repos`) | Active | Inactive | Complete at `S5` (`S1->S5` `27.5m`) | Active Pending (`S5`) | Errored (`Failed` `S5`) | Latest Error Class |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `pds.wsocial.network` | 50,522 | 48,315 | 48,065 | 250 | 42,464 (+2,127) | 5,601 | 0 | `none` |
| 2 | `keik.info` | 10,392 | 10,391 | 10,391 | 0 | 7,608 (+2,199) | 2,783 | 0 | `none` |
| 3 | `pds.louisvillebsky.app` | 6,520 | 6,512 | 6,511 | 1 | 4,060 (+2,521) | 2,451 | 0 | `none` |
| 4 | `pds.si46.world` | 8,927 | 11,073 | 11,073 | 0 | 8,916 (+2,813) | 2,157 | 0 | `none` |
| 5 | `certified.one` | 15,024 | 14,874 | 14,869 | 5 | 11,918 (+2,018) | 2,951 | 0 | `none` |
| 6 | `roomy.chat` | 5,633 | 5,620 | 5,619 | 1 | 3,074 (+2,423) | 2,545 | 0 | `none` |
| 7 | `pds.1095.me` | 10,325 | 10,085 | 10,085 | 0 | 7,471 (+2,353) | 2,614 | 0 | `none` |
| 8 | `pds.radial.fan` | 5,661 | 5,599 | 5,595 | 4 | 3,218 (+2,589) | 2,377 | 0 | `none` |
| 9 | `tdpl.io` | 9,516 | 9,626 | 9,534 | 92 | 7,148 (+2,538) | 2,385 | 0 | `none` |
| 10 | `tngl.sh` | 11,843 | 11,196 | 11,171 | 25 | 8,418 (+2,139) | 2,753 | 0 | `none` |
| 11 | `surf.social` | 6,312 | 5,998 | 5,253 | 745 | 3,203 (+2,268) | 2,047 | 3 | `car` |
| 12 | `certified-apppds-core-pr-base.up.railway.app` | 4,486 | 4,349 | 4,349 | 0 | 2,838 (+2,138) | 1,511 | 0 | `none` |
| 13 | `atproto.brid.gy` | 66,183 | 44,439 | 31,143 | 13,296 | 29,824 (+2,071) | 1,314 | 0 | `none` |
| 14 | `pds.chicago.com` | 6 | 2,715 | 2,715 | 0 | 1,380 (+1,356) | 1,335 | 0 | `none` |
| 15 | `x.mt.social` | 1,051 | 2,108 | 2,108 | 0 | 1,939 (+1,911) | 169 | 0 | `none` |
| 16 | `fed.brid.gy` | 0 | 49,987 | 34,962 | 15,025 | 34,768 (+1,886) | 183 | 0 | `none` |
| 17 | `repo.layers.pub` | 2 | 2,185 | 2,185 | 0 | 584 (+242) | 1,595 | 6 | `car` |
| 18 | `pds.ekamdor.social` | 1,618 | 1,748 | 1,741 | 7 | 1,741 (+1,717) | 0 | 0 | `none` |
| 19 | `pds-staging.ridgeway.network` | 6 | 1,489 | 1,489 | 0 | 1,489 (+1,436) | 0 | 0 | `none` |
| 20 | `pds.sprk.so` | 8,320 | 8,422 | 3,351 | 5,071 | 3,351 (+1,433) | 0 | 0 | `none` |
| 21 | `eurosky.social` | 38,021 | 1,748 | 1,686 | 62 | 1,461 (+1,051) | 225 | 0 | `timeout` |
| 22 | `cryptoanarchy.network` | 9 | 12,326 | 2,662 | 9,664 | 2,455 (+987) | 207 | 0 | `none` |
| 23 | `berlin-demo.eurosky.social` | 4 | 1,695 | 1,646 | 49 | 1,652 (+934) | 0 | 0 | `none` |
| 24 | `pds.oathling.ai` | 770 | 853 | 853 | 0 | 853 (+824) | 0 | 0 | `none` |
| 25 | `pds.clear.community` | 881 | 875 | 875 | 0 | 875 (+821) | 0 | 0 | `none` |


---

## 2. Jetstream `v0.4.4` (`d4db8c0`) & `atmos` `v0.7.3` Code Audit

### 2.1 Complete CLI Flag Audit (`docker exec js jetstream serve --help` & `internal/jetstreamd/options.go`)

| CLI Flag / Env Var | Binary Default (`options.go`) | Deployed Container Value | Role in `bootstrap`, `merging`, and `steady_state` |
|---|---|---|---|
| `--backfill-global-downloads` | `256` | **`64`** | Fleet-wide `getRepo` concurrency cap in `bootstrap` (`atmosbackfill.Options.GlobalDownloads`). |
| `--backfill-host-workers-max` | `32` | **`2`** | Per-host `getRepo` worker cap in `bootstrap` (`atmosbackfill.Options.HostWorkers`). Capped `jellybaby.us-east` at 2 workers during the final 7.35h of `bootstrap`. |
| `--backfill-max-active-hosts` | `512` | **`32`** | Concurrent `listRepos` host crawlers in `bootstrap` (`atmosbackfill.Options.MaxActiveHosts`). |
| `--backfill-async-flush-workers` | `4` | **`8`** | Async zstd segment block compression workers during `bootstrap`. |
| `--pending-repo-pass-workers` | `256` | **`32`** | Global worker count for `runPendingRepoRetryPass` in `merging` (`merge.go:235`). Currently driving **`186.8 repos/sec` (`672k repos/hr`)**. |
| `--failed-repo-retry-host-workers` | `4` | **`1`** | Per-host concurrency limit (`s.limit` in `retryScheduler`, `retry_sched.go:41` and `acquireHost`, `retry.go:709`) shared by **both** `runPendingRepoRetryPass` (`merging`) and `RunFailedRepoRetry` (`steady_state`). |
| `--failed-repo-retry-workers` | **`16`** | *Unset (`16`)* | Global worker count for `RunFailedRepoRetry` in `steady_state` (`steady.go:138`, `options.go:79`). **Omitted from container command**, cutting global concurrency in half (`32 -> 16`) upon entering `steady_state`! |
| `--failed-repo-retry-interval` | **`4h0m0s`** | *Unset (`4h0m0s`)* | Sleep interval *after* each `RunFailedRepoRetry` pass finishes (`retry.go:179`) AND base exponential backoff delay (`nextAttemptAt`, `retry.go:850`). |
| `--failed-repo-retry-max-delay` | `168h0m0s` | *Unset (`168h0m0s`)* | Maximum per-repo retry backoff ceiling (`retry.go:275`). |
| `--compaction-interval` | `4h0m0s` | **`0`** | Disables `runDeleteCompaction` in `merging` (`compact_deletes.go:72`), `rebuildLiveTombstones` at `steady_state` start (`compact_deletes.go:655`), and `runSteadyCompactor` (`compact_deletes.go:558`). |
| `--skip-merge-discovery` | `false` | *Unset (`false`)* | When `false`, `runMerge` runs `runMergeDiscovery` (`merge.go:149`) after `runPendingRepoRetryPass` to re-scan `listRepos` from each host's `LastNonEmptyCursor`. |

### 2.2 Four Critical Code Mechanisms Governing the Remaining Work

1. **Why 18 `*.host.bsky.network` shards hit `PDS host exhausted` during `bootstrap` (`atmos/v0.7.3/backfill/engine.go:442` + `internal/ingest/backfill/run.go:33`):**
   - In [`internal/ingest/backfill/run.go:33, 225`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/run.go#L33), `jetstream` overrides `atmos`'s `defaultHostMaxAttempts = 8` down to `hostMaxAttempts = 2`.
   - On Bluesky PDS shards (`*.host.bsky.network`), `com.atproto.sync.listRepos` and `com.atproto.sync.getRepo` share the same per-IP rate-limit bucket (`ratelimit-policy: 3000;w=300` = 3,000 requests per 5 minutes). When concurrent `getRepo` workers or live firehose verification resyncs momentarily drained `ratelimit-remaining` to `0` while `listRepos` paged, `listRepos` received `HTTP 429`.
   - Unlike `enumerateHostsWithRetry` ([`engine.go:180-182`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/atmos_v073_engine.go#L180-L182)), which checks `if xrpc.IsRateLimited(err)` and sleeps until `RateLimit-Reset` (`rateLimitDelay`), `runHostManager` ([`engine.go:441-443`](file:///Users/joshlacalamito/.gemini/antigravity/brain/54b24adf-96f7-43d6-af34-7f5452da9729/scratch/atmos_v073_engine.go#L441-L443)) sleeps `backoffDelay(e.hostBackoffBase(), e.hostBackoffMax(), attempt-1)` with `defaultHostBackoffBase = 1 * time.Minute` (`60s–90s`).
   - Because the 5-minute PDS rate-limit window had ~3.5 minutes remaining when Attempt 2 fired 60–90 seconds later, Attempt 2 immediately failed with `listRepos: xrpc 429` on its first page, hit `attempt == hostMaxAttempts` (`2 == 2`), and permanently marked the shard `exhausted` (`Store.OnHostExhausted`).

2. **Why `runMergeDiscovery` at the end of `merging` (`~15:05Z` today) will automatically recover all 18 exhausted shards (`internal/ingest/orchestrator/merge_discovery.go:62-103`, `internal/ingest/backfill/store.go:1355-1359`):**
   - Right after `runPendingRepoRetryPass` finishes (`~15:05Z`), `runMerge` calls `o.runMergeDiscovery(ctx, runner)` ([`merge.go:149`](file:///tmp/jetstream-deployed-audit/internal/ingest/orchestrator/merge.go#L149)).
   - `discoveryOnlyAtmosStore.HostCursor` calls `Store.HostDiscoveryCursors` ([`store.go:1355-1359`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/store.go#L1355-L1359)), which returns `{Cursor: host.LastNonEmptyCursor, Drained: false}` for **every** host in `pdshost/*`—including all 18 `exhausted` shards!
   - Because `runMergeDiscovery` runs with `DiscoverOnly: gt.Some(true)` (zero `getRepo` downloads!) and `live.Consumer` is not yet running (zero verification resyncs!), each PDS shard has its full `3,000` requests/5m quota dedicated solely to `listRepos?limit=1000`. Even the furthest-behind shard (`chalciporus.us-west`, `574,449` unenumerated accounts) needs only **`575` pages** (`< 20%` of one 5-minute window).
   - Every active newly discovered repo (`~3.03M` across the 18 shards) is written via `Store.OnDiscoverForRetry` ([`store.go:1170-1171`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/store.go#L1170-L1171)) with `Backfill.Status = StatusFailed`, `LastError = "discovered post-bootstrap; queued for retry"`, and `NextAttemptAt = time.Time{}` (immediately eligible on the first `steady_state` retry pass).

3. **Why `RunFailedRepoRetry` in `steady_state` has a per-repo `fsync` serialization bottleneck (`internal/ingest/backfill/retry.go:144, 577-600`, `internal/ingest/writer.go:586-618`):**
   - During `runPendingRepoRetryPass` (`merging`), `r.completions` is a `CompletionBatcher` (`retry.go:205-209`) that batches Pebble completions with segment block flushes every 4,096 events or 30 seconds.
   - In `RunFailedRepoRetry` (`steady_state`, [`retry.go:142-144, 577-600`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/retry.go#L577-L600)), `r.completions` is `nil`. Consequently, **every single completed repo** executes:
     1. `r.cfg.Writer.DrainDurability(ctx)` ([`writer.go:605-618, 586-601, 695-705`](file:///tmp/jetstream-deployed-audit/internal/ingest/writer.go#L586-L618)), which acquires `w.drainMu.Lock()` and `w.mu.Lock()`, flushes and **`fsync`s** a partial zstd block to the active segment file, and executes **two** synchronous `pebble.Sync` metadata commits (`commitDurableBatchLocked`).
     2. `r.store.OnComplete(ctx, ...)` ([`store.go:1584-1617`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/store.go#L1584-L1617)), which acquires `s.countsMu.Lock()` and `s.writeMu.Lock()` and executes a **third** synchronous Pebble commit (`updateRepoStatusAndCounts`).
   - While network `getRepo` downloads (`~180 ms` RTT) run in parallel across `--failed-repo-retry-workers`, the final `~2.5–4.5 ms` durability drain + Pebble commit serializes behind `w.drainMu.Lock()`. Raising `--failed-repo-retry-workers` from `16` (the unset default) to `64` overlaps the `180 ms` network RTT across 64 shards up to the `~220–350 repos/sec` local disk commit ceiling.

4. **The `steady_state` `skipParked` Pass-Dumping Hazard (`internal/ingest/backfill/retry_sched.go:97`, `internal/ingest/backfill/retry.go:470-477, 525-536, 179`):**
   - In [`retry.go:470-476`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/retry.go#L470-L476), before every download, `attemptCandidate` checks `if until := r.clientParkedUntil(cand.PDS); !until.IsZero() { r.parkHost(cand.Host, until); return r.skipParked(ctx, cand, until) }`.
   - `client.GetRepoRateLimitedUntil()` becomes non-zero whenever a PDS shard's `RateLimit-Remaining` drops to `<= 150` (`5%` reserve of `3,000` in a 300-second window, i.e., after `2,850` requests in 300 seconds = **`9.50 req/sec`**).
   - In `runPendingRepoRetryPass` (`waitParks == true`), `retryScheduler.next()` ([`retry_sched.go:97-105`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/retry_sched.go#L97-L105)) sets parked hosts aside until `until` and requeues the candidate in the same pass.
   - **In `RunFailedRepoRetry` (`steady_state`, `waitParks == false`), `retryScheduler.next()` does NOT pause a parked host!** Instead, it immediately pops every remaining candidate in `h.queue` (`100k–467k` repos!) in `< 1 second`, and `skipParked` ([`retry.go:525-536`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/retry.go#L525-L536)) defers all of them out of the current pass (`NextAttemptAt = until`, `requeue = false`), after which `RunFailedRepoRetry` sleeps for `--failed-repo-retry-interval` (`4h` default, [`retry.go:179`](file:///tmp/jetstream-deployed-audit/internal/ingest/backfill/retry.go#L179))!
   - **Operational implication:**
     - Keeping `--failed-repo-retry-host-workers=1` ensures each shard runs at `<= 5.5 req/sec` (`< 9.50 req/sec`) even when `< 64` shards remain, **preventing `clientParkedUntil` from ever parking a shard due to self-induced rate exhaustion**.
     - Lowering `--failed-repo-retry-interval` from `4h0m0s` to **`10m` (or `5m`)** ensures that if an external burst (such as `live.Consumer` verification resyncs) ever parks a shard during a pass, the next pass starts `10 minutes` after the pass finishes rather than stalling for `4 hours`.

---

## 3. Rate Limiting & Upstream Error Analysis (`docker logs --since 6h js` & `/status?tab=hosts`)

### 3.1 Exact 6-Hour `docker logs js` Error & Event Counts (`07:11Z` – `13:11Z` Oct 22)

Command executed on `jetstream-selfhost-test`:
```bash
docker logs --since 6h js > /tmp/js_logs_6h.jsonl 2>&1
```
Parsed JSON log summary over the 6-hour window:
- **Log message counts (6h):**
  - `verification failure`: `14,481` (live firehose commit verification mismatches triggering async resyncs)
  - `async error`: `699`
  - `failed repo retry attempt failed`: `663` (from custom PDS hosts during the first 5 minutes of `pending_pass`)
  - `rotated segment`: `56`
  - `dropped stale-ordered async resync`: `52`
  - `repo failed`: `15`
  - `PDS host exhausted`: `1` (`jellybaby.us-east.host.bsky.network` at `12:49:15.821810022Z`)
  - `PDS-direct backfill drained`: `1` (`12:49:24.648549184Z`)
  - `cutover begin`: `1` (`12:49:24.648568739Z`)
- **HTTP `429` errors by host over the last 6 hours:**
  - `jellybaby.us-east.host.bsky.network`: **`1`** (`listRepos: xrpc 429` at `12:49:15Z`)
  - All other hosts (`getRepo` and `listRepos` across `278,750` `pending_pass` repo downloads): **`0`**
- **HTTP `5xx` errors by host over the last 6 hours:**
  - `bsky.nrbrtspvk.com`: `209` (`xrpc 502`)
  - `skeetcentral.com`: `6` (`xrpc 502`)
  - `pds.stupid.fish`: `5` (`xrpc 502` / `500`)
- **Other transport/XRPC errors by host over the last 6 hours (top hosts):**
  - `pds.angussoftware.dev`: `61`
  - `certified-apppds-core-epds-pr-*.up.railway.app` (8 ephemeral PR preview hosts): `328` total
  - `earthstar.us-east.host.bsky.network`: `12` (`incomplete CAR: ... missing: unexpected EOF: block not found`)
  - `pds1.rx-sns.net`: `7`, `pds9.rx-sns.net`: `5`, `pds.sensorthings.dev`: `4`, `2queer.xyz`: `3`, `bridgy-hubble.microcosm.blue`: `3`

### 3.2 Cumulative `StatusFailed` (`3,813,113` repos) Origin & Breakdown

Across all `3,351` host buckets in `/status?tab=hosts`:
- **`3,811,751` (`99.964%`) of the `3,813,113` `StatusFailed` repos are on the 89 `*.host.bsky.network` shards** (`1,699,525` across the 27 `us-east` shards; `2,112,226` across the 62 `us-west` shards), and only `1,362` (`0.036%`) are on custom PDS hosts.
- Inspecting `<tr class="host-errors-row">` timestamps across all 89 `*.host.bsky.network` shards proves that the `3.81M` `http_429 backfill: still rate limited after 4 attempts: xrpc 429` failures occurred **`18h` to `1d 16h` ago** (during the pre-`2026-10-21T19:07Z` high-concurrency `bootstrap` run before `--backfill-host-workers-max=2` was applied).
- Because neither `bootstrap` nor `merging` runs `RunFailedRepoRetry`, those `3,811,751` `StatusFailed` repos have been sitting untouched in Pebble with `RetryCount = 0` and `NextAttemptAt = 0` waiting for `steady_state` (`~2026-10-22T15:10Z`).

---

## 4. Verification of the "Relay Floor vs Enumerated" Hypothesis

### 4.1 Live Rate-Limit Header Probe on `chalciporus.us-west.host.bsky.network`

Command executed from `jetstream-selfhost-test` at `2026-10-22T13:22:41Z`:
```bash
curl -sSI "https://chalciporus.us-west.host.bsky.network/xrpc/com.atproto.sync.listRepos?limit=1" | grep -iE "HTTP/|ratelimit|retry"
```
Output:
```text
HTTP/2 200 
ratelimit-limit: 3000
ratelimit-reset: 1792675662
ratelimit-remaining: 2999
ratelimit-policy: 3000;w=300
access-control-expose-headers: RateLimit-Limit, RateLimit-Reset, RateLimit-Remaining, RateLimit-Policy
```

### 4.2 Why `chalciporus` Enumerated `184,972` vs `759,421` Relay Floor (and Why Both Phenomenon Directions Exist)

1. **Why `chalciporus.us-west` stopped at `184,972` (`24.36%` of `759,421`):**
   - `docker logs js` records the exact termination event at `2026-10-21T19:53:38.612744099Z`:
     ```json
     {"time":"2026-10-21T19:53:38.612744099Z","level":"WARN","msg":"PDS host exhausted","component":"backfill/run","host":"chalciporus.us-west.host.bsky.network","attempts":2,"err":"backfill: host chalciporus.us-west.host.bsky.network listRepos: xrpc 429"}
     ```
   - `184,972` is **185 pages of 1,000 entries** (minus 28 duplicate/migrated DIDs reconciled across hosts). `chalciporus.us-west` did **not** lose 574k accounts to deletion/migration; its `listRepos` pagination was aborted at page 185 by `xrpc 429` after 2 attempts (`1m` apart inside a `300s` rate-limit window; see §2.2.1).
   - The same exact mechanism truncated all 18 exhausted `*.host.bsky.network` shards (`chalciporus` at `184,972`, `hygrophorus` at `244,980`, `agrocybe` at `409,965`, `leccinum` at `409,956`, `entoloma` at `409,961`, `discina` at `414,946`, `hebeloma` at `419,968`, `poisonpie` at `429,939`, `auriporia` at `449,951`, `pholiota` at `454,967`, `phellinus` at `519,942`, `rhizopogon` at `524,937`, `bankera` at `539,955`, `calocybe` at `574,941`, `cortinarius` at `589,943`, `stropharia` at `644,883`, `brittlegill` at `694,904`, `jellybaby` at `829,832`), leaving **`3,701,018` unenumerated accounts** (`~3,034,530` active) that `runMergeDiscovery` will enumerate at `~15:05Z` today.
2. **Why the raw sum of `relay_floor` across all 6,638 hosts is `24,697,876` while `Discovered` is `42,154,129` (reconciled network floor `46,231,697`):**
   - As cross-checked against [`catbird-intel/eval/pds_account_count_reconciliation_20261022.json`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/pds_account_count_reconciliation_20261022.json), `bsky.network`'s `com.atproto.sync.listHosts` `accountCount` counter was initialized after older 2023–2024 mushroom PDS shards (`26` of `27` `us-east` shards and `27` older `us-west` shards) already held hundreds of thousands of accounts (`listHosts` reports `129k–217k` on shards where `listRepos` enumerates `455k–500k` accounts).
   - Conversely, on newer 2025–2026 shards (`580k`, `760k`, and `1.14M` cohorts, which include the 18 exhausted shards), `listHosts` `accountCount` accurately tracks the true total repository count on the shard.

---

## 5. Completion Projections, Costed Options (a) / (b) / (c), and Safe Restart Procedure

### 5.1 Phase Completion Timeline From `2026-10-22T13:33:31Z` (`S5`)

1. **Remaining `runPendingRepoRetryPass` (in `merging`):**
   - Remaining active `StatusPending` repos at `13:33:31Z`: **`968,210`** (across ~87 `*.host.bsky.network` shards + 3 custom PDSs; largest single shard is `gomphus.us-west` with `17,452` active pending).
   - Measured speed (`S4 -> S5`, `workers=32, host_workers=1`): **`672,482` repos/hr (`11,208` repos/min, `186.8` repos/sec)**, `5.84` repos/sec/worker.
   - Projected completion of `runPendingRepoRetryPass`: **`~1.55 hours` from `13:33Z` -> `~2026-10-22T15:05Z–15:10Z`**.
   - At `~15:10Z`, `Downloaded` (`Complete`) will reach **`~35,175,000` repos** (`90.2%` of the `38.98M` currently enumerated active accounts).
2. **`runMergeDiscovery` (final step of `merging` at `~15:05Z–15:10Z`):**
   - Re-enumerates `listRepos?limit=1000` from `LastNonEmptyCursor` across all hosts (`DiscoverOnly: true`, `MaxActiveHosts: 32`).
   - Enumerates the `~3,701,018` remaining accounts (`~3,034,530` active) on the 18 exhausted `*.host.bsky.network` shards in **`~3–5 minutes`** (`<= 575` pages max on `chalciporus.us-west` vs `3,000` req/5m limit).
   - Queues the `~3,034,530` newly discovered active accounts into `StatusFailed` alongside the existing `3,813,113` `StatusFailed` repos -> **`~6,858,827` total active `StatusFailed` repos** across 178 hosts (largest shard: `chalciporus.us-west` with `466,972` `StatusFailed` repos; second largest: `stropharia.us-west` with `401,001`).
3. **Transition to `steady_state` (`~2026-10-22T15:15Z`):**
   - `42.9 hours` remain between `steady_state` entry (`2026-10-22T15:15Z`) and the `catbird-gcs-buffer.service` power-off deadline (`2026-10-24T10:10:00Z`).

### 5.2 Comparison of Options (a), (b), and (c)

We simulated the exact `retryScheduler` (`retry_sched.go`) per-shard queue depletion across all 178 hosts (`6,858,827` `StatusFailed` repos at `steady_state` entry `~15:15Z`), accounting for both the measured network rate (`5.32–5.84 repos/sec/worker`), the `steady_state` `DrainDurability` + `OnComplete` serial commit ceiling (`~200–300 repos/sec`), and the `9.50 req/sec` (`2,850 req/300s`) per-shard `clientParkedUntil` / `skipParked` threshold:

| Option | Configuration / Action | Projected State at `2026-10-24T10:10:00Z` (`+42.9h` of `steady_state`) | 100% Active Repo Completion Time (`~42.03M` active repos) | Projected GCS Buffer Usage (vs `2,748.78 GB` Cap) | Incremental Cost (Compute + GCS) |
|---|---|---|---|---|---|
| **(a) No change** | Keep current container flags (`--failed-repo-retry-workers` unset = **`16`**, `--failed-repo-retry-host-workers=1`, `--failed-repo-retry-interval=4h0m0s`), stop at `2026-10-24T10:10Z` | **High risk of tail truncation (`91%–98%` of `StatusFailed` done; `145k–600k` active repos left unarchived on `chalciporus` & `stropharia`)**: With `workers=16, host_workers=1`, ideal zero-park completion takes **`35.9h–50.3h`** (`2026-10-24T03:10Z` to `2026-10-24T17:35Z`). Worse, any transient `429` or verification-resync quota dip on `chalciporus` (`467k` repos) or `stropharia` (`401k` repos) triggers `skipParked` and stalls that shard for **`4 hours` per park** (`--failed-repo-retry-interval=4h`). | `2026-10-24T03:10Z` (ideal) to **`2026-10-24T23:00Z+`** (if 1–2 rate-limit parks occur) — **exceeds deadline under conservative rate or any park** | `~1,810 GB` (`65.8%` of `2.5 TiB` cap) | **`$0.00`** incremental (`$13.90` already budgeted for remaining `44.6h` on `n2-highmem-8` Spot) |
| **(b) Flag change only (Recommended)** | At `~2026-10-22T15:15Z` (once `Phase` enters `steady_state`), restart `js` with **`--failed-repo-retry-workers=64 --failed-repo-retry-host-workers=1 --failed-repo-retry-interval=10m`** (keeping deadline at `2026-10-24T10:10Z`) | **`100.0%` of reachable active repos (`~42.01M` / `42.03M`) completed `14h–18h` BEFORE the `2026-10-24T10:10Z` deadline**: `workers=64` 4x's global retry concurrency during the 89-shard bulk phase (`~220–280 repos/sec`), `host_workers=1` keeps per-shard rate at `<= 5.5 req/s` (`< 9.5 req/s`, **zero self-induced `skipParked` events**), and `interval=10m` cuts any external park recovery from `4h` to `10m`. | **`2026-10-23T16:15Z` to `2026-10-23T20:30Z`** (`+25.0h` to `+29.2h` after `steady_state` start; **`13.7h–17.9h` margin before deadline**) | `~1,850 GB` (`67.3%` of `2.5 TiB` cap; leaves `~898 GB` headroom) | **`$0.00` compute** + `~$0.92/mo` (`$0.03/day`) for the extra `~40 GB` of recovered tail segments |
| **(c) Flag change + +24h stop extension (Contingency)** | Same flag change as (b) **plus** extend `catbird-gcs-buffer.service` `--deadline` by `+24h` to `2026-10-25T10:10:00Z` (`--run-hours 87.05`) | **`100.0%` completed + 37h–42h of extra multi-pass retry soak** for intermittently failing `http_5xx` / `incomplete CAR` repos (`~15k–30k` repos across `bsky` shards and custom PDSs). Only needed if `runMergeDiscovery` or `steady_state` encounters an unexpected upstream outage. | Primary pass done **`2026-10-23T16:15Z–20:30Z`**; multi-pass `5xx`/CAR tail soak through `2026-10-25T10:10Z` | `~1,940 GB` (`70.6%` of `2.5 TiB` cap; includes `+24h` of live firehose at `~3.8 GB/hr`) | **`+$7.48` compute** (`24h` on `n2-highmem-8` Spot at `$0.3116/hr`, **covered 6.0x over by the `$45.27` saved** from downsizing `n2-highmem-32 -> n2-highmem-8`) |

> **Why `--failed-repo-retry-host-workers=1` is preferred over `2` in Option (b), with `2` as an optional turbo knob:**
> With `--failed-repo-retry-workers=64 --failed-repo-retry-host-workers=1`, the 71 drained shards (`23k–94k` failed repos each) drain in the first `~6–8 hours` at `~250 repos/sec` fleet-wide, while `chalciporus.us-west` (`466,972` repos) and `stropharia.us-west` (`401,001` repos) drain continuously at `~5.2 repos/sec` (`~18,700 repos/hr`) without ever exceeding the `9.50 req/sec` (`2,850 req/300s`) PDS `clientParkedUntil` threshold (`466,972 / 18,700 = 25.0 hours`, finishing at `~2026-10-23T16:15Z`, **18 hours before the `2026-10-24T10:10Z` deadline**). If `--failed-repo-retry-host-workers=2` is used alongside `--failed-repo-retry-interval=5m`, completion drops to `~14–19 hours` (`~2026-10-23T06:00Z–10:15Z`), though the top 15 shards may periodically hit `2,850 req/300s` and cycle through `5m` intervals between passes.

### 5.3 Exact Safe Checkpoint & Container Restart Procedure (Execute at `~2026-10-22T15:15Z` When `Phase == steady_state`)

**Timing recommendation:** Do **not** interrupt `js` while `Phase` is `merging` (`13:05Z`–`~15:15Z`). Right now `runPendingRepoRetryPass` is already running at `672,482 repos/hr` (`--pending-repo-pass-workers=32`) and will finish around `15:05Z`, followed immediately by `runMergeDiscovery` (`~15:05Z–15:10Z`). Wait until `/status?tab=summary` shows `Phase: steady_state` (or `docker logs js` logs `"cutover complete"` / `"entering steady-state"`), then run:

```bash
# 1. SSH to jetstream-selfhost-test
gcloud compute ssh jetstream-selfhost-test --project=catbird-436600 --zone=us-west1-b

# 2. Verify collector has entered steady_state and check offloader queue
CIP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' js)
curl -sS "http://${CIP}:8080/status?tab=summary" | grep -A4 -E "Phase|State|Discovered|Downloaded|Pending|Errored"
systemctl status catbird-gcs-buffer.service --no-pager | tail -n 5

# 3. Record pre-restart container command & image SHA, and save a status snapshot
docker inspect js > /tmp/js_container_inspect_pre_restart_$(date -u +%Y%m%dT%H%M%SZ).json
curl -sS "http://${CIP}:8080/status?tab=summary" > /tmp/status_summary_pre_restart.html
curl -sS "http://${CIP}:8080/status?tab=hosts" > /tmp/status_hosts_pre_restart.html

# 4. Gracefully stop js (honors --shutdown-timeout=60s to flush active segment block & sync Pebble WAL)
docker stop -t 60 js
docker logs --tail 30 js

# 5. Checkpoint Pebble MANIFEST/CURRENT list to verify clean shutdown (no .tmp files, clean LOCK)
ls -la /data/jetstream/meta.pebble/

# 6. Recreate and start js with updated steady-state retry flags
#    (--failed-repo-retry-workers=64 --failed-repo-retry-host-workers=1 --failed-repo-retry-interval=10m)
docker rm js
docker run -d --name js   --restart unless-stopped   -v /data/jetstream:/data   sha256:bb779a4831afab96eea1cf399469619772ca8eed69db6f3a5202ae858d393745   /usr/local/bin/jetstream serve   --data-dir=/data   --backfill-global-downloads=64   --backfill-host-workers-max=2   --backfill-max-active-hosts=32   --backfill-async-flush-workers=8   --failed-repo-retry-workers=64   --failed-repo-retry-host-workers=1   --failed-repo-retry-interval=10m   --pending-repo-pass-workers=64   --shutdown-timeout=60s   --compaction-interval=0

# 7. Verify startup in steady_state and confirm first retry pass launches after 5m (failedRepoRetryFirstPassDelay)
sleep 5
docker ps --no-trunc
docker logs --tail 30 js
```

---

## Remaining Questions & Gaps

1. **Exact post-`runMergeDiscovery` `StatusFailed` count (`~15:10Z–15:15Z`):**
   - Because `runPendingRepoRetryPass` is still in progress (`22.35%` complete at `13:33:31Z`, projected to finish at `~15:05Z`), `runMergeDiscovery` has not yet executed. While our per-shard calculation (`3,701,018` unenumerated accounts -> `~3,034,530` active accounts based on each shard's observed active ratio) is tightly bounded by `com.atproto.sync.listHosts`, the exact post-discovery `Discovered` and `Errored` totals should be verified at `~15:15Z` right before executing the Option (b) container restart.
2. **Realized `steady_state` per-repo `DrainDurability` + `OnComplete` lock latency on `pd-balanced`:**
   - In `steady_state`, `RunFailedRepoRetry` runs with `r.completions == nil` (`retry.go:144`), meaning each completed repo executes `w.DrainDurability(ctx)` + `r.store.OnComplete(...)` serially under `w.drainMu.Lock()`. While `pd-balanced` SSD IOPS (`3,000` IOPS) comfortably supports the `~5.2 repos/sec` single-shard tail rate and `~200–300 repos/sec` 64-worker fleet rate, observing the first 10 minutes of `RunFailedRepoRetry` in `steady_state` (`~15:20Z–15:30Z`) will confirm the exact realized `steady_state` retry throughput.
