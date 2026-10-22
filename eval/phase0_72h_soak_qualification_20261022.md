# Phase 0 72-Hour Soak Qualification Report (`netcup`)

- **Soak Window:** `2026-10-19T13:00:00Z` → `2026-10-22T13:00:00Z` (`72.00` hours)
- **Measured At:** `2026-10-22T13:03:27Z` – `2026-10-22T13:11:30Z` (`72.05` hours elapsed)
- **Governing Contracts:**
  - [`docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md#L89-L93) (§4 Phase 0 Done Criteria)
  - [`docs/design-docs/2026-10-20-catbird-intelligence-protocol-and-soak-contract.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/design-docs/2026-10-20-catbird-intelligence-protocol-and-soak-contract.md) & [`catbird-intel/ops/evaluate_soak_receipt.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/evaluate_soak_receipt.py)
- **Machine-Verifiable Receipt:** [`catbird-intel/eval/phase0_72h_soak_qualification_20261022.json`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/phase0_72h_soak_qualification_20261022.json)
- **Overall Qualification Verdict:** **`UNQUALIFIED`** (`2 / 4` Master Plan Done Criteria PASS, `2 / 4` FAIL; `7 / 12` `evaluate_soak_receipt.py` gates PASS, `3 / 12` FAIL, `2 / 12` UNPERFORMED)

---

## 1. Master Plan §4 Phase 0 Done Criteria Summary

| # | Done Criterion | Verdict | Summary of Live Evidence (`2026-10-22 13:03Z–13:11Z`) |
|---|---|---|---|
| **1** | **72h continuous ingest (gap ≤ 60s after reconnect)** | **FAIL** *(Heartbeat PASS; Stream Lag FAIL)* | Process heartbeat logged **`25,916 / 25,920`** expected 10s intervals (`99.985%`), `health_gaps_over_60s = 0` (`longest_gap_s = 0`), and `8,640 / 8,640` (`100.0%`) in the final 24h (`zero_event_24h = 0`). However, reported stream lag exceeded `60,000 ms` on **`404` intervals in 72h** (`61` in final 24h) across 4 episodes (`max_lag_ms = 2,228,309 ms = 37.1m` on Oct 19; `max_lag_ms = 335,683 ms = 5.59m` in final 24h on Oct 21 `17:42:42Z..17:52:42Z` following a `17:33:28Z` WebSocket `1006` reconnect). Also uncovered a cross-source `seq` collision (`us-west` vs `us-east`) shadowing `206` `account_status` and `173` `identities` rows. |
| **2** | **Timed delete propagation ≤ 5s across every serving table** | **FAIL** *(Base Tables PASS; Derived Serving Tables FAIL)* | Static delete invariant across all 10 `ReplacingMergeTree` base tables (`engagements`, `posts`, `follows`, `blocks`, `listitems`, `lists`, `starterpacks`, `feedgens`, `profiles`, `visibility`) against `3,782,404` delete rows has **`0` visible deleted records** under `FINAL`. Live timed natural-delete probe (`n = 125` live `app.bsky.feed.post` deletes, 100ms polling) measured wire `indexed_at` → hidden in `catbird.v_posts_live` at **`p50 = 1,895 ms`, `p95 = 2,864 ms`, `p99 = 3,049 ms`, `max = 3,065 ms` (`<= 5,000 ms`)**. However, derived serving tables (`catbird.item_communities`, `catbird.community_directory`, and `catbird-intel-api` in-memory cache) refresh only every 15 minutes (`900s >> 5s`): **`15 / 125`** newly deleted probe posts and **`10,831`** total deleted posts remain visible in `catbird.item_communities FINAL`, plus **`15`** inactive accounts in `catbird.community_directory FINAL`. Controlled test-account write+delete probe is **UNPERFORMED**. |
| **3** | **14-day steady-state hot-window disk projection ≤ 30 GB** | **PASS** | Current `catbird` DB active disk after `72.05h` is **`4.73 GiB`** (`5,074,195,721` bytes: `engagements = 2.62 GiB`, `posts = 1.36 GiB`, `follows = 395.73 MiB`, `deletions = 108.18 MiB`, `blocks = 58.13 MiB`, `item_communities = 56.21 MiB`, `other_records = 52.80 MiB`). 14-day linear projection: **`hot_14d = 18.81 GiB` (`20.20 GB <= 30 GB`)**, **`state_growth_14d = 3.20 GiB`**, **`total_14d = 22.00 GiB` (`23.62 GB <= 30 GB`)**. Root filesystem has **`175.7 GiB` (`34.96%`) free**. |
| **4** | **Production service p99 latencies unchanged vs `2026-10-19` baseline** | **PASS** *(minor `svc:3006` / `clickhouse` p99 drift noted)* | All 12 production services (`catbird-mls-server`, `catbird-nest-prod`, `catbird-pds`, `catbird-firehose`, `caddy`, `postgresql`, `swan`, `catbird-feedgen`, `catbird-feed-ingest`, `catbird-intel-ingest`, `catbird-intel-api`, `clickhouse-server`) remain `active` with `0` new service errors. `deploy/prod-latency-probe.sh` (`n = 200` each, steady-state Run 2 at `13:11:22Z`) vs `2026-10-19 13:00Z` baseline: `mls-ds:3001/health` `p50=1.1ms p99=3.2ms` (baseline `p50=1.0ms p99=2.4ms`, `1.33x`); `svc:3006/health` `p50=0.8ms p99=2.8ms` (baseline `p50=0.8ms p99=1.3ms`, `+1.5ms` / `2.15x`); `clickhouse:select1` `p50=2.0ms p99=3.9ms` (baseline `p50=1.9ms p99=2.6ms`, `1.50x`). |

---

## 2. Detailed Criterion-by-Criterion Evidence

### 2.1 Criterion 1: 72-Hour Continuous Ingest & Reconnect Lag (`FAIL`)

- **Heartbeat Continuity (`PASS`):**
  - `catbird.ingest_health` recorded `25,916` rows between `2026-10-19 13:00:06.432 UTC` and `2026-10-22 12:59:56.340 UTC` (`99.9846%` of `25,920` expected 10s intervals; `0` gaps `> 60s`).
  - Final 24h (`2026-10-21 13:00:00Z` → `2026-10-22 13:00:00Z`): `8,640 / 8,640` intervals (`100.0%`), `zero_event_24h = 0`, `p50_lag_ms = 141 ms`, `p99_lag_ms = 2,828 ms`, `p99_flush_ms = 226 ms`, `max_flush_ms = 840 ms`, `max_rss_mb = 805 MB`.
- **Stream Delivery Lag (`FAIL`):**
  - Across the 72h window, `404` health intervals (`61` in the final 24h) had `lag_ms > 60,000`:
    1. `2026-10-19 13:06:06Z..13:19:43Z`: `81` samples (`max_lag_ms = 848,702` / `14.1m`).
    2. `2026-10-19 14:17:33Z..14:55:17Z`: `228` samples (`max_lag_ms = 2,228,309` / `37.1m`) triggered by two `MalformedError: Invalid string length` exceptions (`RangeError` inside `@atcute/firehose` on `jetstream.us-east.bsky.network`), cursor rewind (`-5,000,000 us`), and failover to `jetstream.us-west.bsky.network`.
    3. `2026-10-20 06:23:20Z..06:28:50Z`: `34` samples (`max_lag_ms = 149,261` / `2.49m`).
    4. `2026-10-21 17:42:42Z..17:52:42Z` (final 24h): `61` samples (`max_lag_ms = 335,683` / `5.59m`), occurring ~9 minutes after the `2026-10-21 17:33:28Z` WebSocket `code=1006` abnormal disconnect/reconnect.
  - **Clock Caveat:** `catbird-intel/src/ingest.ts` computes `lag_ms` from `record.createdAt` (client-supplied display clock) rather than relay `witnessedAt` / `time_us`.
- **Cross-Source Sequence Inversion Defect (`FAIL`):**
  - During the `2026-10-19 14:18Z..14:55Z` failover to `jetstream.us-west.bsky.network`, `us-west` emitted `time_us` sequence numbers around `27,154,623,469`, which is `~848,000,000` higher than `jetstream.us-east.bsky.network`'s `seq` on `2026-10-22 13:08Z` (`26,306,655,634`).
  - Because `catbird.account_status` and `catbird.identities` use `ReplacingMergeTree(seq)` without normalizing `seq` across Jetstream instances, **`206` rows in `catbird.account_status`** and **`173` rows in `catbird.identities`** written during that 37-minute `us-west` window have `seq > 27,000,000,000` and shadow any subsequent `us-east` updates for those DIDs until `us-east`'s `seq` catches up (~10 days).

### 2.2 Criterion 2: Timed Delete Propagation ≤ 5s Across Every Serving Table (`FAIL`)

- **Base Tables & `catbird.v_posts_live` (`PASS`):**
  - Static check across all 10 `ReplacingMergeTree` tables (`engagements`, `posts`, `follows`, `blocks`, `listitems`, `lists`, `starterpacks`, `feedgens`, `profiles`, `visibility`) against `3,782,404` rows in `catbird.deletions`: **`0` visible deleted records** under `FINAL`.
  - Live timed natural-delete probe (`2026-10-22T13:09:47Z` → `13:10:41Z`, polling every `100 ms` across `n = 125` newly arriving `app.bsky.feed.post` deletes):
    - Wire `indexed_at` → hidden in `catbird.v_posts_live`: **`min = 796 ms`, `p50 = 1,895 ms`, `p95 = 2,864 ms`, `p99 = 3,049 ms`, `max = 3,065 ms`** (`<= 5,000 ms` gate).
    - `0 / 125` newly deleted posts were visible in `catbird.v_posts_live`.
- **Derived Serving Tables & API Cache (`FAIL`):**
  - `catbird.item_communities`, `catbird.community_items`, and `catbird.community_directory` are rebuilt by `catbird-intel-refresh.timer` every **15 minutes (`900s`)**, and `catbird-intel-api` (`src/api.ts`) caches `community_items` in memory for 5 minutes without a live read-time anti-join against `catbird.deletions` or `catbird.account_status`.
  - In the live natural-delete probe, **`15 / 125`** newly deleted posts remained visible in `catbird.item_communities FINAL`.
  - Across the full table, **`10,831` deleted posts** remain visible in `catbird.item_communities FINAL` and **`15` inactive/deleted accounts** remain visible in `catbird.community_directory FINAL`.
- **Controlled Test-Account Write+Delete Probe (`UNPERFORMED`):**
  - Not executed because explicit test-account write authorization was not granted.

### 2.3 Criterion 3: 14-Day Steady-State Hot-Window Disk Projection ≤ 30 GB (`PASS`)

- **Active ClickHouse Disk (`72.05h` elapsed):**
  - `engagements`: `2.62 GiB` (`66,356,317` physical rows, `25` active parts)
  - `posts`: `1.36 GiB` (`9,270,562` physical rows, `20` active parts)
  - `follows`: `395.73 MiB` (`7,780,733` physical rows, `132` active parts)
  - `deletions`: `108.18 MiB` (`3,795,770` physical rows, `14` active parts)
  - `blocks`: `58.13 MiB` (`1,039,205` physical rows, `114` active parts)
  - `item_communities`: `56.21 MiB` (`3,077,601` physical rows, `5` active parts)
  - `other_records`: `52.80 MiB` (`941,769` physical rows, `14` active parts)
  - Total `catbird` DB active disk: **`4.73 GiB`** (`5,074,195,721` bytes)
- **14-Day Linear Projection:**
  - `hot_14d_projection`: **`18.81 GiB`** (`20.20 GB <= 30.0 GB`)
  - `state_growth_14d`: **`3.20 GiB`**
  - `total_14d_projection`: **`22.00 GiB`** (`23.62 GB <= 30.0 GB`)
- **Filesystem Headroom (`df -B1 /`):**
  - Total: `539,792,977,920` bytes (`502.7 GiB`); Used: `329,053,433,856` bytes (`306.5 GiB`); Available: **`188,692,852,736` bytes (`175.7 GiB` / `34.96%` free)**.
  - Projected used fraction with 14d DB growth + 50 GB scratch margin: **`68.40% <= 70.0%`**.

### 2.4 Criterion 4: Production Service p99 Latencies Unchanged (`PASS`)

| Probe Target | `2026-10-19 13:00Z` Baseline | `2026-10-22 13:03:57Z` (Run 1, concurrent CH verify) | `2026-10-22 13:11:22Z` (Run 2, steady state) | Steady-State p99 Ratio |
|---|---|---|---|---|
| `mls-ds:3001/health` (`n=200`) | `p50=1.0ms p95=1.5ms p99=2.4ms max=3.0ms` | `p50=1.2ms p95=1.9ms p99=2.7ms max=4.4ms` | `p50=1.1ms p95=1.8ms p99=3.2ms max=5.3ms` | `1.33x` (`+0.8ms`) |
| `svc:3006/health` (`n=200`) | `p50=0.8ms p95=1.0ms p99=1.3ms max=1.8ms` | `p50=0.8ms p95=1.4ms p99=3.2ms max=11.7ms` | `p50=0.8ms p95=1.4ms p99=2.8ms max=8.0ms` | `2.15x` (`+1.5ms` abs) |
| `clickhouse:select1` (`n=200`) | `p50=1.9ms p95=2.3ms p99=2.6ms max=3.2ms` | `p50=2.0ms p95=2.9ms p99=3.9ms max=7.6ms` | `p50=2.0ms p95=2.7ms p99=3.9ms max=5.9ms` | `1.50x` (`+1.3ms`) |
