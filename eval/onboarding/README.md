# Onboarding engine checks

Scripts that measure and exercise the live onboarding engine (`src/onboarding_live.ts`) on a real
serving bundle. None of them touches production; they read local files or talk to a local copy of
the service.

| Script | What it answers |
|---|---|
| `live_sim.ts <bundle> [pool]` | Static engine against live engine over random selections: how many follows, how many of the faces shown on a card are followed, how many picked scenes contribute nobody, and how many different accounts lead a scene across sessions. |
| `duel_probe.ts <bundle> <pool>` | The duels generated for a few realistic sets of picks, two sessions each, and how many scenes yield a full set of duels as a single pick. |
| `http_walk.ts <viewer did or anon> [base url]` | One pass through the flow over HTTP (catalog, scene detail, lineup, preview, duels), checked against an independent copy of the viewer's public follow list. |
| `load_check.ts <bundle> <pool> [sessions]` | Memory held per session and time for a viewer's whole flow, with many sessions open at once. |
| `dev_viewer_proxy.ts` | Stands in for the gateway on a developer machine: signs each forwarded request for one viewer, so the local service sees a signed-in viewer. |

## Running the service locally with the live engine

The files below are the candidate bundle and pool in the Mac lake
(`~/catbird-lake-local/s17-picker-v1-20261030/`).

```bash
export INTEL_INTERNAL_BFF_SECRET=<any 32+ byte value, local only>
INTEL_SERVING_BUNDLE_PATH=<lake>/intel_serving_v2.candidate-20261121.json \
INTEL_ONBOARDING_POOL_PATH=<lake>/intel_onboarding_pool_v1.candidate-20261122.json \
INTEL_ONBOARDING_ENGINE=live INTEL_SKIP_CLICKHOUSE=1 INTEL_REQUIRE_AUTH=0 PORT=3298 \
bun src/api.ts
```

`/health` then reports `onboarding_engine: "live"`, the pool version, and `onboarding_pool_skipped`
when a pool was supplied but built for a different catalog.

To act as a signed-in viewer, run the proxy with the same secret and send requests to it instead:

```bash
UPSTREAM=http://127.0.0.1:3298 PORT=3299 bun eval/onboarding/dev_viewer_proxy.ts
bun eval/onboarding/http_walk.ts did:plc:<a public account> http://127.0.0.1:3299
```

The viewer comes from the `x-debug-viewer-did` request header, or from `VIEWER_DID` when the proxy
is started with it. Use a public account that is not a team member's.
