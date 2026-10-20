# Offline RI/RM candidate builder — 20 October 2026

This change prepares two account features for controlled evaluation: audience-based Influence (RI) and per-post-type account profiles (RM). The existing production refresh uses a popularity heuristic for Influence and can turn missing content scores into reassuring mode statistics. A bounded local builder now computes explicit candidate aggregates from sealed, prevalidated inputs while preserving missing scores, incomplete audience coverage, deletion boundaries, and generation identity. Eight numeric structural tests and an actual CLI invocation pass; real input quality, model quality, and follow-mode usefulness remain unmeasured.

Governing plan: [review remediation, RI and RM](</Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-20-catbird-intelligence-review-remediation.md>). Integration owns the [decision log](</Users/joshlacalamito/Developer/Catbird+Petrel/docs/DECISIONS.md>); provisional implementation choices are listed below for its entry. No production refresh SQL or running graph process was touched.

## Files and invocation

- [ops/build_mode_influence_candidate.py](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/build_mode_influence_candidate.py>)
- [ops/test_mode_influence_candidate.py](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/test_mode_influence_candidate.py>)

```sh
PYTHONDONTWRITEBYTECODE=1 graph/.venv/bin/python \
  ops/build_mode_influence_candidate.py --print-input-contract

PYTHONDONTWRITEBYTECODE=1 graph/.venv/bin/python \
  ops/build_mode_influence_candidate.py \
  --manifest /path/to/sealed/input_manifest.json \
  --output-dir /path/to/new/candidate-directory \
  --bootstrap-replicates 200 --seed 0
```

The full machine-readable contract is both executable through `--print-input-contract` and preserved in [input_contract.json](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/mode_influence_input_contract_v1.json>). It requires no post text. Inputs are local JSONL tables for accounts, eligible post URIs, permitted communities, producer/baseline metadata, current live posts, nullable scores, current live engagements, pre-window interest vectors, and deletions. A separate current-state validation receipt binds every exact table hash, cutoff, upstream generation, account-state generation, visibility-policy generation, and required fold checks.

The builder rejects duplicate identities, mismatched hashes, stale score/post content identities, future events, interest generations that do not match the graph, interests computed after the event window begins, malformed null-score reasons, inconsistent eligibility/deletion universes, and unsupported community scope. Relative artifacts must stay within the manifest directory. Defaults cap each input at 100,000 rows and 256 MiB; an additional cap limits producer/type/dimension profile cells to 100,000.

The upstream receipt remains an assertion supplied by a separate validator. This command does not repair or independently prove state reconstruction. Even correctly sealed inputs produce a candidate marked `production_publishable: false`, `follow_gate_qualified: false`, and `feature_quality_measured: false`.

## RI: audience aggregate and separate popularity baseline

The primary aggregate sums each distinct eligible engager’s pre-window InterestedIn vector once per producer and then normalizes the producer’s observed community mass. Repeated interactions do not multiply that primary audience representation. A separately named comparison sums repeated like/repost interactions using weights one and two.

Interest vectors can be truncated or missing; they are never individually renormalized. Every producer exposes distinct engager count, engagers with/without interest rows, known and missing interest mass, weighted coverage, engagement event count, first/latest event time, and active engagement days. Each community reports its observed normalized share alongside lower/upper mass bounds relative to the full eligible engager denominator. These bounds allocate unresolved mass conservatively; they are not confidence intervals. Unlisted communities have an explicit possible missing-mass upper bound.

There are no impression logs, so exposure is explicitly unavailable. Engagement counts do not become exposure estimates. Only communities explicitly classified as permitted and non-sensitive enter this bounded candidate; individual audience interest profiles are not emitted.

The existing formula is preserved separately under `legacy_primary_knownfor_log_engagement_followers_baseline`:

```text
primary_score * log1p(primary_score * weighted_engagement_events
                     + 0.1 * followers_est)
```

It uses the same corrected eligible event universe for comparison. Missing primary membership or follower estimates produces a null baseline, with no invented fallback follower count.

## RM: exclusive types and unresolved scores

Authored posts contribute to exactly one category: reply takes precedence when a post also quotes; otherwise quote or original. Reposts are events belonging to the reposter, but their content scores are joined through the eligible subject post and matching content identity. Unresolved or ineligible subjects remain unscored with explicit reasons. Deletion-ledger matches suppress the affected events entirely.

Each producer/type/dimension reports event counts, scored/unscored counts, scored fraction, missing-reason counts, mean model score, score quantiles, and an exploratory account-day bootstrap interval when at least two scored days exist. All events from a sampled UTC day travel together in resampling. The interval describes the mean of supplied scores conditional on scored data; it is not calibrated prediction uncertainty, a missing-data correction, or evidence that an account is safe. With insufficient days or disabled resampling, uncertainty remains unestimated.

No score threshold, binary political/hostility label, “safe originals” judgment, role assignment, or follow-mode recommendation is manufactured. `assigned_follow_mode` stays null and follow-mode eligibility stays unevaluated.

## Artifacts and verification

The output directory must be new; existing directories are rejected. It contains:

1. `mode_profiles.jsonl`
2. `audience_influence.jsonl`
3. `popularity_baseline.jsonl`
4. `candidate_manifest.json`, written last

The seal includes exact source and input hashes, graph/scoring identities, the event window, interest cutoff, scoring rubric/model identities, parameters, counts, policies, audit exclusions, and artifact hashes. Source and inputs are rechecked before sealing. No network fetch, model inference, production read/write, GPU use, or history mutation occurs.

Verification command:

```sh
PYTHONDONTWRITEBYTECODE=1 graph/.venv/bin/python \
  -m unittest discover -s ops -p 'test_mode_influence_candidate.py' -v
```

Result: **8 tests passed**, including actual CLI execution, artifact/source hash checks, no-overwrite behavior, duplicate/future/content-identity rejection, deletion suppression, unresolved repost subjects, missing audience mass, null-score preservation, and bootstrap behavior. All examples are numeric structural fixtures with no fabricated post text. [Exact source/test receipt](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/review_mode_influence_20261020.json>) and [test output](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/review_mode_influence_tests_20261020.txt>) are preserved externally.

Builder SHA-256: `4d4e039050821b13482cb7024cf628f5be7cf5a678eac31255e239eb7b020f9a`.

Test SHA-256: `7a555aa9a454a1fcba4b597d374be2278c9e72d8aa13ee5bdc139b1cbf118aa2`.

Provisional choices for integration’s decision log: distinct-engager primary RI with a separate repeated-interaction comparison; pre-window interest cutoff; reply-first mutually exclusive post types; exploratory account-day mean-score bootstrap with 200 default replicates; bounded JSONL input/output contract. These choices are reversible before merge and do not qualify feature quality. The next evidence step is a sealed real-input candidate plus held-out account assessment and comparison against the named popularity baseline.
