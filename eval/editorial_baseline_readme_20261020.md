# Editorial baseline preparation — 20 October 2026

This prepares the existing curated catalog for the proposed early onboarding release (RD0). It preserves historical editorial identities and creates a concrete review queue without claiming an independently discovered graph or current eligibility. The source contains 567 account memberships (562 unique DIDs), 40 feed references and 60 pack references across 26 editorial scenes. All 667 candidates remain ineligible for release until current safety checks and editorial review pass.

Source: `scripts/onboarding/data/catbird_onboarding_packs.json`, schema `2026-10-17.v3`, generated `2026-10-17T16:22:02.332545+00:00`. The historical curation plan is `docs/execution-plans/2026-10-17-catbird-onboarding-subculture-curation-plan.md`; current rollout decisions belong to `docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md`. Its historical prose counts (566/561) differ from the present source bytes (567/562); the manifest counts the actual file. Historical names are carried as historical editorial names, not community-quality findings. No biographies, avatars, post text, quality scores or historical visibility judgments are promoted into this artifact.

## Reproduce

```sh
python3 -B catbird-intel/ops/prepare_editorial_baseline.py \
  --source scripts/onboarding/data/catbird_onboarding_packs.json \
  --output-dir catbird-intel/eval
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s catbird-intel/ops/tests -p test_editorial_baseline.py -v
```

The CLI performs no network calls and refuses changed bytes under existing output names. Its output content hash determines versioned filenames. Source SHA-256, source JSON pointers, historical provenance fields and unknown curator identity are retained. Handles in starter-pack URLs remain unresolved record references; they are never mislabeled canonical DID URIs. The queue is suitable for a later bounded read-only identity and record reconciliation, but does not itself perform it.

## Artifacts and evidence

- `editorial_baseline_v1_3c211a51e3e82efc.json`: SHA-256 `3c211a51e3e82efc80b15eaa4730e8b1f5c4d6dfa0abe8d7c1178dd03b31eee7`.
- `editorial_baseline_v1_3c211a51e3e82efc.review.jsonl`: 667 rows; SHA-256 `4c08b680afb2b49d160fd858e762e89b84c842345c02cee2eada534380c5dbeb`.
- `editorial_baseline_v1_3c211a51e3e82efc.receipt.json`: machine-readable counts and hashes.

An independent read-back recomputed both hashes and verified 667 entries, 26 scenes and zero release-eligible candidates. Four unit tests passed: historical evidence cannot grant release; source bytes change generation identity; changed bytes cannot overwrite outputs; malformed pack references are rejected.

## Release boundary

The review-only action types are `follow_account`, `pin_feed`, and `inspect_pack`, all non-executable. The account-count proposal is 25 by default and 50 maximum. Inspecting a pack does not authorize following its members. Per-author repost hiding requires explicit user choice; historical role or dual-mode tags do not authorize it. A topic-feed intent without a concrete feed URI must not become a pinned feed or an account follow.

Before release, resolve current identities and pack/feed records, apply live deletion and visibility policy, reconcile moderation and adult-content eligibility, respect viewer blocks/mutes and reference-list opt-outs, and obtain curator approval. A public profile fetch alone cannot satisfy these checks. No live requests, app deployment, user actions, graph modifications, seed injection, or release qualification occurred here.
