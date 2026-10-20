# Graph evaluation follow-up — 20 October 2026

Catbird’s graph work assigns accounts to communities using shared followers, then checks whether independently selected starter-pack members tend to land together. That check was vulnerable to large packs, popular accounts, missing graph coverage, and a misleading external comparator. The evaluator now exposes those effects through separate metrics, a primary-partition null, and an auditable per-pack artifact; unavailable measurements remain explicitly unperformed. The original archive fold and its waiting continuation driver remain running, so this report qualifies implementation and numeric-fixture execution, not the quality of a completed real graph. The next graph benchmark must use a frozen, source-correct generation before any community or encoder promotion.

Governing review: [community-intelligence review dossier, section 5](</Users/joshlacalamito/Developer/Catbird+Petrel/docs/design-docs/2026-10-20-catbird-community-intelligence-review-dossier.md:264>). Integration owns the dossier and [decision log](</Users/joshlacalamito/Developer/Catbird+Petrel/docs/DECISIONS.md>); this worker did not edit either. The provisional hub threshold described below needs an overridable-before-merge decision entry there.

## Changed code and executable evidence

Owned changes are limited to:

- [graph/eval_phase1.py](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/eval_phase1.py>): pack metrics, explicit missing-input contracts, fold-count diagnostics, evaluator source observation, and per-pack output.
- [graph/test_eval_diagnostics.py](</Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/test_eval_diagnostics.py>): eight numeric structural tests, including an actual evaluator CLI invocation.

No follow inputs, similarity algorithm, clustering parameters, memberships, seed identities, production tables, or training bank were changed by this worker. There were no VCS operations, production writes, GPU work, process restarts, or duplicate folds.

Before editing, the prior evaluator was copied byte-for-byte to [pre-edit source custody](review_graph_source_custody_20261020.json). Its SHA-256 was `b6d83f2601aa3f06802d6b8024403265ca4a006db8c71459d99ee6b39553e8fb`.

The post-edit evaluator SHA-256 is `c28a3f8542804d621e48be519fe874d5cf1aa9d86bb41c4f516b9b30ecd5724d`; the new test file SHA-256 is `9bd4ca32f5251d5d651f4cb98546c9f2d813826f3abc4721454124d183e87b27`. Exact copies and process observations are in [post-edit source custody](review_graph_source_custody_20261020.json).

The final local command was:

```sh
PYTHONDONTWRITEBYTECODE=1 graph/.venv/bin/python -m unittest discover -s graph -p 'test*.py' -v
```

It passed 23 tests in 0.311 seconds, including the concurrent source-generation tests another worker had added. Fourteen are the evaluator diagnostics and existing graph guards. [Complete test output](review_graph_structural_tests_20261020.txt) distinguishes the individual tests. `eval_phase1.py --help` also confirmed both new hub-policy flags exist.

The CLI test uses numeric account/community/pack records, blank profile fields, and no post text. It checks that the executed evaluator emits its source hash, seed-manifest hash, per-pack artifact hash, and explicit unavailable follow-capture state. Its generated labels and recall values are structural fixtures, not empirical results. All fixture directories remain under `/tmp/catbird-*`.

At the 10:33:16 UTC observation, fold PID 95795 had run for approximately 3 hours 45 minutes and driver PID 10905 for approximately 3 hours 29 minutes. Both commands were intact. The driver began before these edits; its in-memory source cannot be retroactively identified by hashing the current driver file. Its later evaluator subprocess will load the then-current evaluator source. The evaluator now observes its own file hash at entry and rejects a changed source file at completion, while labeling that evidence as a source observation rather than interpreter or driver attestation.

## G1: pack adoption is a confound; classification remains unperformed

A person using a pack’s bulk-follow action can create shared-follower evidence among that pack’s members. Consequently, agreement between the follow graph and that pack is not necessarily independent evidence of a preexisting organic community. Current evaluation output now includes a `bulk_follow_adoption` object with `status: not_performed`, the observed archive schema, absent contract columns, and a concrete classification contract.

The projection preserves `did`, `rev`, `rkey`, `operation`, `subject`, and `created_at`, among other fields. These do not establish that rows sharing `(did, rev)` were observed as one complete live commit. Backfilled current-state records can share a repository-head revision even if their follows were created at different historical times. User-supplied `created_at` also cannot establish trustworthy commit timing. The new code deliberately does not scan a billion follow rows to label such groups as adoption.

The revised event contract requires source kind and instance, segment generation, commit CID, repository revision, record operation, operation index/count, verified commit completeness, observation time, backfill status, and synchronization state. The existing account DID and record subject are also necessary. A detector can then group verified live creates, compare them with pack membership at that time, and report the matched subject count and fraction. Without client-action evidence the classification should be named a “pack-matching follow burst,” not confirmed intent.

Adoption-stratified recall remains pending that input. An organic-graph experiment must be a separately named generation with its own edge inventory and hashes. No suspected adoption edges were removed from the running candidate.

## G2 and G6: pack weighting, coverage, hubs, and null

The old top-1/top-3/scene result keys retain their exact legacy population and pair weighting: packs with 3–200 represented members, with every pair counted per pack. Preserving these keys avoids silently changing downstream consumers. The report explicitly calls this the legacy population.

The new diagnostics select packs with 3–200 **raw observed** members before consulting KnownFor, the account-to-community membership table. This includes packs with zero, one, or two represented accounts. It prevents the evaluator from hiding uncovered packs simply because they cannot form enough represented pairs. Missing observed list membership, undersized packs, and oversized packs are reported separately; this remains coverage of the observed archive, not proof of global completeness.

For each membership view—primary community, up to three communities, and coarse scene—the evaluator reports:

| Metric | Numerator and denominator | Purpose |
|---|---|---|
| Pair micro, conditional | Recovered pairs / represented pairs | Preserve the familiar covered-account question |
| Pair micro, end to end | Recovered pairs / raw observed pairs | Expose missing membership coverage |
| Per-pack macro, conditional | Mean within-pack conditional recall over packs with represented pairs | Give each evaluable pack equal influence |
| Per-pack macro, end to end | Mean raw-pair recall over packs with raw pairs | Include uncovered members in pack-level success |
| Inverse-size pair weighting, conditional | Sum `hits/(n−1)` / sum `represented_pairs/(n−1)` | Reduce quadratic pack-size dominance |
| Inverse-size pair weighting, end to end | Sum `hits/(n−1)` / sum `raw_pairs/(n−1)` | Combine reduced size dominance with coverage |

Here `n` is raw pack size in the view being evaluated. Inverse-size weighting gives a fully represented pack total pair mass `n/2`; it is different from equal-pack macro averaging. The numeric test with sizes 150 and 8 verifies that these three weighting choices produce different answers. Repeated pairs in different pack records still count repeatedly, which is disclosed. Duplicate lists and related pack creators therefore remain dependence concerns.

The `dehubbed` view removes an account from metric computation if it occurs in at least two live pack records and at least 10% of all live pack records. This is a provisional, data-derived rule chosen before inspecting quality outcomes. It does not name an account, consult seeds, or alter graph inputs. The report lists the actual hub DIDs and frequencies, removed member occurrences, surviving raw and represented pair counts, and packs with nonzero denominators. The base eligible pack population is fixed for comparison; after removal, zero-pair packs are retained in population counts and excluded only from undefined ratios. Weights and the null universe are recomputed after removal, so differences between views include those explicit denominator effects.

For the primary partition only, let `N` be all represented accounts and `n_c` the number whose first membership is community `c`. The exact same-size permutation expectation is:

```text
p_same = sum_c n_c (n_c - 1) / (N (N - 1))
```

It holds the actual cluster-size distribution fixed while randomly assigning labels across the represented account universe. The report includes the size histogram, expected recall, observed-minus-expected, and observed/expected when defined. Coverage scales the end-to-end expectations. A giant single cluster therefore has expected primary recall one; high observed recall alone is not evidence beyond this null.

This null is not applied to overlapping memberships or scenes. It does not preserve degree, language, popularity, adoption, or pack dependence; it is an expectation diagnostic, not a p-value. A later benchmark should additionally use a degree-aware null and creator/list-aware resampling rather than treating all pair occurrences as independent.

The evaluator writes `pack_recall_diagnostics.parquet` next to its report, using bounded streaming batches. Each row identifies the starter pack, list, base/dehubbed view, raw and represented member/pair counts, and all three hit counts. The JSON report binds its row count and SHA-256. Integration should preserve this artifact when packaging or uploading the graph evidence; hashing only the aggregate report does not make an absent row-level artifact available to reviewers.

## Precision, reconstruction, and the meaning of edge capture

The SimClusters paper’s label-free check reconstructs an undirected input graph: two nodes predict an edge when they share a community. Precision is the observed fraction of predicted edges, and recall is the recovered fraction of observed input edges. Its nearly 70% statement describes communities’ capture of bipartite edges, not the fraction retained by a producer threshold. [Primary paper, Appendix A.1 and graph-reduction discussion](https://venusatuluri.github.io/KDD_SimClusters_paper.pdf).

The current evaluator has no pinned compatible reconstruction reference or declared held-out edge task, so `edge_reconstruction.status` remains `not_performed`. The output supplies the definitions and exact required inputs. Pack nonmembership is not treated as a negative edge, and pack recall is not relabeled precision. In-sample reconstruction, if later added, must be named descriptive fit; it is distinct from predicting held-out follows.

There is one count-based diagnostic the running fold can legitimately support. `eval_follow_edge_capture` reads `stats.tsv`, `producers.tsv`, `edges.bin`, and `input_provenance.json`. It validates dense producer IDs, producer count, summed follower degrees, binary edge bytes, frozen count agreement, and SHA-256 for the two text files. It does not reread the potentially large binary merely to claim a fresh full-content hash; that limitation is explicit.

Its ratio is:

```text
producer_edges / follow_edges
```

The denominator precedes the follower-degree cap; the numerator includes both that cap and the minimum eligible-producer-follower threshold. Accordingly, the result is named **joint consumer-filter and producer-selection edge retention**. Pure producer-selection coverage needs a missing intermediate denominator: all edges from eligible followers before applying the producer threshold. Bipartite community capture additionally needs consumer community memberships, which are not exported. Both remain unperformed, and the paper’s 70% is not a target for the available ratio. These counts describe the current candidate’s own account-status filtering; they cannot validate authoritative current state.

## G8: corrected external baseline and data-source claims

Smith et al. explicitly report 503 communities obtained by Leiden on the unweighted pairwise projection of the largest starter-pack component. The paper also reports approximately 48% pack membership for `@bsky.app` and publishes SOMAR accession `ICPSR300499`, including a roughly 2.4-billion-edge follow network. It describes CC-BY distribution with an additional no-reidentification condition. [Primary full paper, Table 4, community methods, and Data Records](https://pmc.ncbi.nlm.nih.gov/articles/PMC13066036/).

The [authors’ script](https://github.com/nwlandry/a-blue-start/blob/main/get_starterpack_clustering.py) directly builds a deduplicated clique expansion from starter packs and applies Leiden modularity with seed zero. Thus existence and method are established. This worker did not reproduce the partition or download the SOMAR data; current account access is unverified. Its anonymous node identifiers cannot be joined to local DIDs, and no reidentification was attempted.

Inference for Catbird: because the comparator is constructed from pack co-membership, it cannot be independent pack-recall ground truth. It can support an anonymous algorithm benchmark inside its own universe, or a clearly labeled pack-trained reference. The evaluator now records the publication/method and `replicated_here: false`, while retaining unavailable/null comparison fields. The dossier’s earlier “existence unverified” wording should be replaced accordingly.

## G9 and proposed R2a benchmark

R2a means the first controlled graph benchmark after input reconciliation. Its purpose is to choose a useful community representation using frozen evidence, before training an encoder to reproduce that representation.

1. **Freeze one eligible graph generation.** Require a source-aware input inventory, per-source cursor/generation handling, commit/backfill distinctions, record deletion rules, and current account-state semantics. Preserve the running archive-only candidate for comparison. Do not overwrite it or retrofit its provenance. Record edge and account universes, producer/follower thresholds, all intermediate denominator counts, seeds, library versions, source hashes, and hardware/runtime limits.
2. **Predeclare independent evaluation populations.** Keep seed cohorts evaluation-only. Split packs by creator/list or an explicitly chosen dependence unit, retain undercovered packs, and preserve raw-size strata. Freeze hub policy and adoption detector thresholds before scoring. Add blinded human review pairs and hard separations from a separate sample when available; do not infer those labels from pack titles or model output.
3. **Compare the fixed current construction with alternatives under equal scope.** Include the existing weighted Leiden primary partition and its similarity-derived overlapping memberships. If a SimClusters-style overlapping factor model is implemented, compare it as a separate algorithm rather than calling the existing heuristic equivalent. Require the same graph universe, explicit resource ceilings, recorded failures, and repeated seeds for stability. Do not retune against the eight collision traps.
4. **Run a primary-only membership ablation first.** The new evaluator already reports primary versus overlapping recall and their gap. Compare coverage, macro and weighted pack recovery, same-size-null residuals, independent edge reconstruction/held-out prediction where available, and blinded interpretability. Lower primary recall can be acceptable if precision, stability, and usable communities improve; overlap is not a requirement to be forced into every account.
5. **Preserve producer embeddings as a separate concept.** If the primary partition is the better community product, retain any justified continuous producer similarity representation independently for retrieval. Do not rename up to three borrowed neighbor labels as a learned producer embedding. Encoder ablations should state whether they learn primary communities, overlapping memberships, or independent pairwise similarity.
6. **Add the adoption sensitivity experiment when its inputs qualify.** Score pack-matching follow-burst strata on the unmodified generation, then construct a separate graph with the declared suspect edges excluded. Report both the removed edge fraction and producer/account attrition. Differences are sensitivity evidence, not proof of what the network would have looked like without packs.
7. **Use SOMAR only within its permitted anonymous benchmark universe.** First verify actual account access and dataset manifest, then estimate download, RAM, and CPU requirements from a bounded smoke test. Do not assume the repository’s full-graph loading path fits this machine. No spending or large new download was initiated here.
8. **Publish one comparison bundle.** Include full distributions, null definitions, size/degree/coverage/adoption strata, per-pack rows, edge-reference identity, human-review counts where available, runtime and peak memory, and failures. Decide on community representation and encoder targets only after these results exist. There is presently no numerical win threshold justified by the available evidence.

The actionable next integration steps are to log the provisional hub policy, correct the dossier’s external-baseline wording, carry the new row-level artifact with the eventual graph report, and let the existing driver complete. Graph quality, source correctness, adoption-stratified metrics, independent precision, and product-visible behavior remain open.
