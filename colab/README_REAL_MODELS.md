# Catbird Intel real models — revision 3

This revision prepares English-scoped teacher labels and a five-component Ettin student. It replaces the active NLI fallback with a pinned `-c` checkpoint, records model/code/data-license provenance, separates outrage from criticism, and makes unsupported language an explicit unscored state. No revision-3 model inference, training or ONNX export has run. The governing [remediation plan](../../docs/execution-plans/2026-10-20-catbird-intelligence-review-remediation.md) defines the provisional evaluation rules; [decisions](../../docs/DECISIONS.md) retain the historical record.

## Immutable history and new identities

The complete verified revision-2 source bundle is preserved at `archive-bundles/revision-2/`, including its source manifest, V1_1 notebook and smoke receipt. Its manifest SHA-256 is `08af9e174b72344cdb05b3249be0fc9440c6bbde9e806dd333830b80fe9dd592`. The initial delivery remains under `archive-bundles/initial/`. Original notebooks, existing external checkpoints, failed exports and GPU pilot receipts remain unchanged. These new files are not retroactive attribution for old executions.

The new notebook is `Catbird_Intel_Real_Teachers_Ettin_V1_2.ipynb`; its source location is `catbird-lake/colab-v1/revision-3/`, and outputs use a new `runs/real-v1.2-*` directory. This is an unexecuted template. The historical bounded pilot wrapper and its prepared archives are revision-2 evidence; they are not a validated revision-3 GPU runner. No further GPU run is authorized by the completed pilot.

## Model and license identity

`model_licenses_v3.json` records pinned model revisions, declared model licenses, upstream code-license sources and hashes, and training-data licensing limitations. Every new teacher and student identity includes this provenance. The active fallback is `MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c` at `b2730f16019076bb0009481121efbe4705e0e378`, replacing the non-`-c` fallback used by the historical smoke. Old labels cannot be silently reused under this identity.

The fallback author describes `-c` training data as commercially friendly. Component metadata remains explicit: synthetic Mixtral data declares Apache-2.0; MNLI lists several licenses; FEVER repository metadata includes data and code licenses. This is not flattened into blanket legal clearance. Political DEBATE declares MIT for the model, while the inspected Pol_NLI dataset card has no declared license. Qwen and Ettin also lack a complete per-corpus license inventory in the inspected checkpoint cards. Public post access is not a blanket content license, and this project does not invent an implementation-code license where none is declared.

Primary sources: [fallback card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c), [Pol_NLI](https://huggingface.co/datasets/mlburnham/Pol_NLI), [Political DEBATE](https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0), [Qwen](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ), [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-68m). Exact metadata revisions and card hashes are retained in the inventory.

## Observed language coverage

`language_audit_v3.json` was computed from the `langs` metadata column of the sealed 150,000-post bank; no text classifier or model was run. It contains every raw combination, normalized tag counts, primary-language counts, empty/missing/invalid counts, and separate multiple-tag/multiple-language counts.

| Revision-3 scope | Actual rows |
|---|---:|
| Explicitly English primary subtags only, including en-US/en-GB | 90,603 |
| Missing language metadata | 18,273 |
| Mixed primary languages | 181 |
| Other declared languages | 40,943 |
| Total | 150,000 |

These are post declarations, not verified text-language judgments. A nonempty valid list whose every primary subtag is `en` is in scope. Missing, invalid, mixed and unsupported declarations are unscored. Regional variants of one language are distinguished from genuinely mixed primary languages in the report. All source rows remain in label/merge outputs, with `score_status`, `unscored_reason`, null scores and null raw logits where no model runs. Unscored is absence of evidence; downstream health filters must impose no model-based penalty and must not coerce it to a probability of zero. Separate moderation/visibility/deletion rules still apply.

Training excludes unscored rows from loss and reports the exact scope counts. The 32-row smoke selection remains the same first 32 original rows; it is not filled with additional posts after scoping. Joint training requires declared-English metadata on all three triplet roles; missing `anchor_langs`/`positive_langs`/`negative_langs` cannot silently become English. Earlier pair banks may therefore be structurally valid but unusable for this scoped model.

Reproduce the metadata-only report at a new output path:

```sh
PYTHONPATH=colab python -m real_pipeline.language \
  --input /path/to/phase3_health_training_150k_v1_unlabeled.parquet \
  --expected-sha256 4bea16ece379a1c2f0e8d559bca97357b9eb653cc3f8083681d984cd07e0b538 \
  --output /new/path/language-audit.json
```

## Targets and compatibility

Political DEBATE receives the post plus an explicit political-content hypothesis and returns binary NLI entailment. Qwen scores four separate rubrics using ten retained digit log probabilities and expected 0–9 severity divided by nine. NLI entailment and ordinal severity are not interchangeable calibrated probabilities. All model, rubric, language and software identities are retained. Changed source/dependencies refuse inference resume; metadata consumers preserve the producer environment without requiring identical local packages.

The new `catbird-ettin-components-v3` contract produces `health_component_logits` in this exact order:

1. politicalness
2. outraged
3. critical
4. engagement_bait
5. spam

Apply sigmoid to these logits. There is no trained heat head and no invented hostility head. A future heat/argument control requires a separately frozen aggregation, human rubric and calibration; it must not silently recreate `max(outraged, critical)`. The shared social head remains 256 dimensions with separately renormalized 128/64 prefixes. A five-output head adds 513 parameters to the old 512-hidden-size head by arithmetic; this revision has not measured a newly loaded full model.

Old four-output checkpoint keys and configurations are incompatible and are rejected. There is no automatic target or weight migration. Student run identity now binds component/language contracts, licensing, source hashes, dependencies and scoped counts. Export reuse is bound to the actual checkpoint digest and export identity. New ONNX filenames contain `components_v3`.

`train --objective health` explicitly trains health components without consuming graph pairs; the social head remains untrained. `--objective joint` requires a valid, in-scope v1 pair bank for a full run. Objectives use separate run identities: choose a new output directory when changing one. The 5,000-step schedule is provisional random sampling, not a convergence or complete-epoch guarantee.

## Review and evaluation

`review-sheet` can operate directly on the canonical raw holdout without model inference. Teacher and regex columns are withheld from the blind first pass. Existing review files and their status sidecars cannot be overwritten. Human fields remain null. The historical 600-post challenge set stays sealed; it is not silently repurposed as development data or enlarged.

`evaluation_contract_v3.json` and `python -m real_pipeline.evaluation --help` define the complete metrics-only evaluation interface. It consumes independently annotated labels, matched saved predictions, relevance judgments and explicit artifact/runtime identities; it performs no inference. The remediation plan's sample minima and provisional gates are design requirements, not claims of guaranteed power. Prevalence, author/session clustering, enrichment, missing labels and sampling strata affect what an interval means. Missing human evidence, an unresolved hostility instrument or absent target-host evidence cannot yield a qualification pass.

Freeze a versioned evaluation-contract copy outside this sealed source bundle before final data is examined. Its profile and required targets are predeclared: a politicalness-only health evaluation can qualify that target without an untrained social head or unrelated annotations; a joint profile additionally requires the 256/128/64 retrieval gates. Unselected targets cannot be claimed as qualified, and the profile cannot be chosen after seeing outcomes.

The original compact export's maximum-error gate of 0.03 remains a failed historical smoke criterion. Revision-3 export diagnostics retain that limit; independent trained-model promotion additionally requires the selected profile's decision-flip and paired human-metric gates. A low cosine error or small file alone establishes neither model quality nor serving suitability.

## Runtime preparation and remaining boundaries

Before any separately authorized run, verify the new source manifest and canonical bank/holdout seals. Use a separate CUDA runtime and a new output generation. The notebook still installs from requirement ranges; a runtime-specific hash lock must be prepared for a bounded execution. The Python-version-aware, hash-locked revision-2 pilot is historical evidence, not proof that revision 3 installs or runs unchanged.

Checkpoint saves use temporary files, sync, rename and integrity sidecars; recovery preserves damaged files and reconciles logs. Teacher shards and JSON manifests also sync temporary bytes before atomic rename, with shard publication preceding its manifest entry. A failure between those two publications leaves an orphan that must be preserved/archived manually; interrupted `.partial` files likewise block automatic resumption. A nonempty directory without a manifest cannot acquire a new identity. Tests exercise tiny local states. Hosted Drive/FUSE interruption and real optimizer-state recovery remain untested.

Structural verification uses existing real-row metadata, mocked teacher dispatch, tiny local Linear layers and numeric metric fixtures. It performs no pretrained loading, model forward pass, optimizer steps, GPU allocation or large-weight downloads:

```sh
python -m unittest discover -s colab/tests -v
python colab/real_pipeline.py train --help
PYTHONPATH=colab python -m real_pipeline.evaluation --help
```

The historical local run fit 32 posts with the non-`-c` NLI fallback; its four-output model had 68,278,020 parameters and training-probe BCE 2.015553 → 0.355486. FP32 ONNX passed same-row numerical comparison; the 69,433,199-byte compact int8 model failed at 0.06676513 maximum probability error. The later A100 pilot ran Qwen and Political DEBATE on those same 32 posts but did not train a student. None of these observations establishes revision-3 quality, full-bank throughput, language accuracy, a trained social encoder, or deployed CPU serving. The live scorer remains the regex baseline.
