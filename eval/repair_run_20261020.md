# Catbird Intel repair — 2026-10-20

Catbird Intel uses offline communities and text models to recommend Bluesky accounts and posts. Its previous completion reports overstated stub model quality and leaked seed edges into the old graph. This repair corrects those reports, isolates invalid artifacts, exports real data, and supplies a real teacher/student training pipeline. The graph rebuild and human-reviewed model qualification are separate gates; uploading code does not satisfy either.

Governing documents: [master plan](../../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md) and [decision log](../../docs/DECISIONS.md). Netcup production and ingest remain unchanged through the soak ending **2026-10-22 13:00Z**.

## Data and Drive

- Real training bank: **150,000 posts / 91,165 authors**, 22,436,763 bytes. SHA-256 `4bea16ece379a1c2f0e8d559bca97357b9eb653cc3f8083681d984cd07e0b538`.
- Real English holdout: **600 posts / 600 authors**. Sampling strata 120 political-keyword, 120 heated-nonpolitical-keyword, 120 collision-keyword, 80 quotes, 80 replies, 80 originals. Every quota filled; no padded rows. SHA-256 `fd57d61c490cf11860c74341f4c5d16d884d65ae0a0b9dd200b867c52b6b9792`.
- Zero holdout-author overlap against 148,595 authors across both old banks and the new bank. [Export receipt](phase3_v1_export_manifest.json). This is an enriched sampling scheme, not a population estimate or a guarantee of viewpoint balance.
- [Training bank on Drive](https://drive.google.com/file/d/17LSL-9fZSSbRtUqO8rhtPKKN5wa7j3AX/view?usp=drivesdk) and [holdout on Drive](https://drive.google.com/file/d/1ZJk3dcvTl-FbKCXmJup7vdiUHLENIUIj/view?usp=drivesdk). Checksums and export manifests are stored alongside the files.
- [Quarantine](https://drive.google.com/drive/folders/1KDAHX3mBnmJeHy7vTAnwso6nBWFpkhR8): 23 moved files/folders, including 8 preserved vector shards, 976,037,311 bytes. No deletion. Phase 6 report was absent from the source models folder. [Move journal](drive_quarantine_20261020.jsonl) preserves IDs, prior parents and checksums for recovery.
- The uploader refuses duplicate names, differing existing bytes, quarantined paths and known invalidated filenames. Normal synchronization checks the sealed bank/holdout receipts and actual author isolation; it does not rediscover old model artifacts.

## Resource boundaries

The GCE VM was read back as RUNNING on n2-highmem-32 Spot; no restart or configuration change occurred. The Colab CLI returned 0 active assignments, 4,997.64 compute units and 0.00 units/hour. The browser monitor returned `bridge_not_running`; that response cannot establish archive-downloader liveness. Existing decoder, local ClickHouse and ingest processes were left alone.

The existing Phase 1 follow fold remains owned by its original process. The local continuation driver waits for nonempty outputs, reconciles record counts, hashes artifacts, and then runs similarity construction, Leiden clustering, honest evaluation and v1 pair export sequentially. It never ships a table to Netcup. The fold began against live archive globs, so its exact consumed shard set was not frozen; its provenance explicitly records this limitation.

At **07:59:48Z**, the fold had run for 1h11m23s and its observed temporary follow-state table contained **399,064,711 rows**, up from 354,562,820 at 07:31:35Z. This is an intermediate progress counter under concurrent merges, not a final edge count or percent complete. Final producer/edge/stat files were absent and the continuation driver was waiting. About **149.3 GiB** remained free. Community merges, seed-producer coverage, collision-trap failures and pack recall cannot yet be reported. [Dated progress observation](phase1_v1_progress_20261020_0800.json).

A follow-up attached to this conversation, **Finish Catbird Intel archive-only rebuild**, checks every 15 minutes and continues through verified graph results, pair-bank upload and reporting. It must reuse the existing driver, preserve failed evidence, and stop once delivery is complete or a new user decision is required. It does not authorize production changes or GPU allocation. [Resume instructions](../graph/PHASE1_V1_RUN.md).

## Validation completed so far

- 20 upload/provenance/isolation regression tests pass, with mocked network calls and no synthetic post/evaluation text.
- TypeScript typecheck passes after quarantining the synthetic eval module; production source behavior is unchanged.
- Export replay verifies the immutable files rather than resampling or replacing them.
- Six graph regression tests pass. Rust similarity construction passes formatting, warning-denying Clippy and its test command (which contains no Rust unit tests); a numeric graph fixture produced identical binary output with one and four threads. These structural checks do not qualify real communities.

## Real model smoke — measured, partially failed

The Political DEBATE teacher and the explicit DeBERTa-v3-large NLI fallback labeled the same 32 actual training posts. Ettin has 68,278,020 parameters and trained for 8 steps; training-probe BCE fell 2.015553→0.355486. Output shapes are 32×4 health logits and 32×256 embeddings. The social head is **untrained** because v1 pairs were not available; this test does not establish model quality.

FP32 ONNX is 273,555,140 bytes and passes numerical parity over all 32 posts plus padded sequence lengths (max probability error 0.000001550). Compact int8 is 69,433,199 bytes and **fails** the unchanged 0.03 maximum probability-error gate: observed 0.06676513. All failed quantization artifacts and receipts are preserved locally. Do not deploy the int8 candidate or describe it as qualified. [Complete smoke receipt](../colab/smoke_receipt_20261020.json).

The [Colab launch guide](../colab/README_REAL_MODELS.md) supplies runtime requirements, exact Drive paths, ordered notebook cells, a 32-row GPU pilot and checkpoint-resume commands. The source bundle is checksummed. Original local training preceded the final source inventory; receipts explicitly distinguish delivered source identity from that earlier execution.

Use the [revised V1_1 notebook on Drive](https://drive.google.com/file/d/1eAdDcRabEtBbgQKxXz9DwZLdUYKNjC_I/view?usp=drivesdk) and its [revision-2 implementation bundle](https://drive.google.com/drive/folders/1tq6CNMODz4R0EK9E_GFLRSqF7G0068iw). Both were uploaded and read back with matching checksums. Source-manifest SHA-256 is `08af9e174b72344cdb05b3249be0fc9440c6bbde9e806dd333830b80fe9dd592`; notebook SHA-256 is `67c82274ce540869e367bb1e004a8237e8ca42eaff4f240201bf1639d984a042`. The initial delivery remains preserved, locally under `colab/archive-bundles/initial/` and on Drive; it is not the current launch target.

Revision 2 adds sealed export/holdout checks to CLI execution, validates actual author buckets and URI/DID consistency, preserves human review files, supports verified cached merges, and makes checkpoints recoverable with integrity sidecars. **27 structural tests pass**, with no extra model inference or training. Local filesystem recovery was tested; hosted Drive durability remains unverified.

Josh explicitly approved a separate A100 pilot on the same 32 posts, bounded to **30 minutes or 25 existing compute units**, whichever comes first, including setup/download time and without purchases. **Attempt 1 failed before inference** after `11.259874` seconds: the runtime Python version differed from the prepared Python 3.12 lock. Its precise Python version was not captured before that assertion. Runtime shutdown was confirmed; account balance displayed `4,997.64` units before and after, which is not proof of zero charge below the display precision. [Retained attempt receipt](colab_gpu_pilot_20261020_attempt1.json). The retry used runtime-specific locks and a reduced ceiling of **29 minutes / 24 units** within the original approval.

**Attempt 2 passed real CUDA teacher execution on an A100-SXM4-40GB**, using Python 3.13.15, torch 2.9.0, vLLM 0.11.2 and transformers 4.57.3. Qwen and Political DEBATE each labeled exactly the same 32 real training posts used by the local smoke selection. The successful attempt took **769.610 seconds**, including **279.990 seconds** for installation and **420.800 seconds** for the inference subprocess. Qwen's stage receipt records 382.968 seconds and Political DEBATE's 18.261 seconds, including their initialization; these are not sustained-throughput measurements.

Independent verification matched all 11 downloaded file hashes, exact row identities, and executed source hashes against revision 2. All **1,280 Qwen digit log probabilities** are finite; reconstructing its 128 rubric scores and Political DEBATE probabilities gives a maximum error of `3.82e-8`. Both runtime shutdowns were confirmed. Immediate balance decreased by 0.61 units; the delayed readback at 07:59:14Z was **4,996.58**, a **1.06-unit account-wide decrease**, with **0 active assignments / 0.00 units/hour**. The sampled rate integral was **1.06794 units**. Preserve all readings: update latency, display precision and account scope prevent calling them an exact per-job invoice. [Execution receipt](colab_gpu_pilot_20261020_attempt2.json), [independent verification](colab_gpu_pilot_20261020_verified.json), [raw output archive](colab_gpu_pilot_20261020_outputs.zip), [pilot and launch addendum](COLAB_GPU_PILOT_20261020.md).

The pilot ran teachers only; it did not train a student on the GPU. Full 150k teacher labels, 600 holdout labels/human review, graph-pair social training, int8 parity, representative CPU serving measurements and bulk GPU cost remain open. The current Qwen implementation requests all-vocabulary log probabilities on vLLM 0.11.2; its memory/output overhead warrants a measured optimization before bulk labeling. No further GPU allocation is authorized by the completed pilot approval. The pipeline refuses to tune thresholds on teacher outputs or fabricate benchmark numbers.

## Source custody and continuation

The repair was sealed with jj commits `8ca979f5` (quarantine and v1 exports), `c2cd0b88` (real model implementation), and `21980889` (data/review/recovery hardening and pilot runner). Workspace correction commits include `3a77b344` and `606a28c2`. Each seal was followed immediately by `jj new`; unrelated decoder changes and the original stray pycache were excluded from named repair commits. No feature publication, production table swap, model deployment or ingest change occurred.

The remaining graph work runs under the existing fold/driver and the attached 15-minute follow-up. Its completion report must show raw failures and merged communities, then validate and upload the v1 pairs before full social training can begin. Bulk labeling and training require a separately bounded resource decision; human-reviewed holdout thresholds and the failed int8 export remain explicit qualification gates.
