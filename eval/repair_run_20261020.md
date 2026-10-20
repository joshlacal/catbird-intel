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

At **07:19:10Z**, the fold had run for 30m45s and its temporary follow-state table contained **315,366,621 observed rows**. This is an intermediate progress counter, not a final edge count. Final producer/edge/stat files were absent and the continuation driver was waiting. About **154.7 GiB** remained free. Community merges, seed-producer coverage, collision-trap failures and pack recall cannot yet be reported.

## Validation completed so far

- 20 upload/provenance/isolation regression tests pass, with mocked network calls and no synthetic post/evaluation text.
- TypeScript typecheck passes after quarantining the synthetic eval module; production source behavior is unchanged.
- Export replay verifies the immutable files rather than resampling or replacing them.
- Six graph regression tests pass. Rust similarity construction passes formatting, warning-denying Clippy and its test command (which contains no Rust unit tests); a numeric graph fixture produced identical binary output with one and four threads. These structural checks do not qualify real communities.

## Real model smoke — measured, partially failed

The Political DEBATE teacher and the explicit DeBERTa-v3-large NLI fallback labeled the same 32 actual training posts. Ettin has 68,278,020 parameters and trained for 8 steps; training-probe BCE fell 2.015553→0.355486. Output shapes are 32×4 health logits and 32×256 embeddings. The social head is **untrained** because v1 pairs were not available; this test does not establish model quality.

FP32 ONNX is 273,555,140 bytes and passes numerical parity over all 32 posts plus padded sequence lengths (max probability error 0.000001550). Compact int8 is 69,433,199 bytes and **fails** the unchanged 0.03 maximum probability-error gate: observed 0.06676513. All failed quantization artifacts and receipts are preserved locally. Do not deploy the int8 candidate or describe it as qualified. [Complete smoke receipt](../colab/smoke_receipt_20261020.json).

The [Colab launch guide](../colab/README_REAL_MODELS.md) supplies runtime requirements, exact Drive paths, ordered notebook cells, a 32-row GPU pilot and checkpoint-resume commands. The source bundle is checksummed. Original local training preceded the final source inventory; receipts explicitly distinguish delivered source identity from that earlier execution.

The initial [real-model notebook on Drive](https://drive.google.com/file/d/1giCfXAv8n93d4kggO4fuuetgMIHOpcLj/view?usp=drivesdk) and its [implementation bundle](https://drive.google.com/drive/folders/1kiYqtro-NKsl0hcAccLOV3tfV_JgwsgV) were uploaded and read back with matching checksums. Upload is not GPU execution. A separate A100 pilot on the same 32 posts, bounded to 30 minutes or 25 existing compute units, awaits the requested spending authorization; no GPU has been allocated for this repair.

Full Qwen CUDA inference, full 150k teacher labels, 600 holdout labels/human review, graph-pair social training, int8 parity, representative CPU serving measurements and measured GPU credit use remain open. The pipeline refuses to tune thresholds on teacher outputs or fabricate benchmark numbers.
