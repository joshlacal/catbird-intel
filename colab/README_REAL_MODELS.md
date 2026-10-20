# Catbird Intel real-model repair

This pipeline labels real Bluesky posts with pretrained teachers and trains a real Ettin encoder with two output heads. It replaces the quarantined regex-label and hashed-vector artifacts; it does not qualify the existing production scorer. The governing [master plan](../../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md) and [decision log](../../docs/DECISIONS.md) retain the invalidated history. Production ingest stays on its explicitly named regex baseline during the soak ending 2026-10-22 13:00Z.

## Launch in Colab

1. Open `CatbirdCommunity/catbird-lake/Catbird_Intel_Real_Teachers_Ettin_V1_1.ipynb` from Google Drive using **Open with → Google Colaboratory**. Choose a **separate runtime**, not the archive downloader session. Choose **Python 3, A100 GPU** (40 GB or more), or H100 if available. The Qwen 32B AWQ teacher is not supported by this notebook on a T4. An L4 may have insufficient headroom; choose A100 or explicitly use the NLI fallback.
2. Run cells **1–3** to mount Drive, install requirements and verify the real banks. The notebook expects its sources under `/content/drive/MyDrive/CatbirdCommunity/catbird-lake/colab-v1/revision-2/`. Installation/runtime versions and an exact SHA-256 inventory of implementation sources are recorded. Earlier smoke execution receipts without a source inventory are not retroactively attributed to the current files. If Colab asks for a runtime restart after installing vLLM, restart and rerun cells 1 and 3.
3. Set `BEHAVIOR_TEACHER = "qwen"` in cell 3. The explicit fallback is `"nli"`, which uses `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`; its outputs remain identified as NLI rather than Qwen. No automatic fallback can disguise a failed Qwen run. Read the GPU name/memory and input hashes printed by cell 3.
4. Run cells **4–5** to label the 150k training bank and 600-post holdout, then create Josh's review sheet. Every teacher is pinned to an immutable Hugging Face revision. Each stage runs in a separate process so its GPU memory is released before the next stage. Label shards on Drive support resume; rerun the same cells with the same input/model/rubric. A failed or partial shard produces a hard error; archive that shard and rerun it.
5. Run cell **6** only after the archive-only v1 pair bank exists. The validator checks its KnownFor digest, file digest, `graph_version=phase1-v1`, supplemental-edge exclusion and holdout digest. It refuses old v0 pairs. Pairs currently represent **weak same-community supervision**, not verified co-engagement.
6. Run cell **7** to train 5,000 optimizer steps and export the actual two-head ONNX model. Change steps/batch size only before starting a new run. Checkpoints save every 250 steps using a temporary file, filesystem sync and rename, followed by a SHA-256/size/configuration integrity sidecar. Set `RESUME` to the latest checkpoint path after a disconnect and keep configuration unchanged. Recovery chooses the latest intact checkpoint, archives damaged/partial files and log records beyond that step, and records the recovery. Legacy v1 checkpoints without integrity sidecars are not silently trusted. This is an initial training schedule, not a quality or convergence guarantee.
7. Read the actual `training_receipt.json` and `onnx_receipt.json` printed by cell **8**. These contain measured loss, shapes, parameter count, elapsed time, model sizes, hashes and torch/ONNX numerical parity. No F1, posts/credit or credit burn is invented. Record the Colab account balance before/after separately; disconnect the GPU after the run.

All outputs go to `catbird-lake/runs/real-v1.1-qwen/` or `real-v1.1-nli/`, never to live production. Uploading source and data does not launch or pay for a GPU session.

## Exact data paths

- `training-banks/phase3_health_training_150k_v1_unlabeled.parquet` and `.manifest.json` with the `.parquet` suffix replaced
- `eval/phase3_holdout_600_unlabeled.jsonl` and `phase3_v1_export_manifest.json`
- `training-banks/phase5_social_encoder_pairs_v1.parquet` and adjacent `.manifest.json` with the `.parquet` suffix replaced
- `colab-v1/revision-2/real_pipeline.py`, `colab-v1/revision-2/real_pipeline/`, `colab-v1/revision-2/requirements-colab.txt`

The audited export manifest is pinned in source. Every CLI stage verifies its SHA-256 and the canonical holdout SHA-256, exact integer split buckets and original URI/author correspondence. Raw inputs must match the sealed bank or holdout; labeled inputs must retain their sealed source hash and teacher selection/model/rubric identity. A different export requires an audited source revision. Pair validation requires all three author DIDs, their non-holdout buckets, row counts and the source bucket audit.

The holdout gets teacher labels and empty human-review fields. Review-sheet creation refuses an existing output or status sidecar, preserving human edits on rerun. Josh reviews those fields before thresholds can be locked. The code does not compute quality against its own teacher predictions or tune on this final holdout. A separate future engagement retrieval benchmark and an independent reviewed quality set are still needed for Phase 5/Phase 3 qualification.

## Model semantics and output contract

`mlburnham/Political_DEBATE_large_v1.0` is a **binary natural-language inference model**, not a dedicated bare-text politicalness classifier. The pipeline feeds each post plus the explicit hypothesis “This text is about politics, government, elections, or public policy.” It records both actual logits and the entailment probability. This is a zero-shot teacher estimate requiring human review.

Qwen receives each post as quoted data and one rubric at a time for outraged, critical, engagement bait and spam. Each rubric asks for one digit 0–9. The code verifies each digit is one tokenizer token, obtains all ten single-token log probabilities from vLLM, normalizes the ten scores and takes their expected value divided by nine. The labels retain the raw digit log probabilities. `heat = max(outraged, critical)`; critical here means attacking/condemning/disparaging, not every neutral disagreement. All rubrics and revisions are recorded.

The student uses the actual `AutoModel` Ettin-68M encoder, masked mean pooling, a four-output `health_logits` head and a normalized 256-dimensional `matryoshka_emb` head. Soft health targets use binary cross-entropy with logits. The social objective averages InfoNCE at 256, 128 and 64 dimensions and masks same-community false negatives. Re-normalize 128/64-dimensional prefixes after slicing at inference. The final health output order is politicalness, heat, engagement_bait, spam.

ONNX export runs `quantize_dynamic` on MatMul and Gather, including embedding weights. The expected 65–70 MB is only a rough size expectation; receipts always report actual bytes. FP32 and int8 outputs are compared with torch on real texts at multiple batch/sequence sizes. Passing numerical parity establishes exporter agreement, not model quality. Smoke-only exports have an untrained social head when no real v1 triplets occur among the same 32 posts.

## Local 32-row smoke

Use a separate Python 3.12 environment and `requirements-local.txt`. The CLI's smoke mode selects the same first 32 training rows and rejects more than 12 optimizer steps. It never labels the holdout locally. Both actual NLI teachers can run with `--device cpu --threads 2 --batch-size 2`; Qwen runs only in Colab. Artifacts belong outside the repository, e.g. `~/catbird-lake-local/real-model-smoke-v1/`.

Run `python real_pipeline.py --help` and each subcommand's `--help` for exact flags. Full labeling/training requires CUDA and is rejected on macOS. A smoke loss decrease demonstrates only fitting on those 32 rows. It cannot replace human validation, real social-pair training, representative throughput or the full teacher pipeline.

## Primary documentation

- [Political DEBATE model card](https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0)
- [Ettin model card and AutoModel usage](https://huggingface.co/jhu-clsp/ettin-encoder-68m)
- [NLI fallback model](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## Bounded GPU pilot before a full run

After a separately approved GPU allocation, run notebook cells 1–3, then paste this in a new cell before starting the 150k labels. It uses the same first 32 training posts, Qwen on CUDA, eight student steps and a separate output directory. It does not access a new paid runtime automatically or label the holdout. Record the actual account balance/usage before and after through the Colab CLI; the pipeline leaves credit fields null.

```python
PILOT = LAKE / 'runs/gpu-pilot-real-v1.1-qwen'
common = ['--input', BANK, '--holdout', HOLDOUT, '--export-manifest', EXPORT_MANIFEST, '--mode', 'smoke', '--device', 'cuda']
run('label', *common, '--teacher', 'political', '--max-length', 512, '--batch-size', 4, '--output', PILOT / 'political')
run('label', *common, '--teacher', 'qwen', '--max-length', 1024, '--batch-size', 4, '--output', PILOT / 'qwen')
run('merge', *common, '--political', PILOT / 'political', '--behavior', PILOT / 'qwen', '--behavior-teacher', 'qwen', '--output', PILOT / 'train-labeled.parquet')
run('train', '--input', PILOT / 'train-labeled.parquet', '--holdout', HOLDOUT, '--export-manifest', EXPORT_MANIFEST, '--mode', 'smoke', '--device', 'cuda', '--steps', 8, '--save-every', 8, '--batch-size', 8, '--max-length', 128, '--learning-rate', 0.0001, '--output', PILOT / 'student')
```

A GPU pilot without real v1 pairs fits only the health head; the social head is explicitly untrained. Stop on a failed numerical parity check and read its measured receipt. Do not relax a gate to make the report green.

## Measured local result, 2026-10-20

The CPU smoke used exactly 32 real training posts, two CPU threads and eight optimizer steps. Political DEBATE and the explicitly named NLI fallback produced actual targets. Ettin has **68,278,020 parameters**; the fixed training-batch BCE decreased from **2.015553 to 0.355486**. The observed output shapes were **32×4** and **32×256**. There were no real v1 social triplets among this test input, so this test did not train or validate the social head.

The final parity check covered **all 32 unique posts**, plus repeated real examples padded to sequence lengths 64 and 128. The **273,555,140-byte FP32 ONNX passed**: maximum health-probability error **0.000001550**, minimum embedding cosine **0.99999988**. The **69,433,199-byte int8 candidate failed** the unchanged 0.03 health-probability-error gate, measuring **0.06676513** (embedding cosine **0.99531621**). Keeping embedding weights FP32 produced a **146,797,996-byte** candidate that also failed, at **0.07467008** probability error. These are export-agreement measurements, not quality scores.

The compact int8 candidate remains unqualified and the training command exits with an error on that measured failure. All earlier failed exports/receipts remain in `export-previous-*/` folders. The smaller preliminary checks used seven row instances covering six unique real posts; they were superseded by the complete 32-post checks. No thresholds were relaxed and no additional examples or training steps were introduced during export repairs.

Local environment: `~/catbird-lake-local/real-model-smoke-venv/bin/python`. Outputs: `~/catbird-lake-local/real-model-smoke-v1/`. The repository's `smoke_receipt_20261020.json` summarizes exact artifact paths and measurements. Original smoke execution predated the final source inventory; the inventory identifies the delivered implementation, not a retroactively reconstructed training-source snapshot.

Bulk Qwen labeling, full dual-head training, GPU throughput/credit use, human quality review and representative serving CPU measurements remain unperformed. Labels can be generated before v1 pairs arrive, but the full training command refuses a missing or invalid pair bank.

## Revision 2 structural verification

This source revision hardens dataset validation, review preservation and disconnect recovery. It does not replace or retroactively reattribute the original measured smoke results. The immutable initial delivery is retained under `colab/archive-bundles/initial/`; the Drive source lives in `colab-v1/revision-2/`. Run `python -m unittest discover -s colab/tests -v` from the repository with the existing sealed files available. These checks use existing real-row metadata and tiny checkpoint objects; they run no model inference or optimizer steps. The approved CLI pilot wrapper is `ops/run_colab_teacher_pilot.py`. Its prepare mode packages the full canonical bank/holdout and their seals plus a separate 32-row selection audit; inference still uses exactly the first 32 rows through the ordinary smoke CLI.
