# Real Colab pilot and launch addendum — 2026-10-20

Catbird Intel's replacement model pipeline now has a measured GPU teacher pilot in addition to its local Ettin smoke. Qwen 32B AWQ and Political DEBATE each labeled the same 32 real posts on an A100, and the returned files passed independent integrity and score checks. This establishes that the pinned teacher implementation executes on Colab; it does not establish recommendation quality, bulk throughput, or student training on the GPU. The runtime is stopped, and the initial 30-minute/25-unit approval is exhausted as authorization even though the actual run used much less.

Governing documents: [master plan](../../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md), [decision log](../../docs/DECISIONS.md), and [repair report](repair_run_20261020.md). This dated addendum updates the GPU status in the immutable revision-2 launch guide without changing the checksummed source bundle.

## Measured result

| Observation | Actual result |
|---|---:|
| Unique real training posts | 32 |
| Qwen rubric scores / retained digit log probabilities | 128 / 1,280 |
| Successful attempt, including setup and cleanup | 769.610 seconds |
| Dependency installation | 279.990 seconds |
| Inference subprocess, including both teachers | 420.800 seconds |
| Qwen stage, including model initialization | 382.968 seconds |
| Political DEBATE stage, including initialization | 18.261 seconds |
| Delayed account balance change | 4,997.64 → 4,996.58 units |
| Integrated observed account usage rate | 1.06794 units |
| Final active assignments / hourly rate | 0 / 0.00 |

The first attempt failed before inference on a Python-version assertion after 11.260 seconds; its runtime also stopped. The successful retry selected a prepared Python 3.13 dependency lock after observing the actual runtime, rather than assuming the local Python version. Hardware/software: NVIDIA A100-SXM4-40GB, Python 3.13.15, torch 2.9.0, vLLM 0.11.2, transformers 4.57.3, pyarrow 26.0.0. Source hashes match the uploaded revision-2 bundle.

The immediate balance drop was only 0.61 units; the later 1.06-unit observation is retained separately. These are account-wide measurements with reporting latency and display precision, not exact job billing. Do not multiply this small run into a 150k-post cost or throughput claim.

See the [independent verification](colab_gpu_pilot_20261020_verified.json), [execution receipt](colab_gpu_pilot_20261020_attempt2.json), [preparation manifest](colab_gpu_pilot_20261020_preparation.json), and [raw returned logs/labels](colab_gpu_pilot_20261020_outputs.zip). The complete local preparation remains at `~/catbird-lake-local/colab-pilot-prepared-revision2-retry/`; its private session-state file is deliberately not uploaded.

## Launch paths and order

1. Open the [V1_1 notebook on Drive](https://drive.google.com/file/d/1eAdDcRabEtBbgQKxXz9DwZLdUYKNjC_I/view?usp=drivesdk) with Google Colaboratory. Select a separate Python 3 **A100 GPU with at least 40 GB**. The existing archive-downloader session must remain separate.
2. Run cells **1–3** to mount Drive, install dependencies and verify inputs. The lake path is `/content/drive/MyDrive/CatbirdCommunity/catbird-lake/`; sources are in `colab-v1/revision-2/`. Keep `BEHAVIOR_TEACHER = "qwen"` for the tested primary teacher.
3. Cells **4–5** perform bulk labels and create the holdout review sheet. These spend beyond the completed pilot; no agent should launch them under that pilot approval. Review the resource budget first. Their inputs are `training-banks/phase3_health_training_150k_v1_unlabeled.parquet` and `eval/phase3_holdout_600_unlabeled.jsonl`, with the sealed manifests alongside them.
4. Run cell **6** only after the verified `training-banks/phase5_social_encoder_pairs_v1.parquet` and its adjacent manifest arrive. Cell **7** trains and exports; cell **8** prints measured receipts. The v1 pairs are still pending the local graph fold. Outputs stay in `runs/real-v1.1-qwen/`, and holdout thresholds stay unlocked until Josh reviews the labels.

The [full launch guide](../colab/README_REAL_MODELS.md) documents resume and checkpoint recovery. Its pasted GPU smoke includes student training, while this completed bounded CLI pilot ran **teachers only**. The existing local eight-step student smoke uses NLI fallback targets and has an untrained social head. FP32 ONNX passed numerical agreement; compact int8 failed the unchanged 0.03 probability-error gate.

## Before bulk Qwen labels

The measured implementation falls back to all-vocabulary log probabilities because vLLM 0.11.2 has no `logprob_token_ids` parameter. A 32-row shard is saved after all four rubrics finish, so silence during the observed Qwen stage did not indicate a deadlock. Exact-version source shows that the all-vocabulary route performs vocabulary-sized selection and output conversion; we did not profile how much time it consumed. [Sampling parameters](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/sampling_params.py), [input batch](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/worker/gpu_input_batch.py), [output processing](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/engine/logprobs.py).

A candidate optimization is processed log probabilities restricted to the ten allowed digits, with neutral temperature/filter/penalty settings. It has not been implemented or measured. Any future source revision must preserve and compare all 1,280 conditional digit probabilities and 128 scores on these same 32 posts, record the changed raw-log-probability semantics, and measure time/usage before promotion. A separate bounded GPU approval is required for that experiment or bulk labels.
