"""Revision-three model identity, license provenance and component output contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path

MODELS = {
    "political": ("mlburnham/Political_DEBATE_large_v1.0", "1a3aff1ecb97ad93a6de598f7414b1707de7e7c3"),
    "qwen": ("Qwen/Qwen2.5-32B-Instruct-AWQ", "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"),
    "nli": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "b2730f16019076bb0009481121efbe4705e0e378"),
    "student": ("jhu-clsp/ettin-encoder-68m", "ac19ae4bc51093b31c475665ac872a936d056cc2"),
}
HEALTH = ("politicalness", "outraged", "critical", "engagement_bait", "spam")
STUDENT_CONTRACT = {
    "schema": "catbird-ettin-components-v3",
    "health_output": "health_component_logits",
    "health_output_order": list(HEALTH),
    "embedding_output": "matryoshka_emb",
    "embedding_dimensions": [256, 128, 64],
    "heat": "not a trained output; requires separately frozen serving aggregation and calibration",
    "hostility": "not trained; no independent hostility targets exist",
    "legacy_checkpoint_policy": "four-output checkpoints are incompatible; no implicit weight or target migration",
}


def model_identity(kind: str) -> dict:
    catalog = json.loads((Path(__file__).resolve().parents[1] / "model_licenses_v3.json").read_text())
    identity = copy.deepcopy(catalog["models"][kind])
    if (identity["model_id"], identity["revision"]) != MODELS[kind]:
        raise ValueError("License inventory and pinned model differ")
    identity["implementation_code_license"] = catalog["implementation_code_license"]
    identity["runtime_code_licenses"] = copy.deepcopy(catalog["runtime_code_licenses"])
    identity["catbird_training_input_license"] = copy.deepcopy(catalog["catbird_training_input_license"])
    return identity
