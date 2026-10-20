"""Versioned configuration, pricing/budget arithmetic, and credential scrubbing."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "config"

_CREDENTIAL_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{12,}", re.IGNORECASE),
    re.compile(r"ANTHROPIC_API_KEY\s*=\s*\S+", re.IGNORECASE),
)
_SENSITIVE_KEY_NAMES = frozenset({
    "api_key",
    "anthropic_api_key",
    "authorization",
    "x-api-key",
    "secret",
    "token",
    "password",
    "credentials",
})


class CredentialLeakError(ValueError):
    """Raised when a secret or credential pattern appears in an artifact or log."""


class BudgetConstraintError(ValueError):
    """Raised when a reservation or spend violates budget ceilings or bucket limits."""


class ConfigurationMismatchError(ValueError):
    """Raised when resuming a run with changed input, rubric, schema, model, or pricing."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes without NaN or key-order drift."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    """Compute SHA-256 of canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def assert_no_credentials(value: Any, path: str = "root") -> None:
    """Recursively verify that no API keys, bearer tokens, or secret fields exist."""
    if isinstance(value, str):
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(value):
                raise CredentialLeakError(f"Credential pattern detected at {path}")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            key_str = str(k)
            if key_str.lower() in _SENSITIVE_KEY_NAMES and v not in (None, "", "REDACTED", False):
                raise CredentialLeakError(f"Sensitive key '{key_str}' must not be stored at {path}")
            assert_no_credentials(k, f"{path}.{key_str}(key)")
            assert_no_credentials(v, f"{path}.{key_str}")
        return
    if isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            assert_no_credentials(item, f"{path}[{idx}]")


def scrub_environment_snapshot(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a safe subset of environment variables, never including secrets."""
    source = env if env is not None else dict(os.environ)
    safe_keys = ("PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM")
    out: dict[str, str] = {}
    for k in safe_keys:
        if k in source:
            val = source[k]
            assert_no_credentials(val, f"env.{k}")
            out[k] = val
    out["anthropic_api_key_present_in_env"] = str(bool(source.get("ANTHROPIC_API_KEY")))
    return out


@dataclass(frozen=True)
class HarnessConfig:
    rubric: dict[str, Any]
    pricing: dict[str, Any]
    output_schema: dict[str, Any]
    rubric_sha256: str
    pricing_sha256: str
    schema_sha256: str
    bundle_sha256: str

    @classmethod
    def load(
        cls,
        rubric_path: Path | None = None,
        pricing_path: Path | None = None,
        schema_path: Path | None = None,
    ) -> "HarnessConfig":
        r_path = Path(rubric_path) if rubric_path else CONFIG_DIR / "rubric_v1.json"
        p_path = Path(pricing_path) if pricing_path else CONFIG_DIR / "pricing_and_budget_v1.json"
        s_path = Path(schema_path) if schema_path else CONFIG_DIR / "output_schema_v1.json"

        rubric = json.loads(r_path.read_text(encoding="utf-8"))
        pricing = json.loads(p_path.read_text(encoding="utf-8"))
        output_schema = json.loads(s_path.read_text(encoding="utf-8"))

        assert_no_credentials(rubric, "rubric")
        assert_no_credentials(pricing, "pricing")
        assert_no_credentials(output_schema, "output_schema")

        cls._validate_contracts(rubric, pricing, output_schema)

        r_sha = canonical_sha256(rubric)
        p_sha = canonical_sha256(pricing)
        s_sha = canonical_sha256(output_schema)
        bundle_sha = canonical_sha256({
            "rubric_sha256": r_sha,
            "pricing_sha256": p_sha,
            "schema_sha256": s_sha,
        })
        return cls(
            rubric=rubric,
            pricing=pricing,
            output_schema=output_schema,
            rubric_sha256=r_sha,
            pricing_sha256=p_sha,
            schema_sha256=s_sha,
            bundle_sha256=bundle_sha,
        )

    @staticmethod
    def _validate_contracts(
        rubric: dict[str, Any],
        pricing: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        expected_order = ["politicalness", "outraged", "critical", "engagement_bait", "spam"]
        if rubric.get("student_contract") != "catbird-ettin-components-v3":
            raise ValueError("Rubric student_contract must be catbird-ettin-components-v3")
        if rubric.get("component_order") != expected_order:
            raise ValueError(f"Rubric component_order must be {expected_order}")
        if list(rubric.get("components", {}).keys()) != expected_order:
            raise ValueError("Rubric components dictionary must match exact component_order")

        schema_props = output_schema.get("json_schema", {}).get("properties", {})
        for comp in expected_order:
            if comp not in schema_props:
                raise ValueError(f"Output schema missing component {comp}")
        for forbidden in ("heat", "teacher_heat", "hostility", "teacher_hostility"):
            if forbidden in schema_props or forbidden in rubric.get("components", {}):
                raise ValueError(f"Forbidden legacy/future field {forbidden} present in contract")

        ceilings = pricing.get("ceilings_microdollars", {})
        stages = ceilings.get("stages", {})
        stage_sum = sum(int(stages[k]) for k in ("pilot", "haiku_bulk", "sonnet_review", "opus_review"))
        if stage_sum + int(ceilings.get("contingency_reserve", 0)) != int(ceilings.get("total_cash_ceiling", 0)):
            raise ValueError("Stage ceilings plus contingency reserve must equal total_cash_ceiling ($200)")
        if int(ceilings.get("total_cash_ceiling", 0)) != 200_000_000:
            raise ValueError("Total cash ceiling must be 200,000,000 microdollars ($200.00)")
        if pricing.get("cache_savings_assumed") is not False:
            raise ValueError("Budget configuration must never assume prompt caching savings")
        if pricing.get("batch_api_only") is not True:
            raise ValueError("Budget configuration must enforce Batch API pricing only")

    def model_spec(self, tier_or_model_id: str) -> dict[str, Any]:
        """Resolve a model spec by tier ('haiku', 'sonnet', 'opus') or exact model_id."""
        models = self.pricing["models"]
        if tier_or_model_id in models:
            return copy.deepcopy(models[tier_or_model_id])
        for tier, spec in models.items():
            if spec["model_id"] == tier_or_model_id:
                out = copy.deepcopy(spec)
                out["tier"] = tier
                return out
        raise ValueError(f"Unknown or unconfigured model '{tier_or_model_id}'; silent substitution is prohibited")

    def compute_safety_margin_tokens(self, counted_input_tokens: int) -> int:
        if counted_input_tokens <= 0:
            raise ValueError("counted_input_tokens must be positive")
        margin_cfg = self.pricing["input_safety_margin"]
        min_tokens = int(margin_cfg["min_tokens"])
        prop = Decimal(str(margin_cfg["proportional_fraction"]))
        prop_tokens = int((Decimal(counted_input_tokens) * prop).to_integral_value(rounding=ROUND_CEILING))
        return max(min_tokens, prop_tokens)

    def compute_reserved_input_tokens(self, counted_input_tokens: int) -> int:
        return counted_input_tokens + self.compute_safety_margin_tokens(counted_input_tokens)

    def calculate_cost_microdollars(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        *,
        include_tax_multiplier: bool = True,
    ) -> int:
        """Exact ceiling integer microdollar calculation under Batch API pricing."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        spec = self.model_spec(model_id)
        threshold = int(spec.get("long_prompt_threshold_tokens", 100_000))
        if input_tokens > threshold:
            raise BudgetConstraintError(
                f"Input tokens ({input_tokens}) exceed long-prompt threshold ({threshold}) for {spec['model_id']}"
            )
        in_rate = Decimal(int(spec["batch_input_microdollars_per_mtok"]))
        out_rate = Decimal(int(spec["batch_output_microdollars_per_mtok"]))
        multiplier = Decimal(str(self.pricing.get("tax_fee_multiplier", "1.00"))) if include_tax_multiplier else Decimal("1")

        raw_microdollars = (
            (Decimal(input_tokens) * in_rate + Decimal(output_tokens) * out_rate)
            * multiplier
            / Decimal(1_000_000)
        )
        return int(raw_microdollars.to_integral_value(rounding=ROUND_CEILING))

    def calculate_reservation_Bounds(
        self,
        model_id: str,
        counted_input_tokens: int,
        *,
        override_input_bucket_ceiling: int | None = None,
    ) -> dict[str, int]:
        """Compute reserved input tokens, reserved output tokens, and max liability in microdollars."""
        spec = self.model_spec(model_id)
        safety_margin = self.compute_safety_margin_tokens(counted_input_tokens)
        reserved_input = counted_input_tokens + safety_margin
        bucket_ceiling = (
            int(override_input_bucket_ceiling)
            if override_input_bucket_ceiling is not None
            else int(spec["input_bucket_ceiling_tokens"])
        )
        if reserved_input > bucket_ceiling:
            raise BudgetConstraintError(
                f"Reserved input tokens ({reserved_input} = {counted_input_tokens} + {safety_margin} margin) "
                f"exceed bucket ceiling ({bucket_ceiling}) for {spec['model_id']}"
            )
        reserved_output = int(spec["max_output_tokens"])
        reserved_microdollars = self.calculate_cost_microdollars(
            spec["model_id"], reserved_input, reserved_output, include_tax_multiplier=True
        )
        return {
            "counted_input_tokens": counted_input_tokens,
            "safety_margin_tokens": safety_margin,
            "reserved_input_tokens": reserved_input,
            "input_bucket_ceiling_tokens": bucket_ceiling,
            "reserved_output_tokens": reserved_output,
            "reserved_microdollars": reserved_microdollars,
        }


def format_usd(microdollars: int) -> str:
    """Format integer microdollars as exact USD string (e.g. '$7.120000' or '$8.00')."""
    sign = "-" if microdollars < 0 else ""
    val = Decimal(abs(microdollars)) / Decimal(1_000_000)
    return f"{sign}${val:.6f}"
