"""Official Anthropic SDK transport, offline FakeAnthropicTransport, and Clef/SystemOne adapter."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator, Protocol

try:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as AnthropicBatchRequest
except ModuleNotFoundError:  # pragma: no cover - fallback when anthropic SDK is not installed in a local ML venv
    anthropic = None  # type: ignore[assignment]

    def MessageCreateParamsNonStreaming(**kwargs: Any) -> dict[str, Any]:  # type: ignore[misc]
        return dict(kwargs)

    def AnthropicBatchRequest(*, custom_id: str, params: Any) -> dict[str, Any]:  # type: ignore[misc]
        return {"custom_id": custom_id, "params": params}

from .config import HarnessConfig, assert_no_credentials, canonical_sha256
from .schemas import COMPONENT_ORDER, resolve_post_observation


class AuthorizationGateError(PermissionError):
    """Raised when an operation attempts network/paid calls without explicit authorization."""


class ProviderRateLimitError(RuntimeError):
    """Raised when rate limited with Retry-After metadata."""

    def __init__(self, message: str, retry_after_seconds: int = 1):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class BatchStatusSnapshot:
    provider_batch_id: str
    processing_status: str  # "in_progress" | "canceling" | "ended"
    request_counts: dict[str, int]
    created_at: str
    ended_at: str | None
    custom_ids_sample: tuple[str, ...] = ()


class AnthropicTransport(Protocol):
    mode: str

    def verify_model_available(self, model_id: str) -> dict[str, Any]: ...

    def count_tokens(self, params: dict[str, Any]) -> int: ...

    def create_batch(
        self,
        requests: list[AnthropicBatchRequest],
        *,
        client_submission_id: str,
    ) -> BatchStatusSnapshot: ...

    def retrieve_batch(self, provider_batch_id: str) -> BatchStatusSnapshot: ...

    def list_batches(self, limit: int = 20) -> list[BatchStatusSnapshot]: ...

    def cancel_batch(self, provider_batch_id: str) -> BatchStatusSnapshot: ...

    def iter_batch_results(self, provider_batch_id: str) -> Iterator[dict[str, Any]]: ...


class OfficialAnthropicSDKTransport:
    """Production transport wrapping the official `anthropic` Python SDK.

    Strictly gated behind explicit authorization flags; never instantiated in offline mode.
    """

    mode = "live_anthropic"

    def __init__(
        self,
        *,
        offline: bool = True,
        authorize_paid_submission: bool = False,
        authorize_corpus_transmission: bool = False,
        max_retries: int = 2,
    ):
        if offline or not (authorize_paid_submission and authorize_corpus_transmission):
            raise AuthorizationGateError(
                "Live Anthropic SDK transport is blocked in offline mode. "
                "Requires --no-offline, --authorize-paid-submission, and --authorize-corpus-transmission."
            )
        self._authorize_paid = authorize_paid_submission
        self._authorize_transmission = authorize_corpus_transmission
        self._client = anthropic.Anthropic(max_retries=max_retries)

    @staticmethod
    def _handle_rate_limit(exc: anthropic.RateLimitError) -> None:
        retry_after = 5
        resp = getattr(exc, "response", None)
        if resp is not None and hasattr(resp, "headers"):
            hdr = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if hdr and str(hdr).isdigit():
                retry_after = max(1, int(hdr))
        raise ProviderRateLimitError(f"Anthropic rate limit exceeded; retry after {retry_after}s", retry_after) from exc

    def verify_model_available(self, model_id: str) -> dict[str, Any]:
        try:
            info = self._client.models.retrieve(model_id)
            returned_id = getattr(info, "id", model_id)
            if returned_id != model_id:
                raise ValueError(f"Provider returned model {returned_id!r} for requested {model_id!r}")
            return {"model_id": returned_id, "verified": True}
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    def count_tokens(self, params: dict[str, Any]) -> int:
        if not self._authorize_transmission:
            raise AuthorizationGateError("Remote token counting transmits corpus text and requires authorization")
        try:
            resp = self._client.messages.count_tokens(
                model=params["model"],
                system=params.get("system", ""),
                messages=params["messages"],
                output_config=params.get("output_config"),
            )
            return int(resp.input_tokens)
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    def create_batch(
        self,
        requests: list[AnthropicBatchRequest],
        *,
        client_submission_id: str,
    ) -> BatchStatusSnapshot:
        if not self._authorize_paid:
            raise AuthorizationGateError("Paid Batch API submission is not authorized")
        try:
            batch = self._client.messages.batches.create(requests=requests)
            return self._to_snapshot(batch)
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    def retrieve_batch(self, provider_batch_id: str) -> BatchStatusSnapshot:
        try:
            batch = self._client.messages.batches.retrieve(provider_batch_id)
            return self._to_snapshot(batch)
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    def list_batches(self, limit: int = 20) -> list[BatchStatusSnapshot]:
        out: list[BatchStatusSnapshot] = []
        try:
            for batch in self._client.messages.batches.list(limit=limit):
                out.append(self._to_snapshot(batch))
                if len(out) >= limit:
                    break
            return out
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    def cancel_batch(self, provider_batch_id: str) -> BatchStatusSnapshot:
        try:
            batch = self._client.messages.batches.cancel(provider_batch_id)
            return self._to_snapshot(batch)
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    def iter_batch_results(self, provider_batch_id: str) -> Iterator[dict[str, Any]]:
        try:
            for item in self._client.messages.batches.results(provider_batch_id):
                if hasattr(item, "model_dump"):
                    yield item.model_dump(mode="json")
                else:
                    yield json.loads(item.json())
        except anthropic.RateLimitError as exc:
            self._handle_rate_limit(exc)
            raise

    @staticmethod
    def _to_snapshot(batch: Any) -> BatchStatusSnapshot:
        counts = batch.request_counts
        return BatchStatusSnapshot(
            provider_batch_id=str(batch.id),
            processing_status=str(batch.processing_status),
            request_counts={
                "processing": int(getattr(counts, "processing", 0)),
                "succeeded": int(getattr(counts, "succeeded", 0)),
                "errored": int(getattr(counts, "errored", 0)),
                "canceled": int(getattr(counts, "canceled", 0)),
                "expired": int(getattr(counts, "expired", 0)),
            },
            created_at=str(getattr(batch, "created_at", "")),
            ended_at=str(getattr(batch, "ended_at", "")) if getattr(batch, "ended_at", None) else None,
        )


class FakeAnthropicTransport:
    """Deterministic, disk-backed offline transport for unit tests and dry-run simulation.

    Every response generated by this transport carries `simulated: true` metadata so it
    can never be mistaken for actual Claude inference or accepted training ground truth.
    """

    mode = "fake_offline"

    def __init__(
        self,
        state_path: Path,
        *,
        allowed_models: tuple[str, ...] = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"),
        fault_rules: dict[str, dict[str, Any]] | None = None,
    ):
        self.state_path = Path(state_path)
        self.allowed_models = set(allowed_models)
        self.fault_rules = dict(fault_rules or {})
        self._load_or_init()

    def _load_or_init(self) -> None:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "batches": {},
                "rate_limit_next_create": None,
                "crash_after_accept_once": False,
                "timeout_before_accept_once": False,
                "out_of_order_results": True,
            }
            self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def configure_faults(self, **kwargs: Any) -> None:
        self._load_or_init()
        for k, v in kwargs.items():
            if k == "custom_id_faults" and isinstance(v, dict):
                self.fault_rules.update(v)
            else:
                self.state[k] = v
        self._save()

    def verify_model_available(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.allowed_models:
            raise ValueError(f"Model {model_id!r} is not available in configured fake provider")
        return {"model_id": model_id, "verified": True, "simulated": True}

    def count_tokens(self, params: dict[str, Any]) -> int:
        raw = json.dumps(params, sort_keys=True).encode("utf-8")
        return max(128, len(raw) // 4 + 64)

    @staticmethod
    def _extract_verbatim_span(post_text: str, keywords: tuple[str, ...]) -> str:
        clean = " ".join(post_text.split())
        if not clean:
            return ""
        lower = clean.lower()
        for kw in keywords:
            idx = lower.find(kw)
            if idx != -1:
                snippet = clean[idx : min(len(clean), idx + 48)].strip()
                return snippet[:120]
        return clean[: min(len(clean), 40)].strip()

    @classmethod
    def _simulate_rubric_output(
        cls,
        *,
        custom_id: str,
        model_id: str,
        user_envelope_json: str,
        fault_spec: dict[str, Any],
    ) -> dict[str, Any]:
        env = json.loads(user_envelope_json)
        post_text = str(env.get("untrusted_post_text") or "")
        lower = post_text.lower()
        signals = env.get("structural_signals") or {}
        is_reply_without_parent = bool(signals.get("is_reply")) and not bool(signals.get("reply_parent_text_included"))
        has_unseen_media = int(signals.get("image_count") or 0) > 0 and len(post_text.strip()) < 24

        # Check for prompt injection strings - always keep them inert!
        injection_detected = any(
            phrase in lower
            for phrase in ("ignore previous instructions", "system prompt", "score 0 for everything", "override schema")
        )

        # Politicalness simulation
        pol_kws = ("election", "senate", "congress", "president", "policy", "government", "ukraine", "putin", "vote", "parliament", "legislation", "tax", "democrat", "republican")
        has_pol = any(k in lower for k in pol_kws)
        if fault_spec.get("force_insufficient"):
            pol_score: int | str = "insufficient"
        elif has_pol:
            pol_score = 2 if ("senate" in lower or "election" in lower or "ukraine" in lower) else 1
            if model_id == "claude-sonnet-5-5" and "borderline_disagree" in fault_spec:
                pol_score = 3
        else:
            pol_score = 0

        # Outraged vs Critical separation (camera criticism -> critical=2, politicalness=0, outraged=0)
        crit_kws = ("heinous", "terrible", "awful", "worst", "broken", "hate", "garbage", "stupid", "bad", "critic", "camera", "autofocus", "fail")
        outrage_kws = ("furious", "outrage", "unacceptable", "disgusting", "unforgivable", "wtf", "heinous")
        has_crit = any(k in lower for k in crit_kws)
        has_outrage = any(k in lower for k in outrage_kws)

        crit_score: int | str = 2 if has_crit else 0
        outrage_score: int | str = 2 if has_outrage else (1 if ("lol" not in lower and has_crit and "!" in post_text) else 0)
        # Camera / product criticism boundary rule: critical=2, outraged=0, politicalness=0
        if "camera" in lower or "autofocus" in lower:
            pol_score = 0
            outrage_score = 0
            crit_score = 2

        bait_kws = ("repost if", "rt if", "wrong answers only", "bet nobody will", "like and share")
        bait_score: int | str = 2 if any(k in lower for k in bait_kws) else 0

        spam_kws = ("buy now", "free crypto", "airdrop", "click here", "dm for promo", "100x guaranteed")
        if any(k in lower for k in spam_kws):
            spam_score: int | str = 3
        elif is_reply_without_parent and len(post_text.strip()) < 18:
            # Short reply without parent context -> insufficient evidence
            pol_score = "insufficient"
            spam_score = 0
        else:
            spam_score = 0

        if fault_spec.get("force_insufficient") or has_unseen_media:
            pol_score = "insufficient"

        # Deterministic model-tier variation on hash-selected difficult items to exercise disagreement routing
        digest_val = int(hashlib.sha256(f"{post_text}:{model_id}".encode("utf-8")).hexdigest()[:4], 16)
        if not has_pol and not has_crit and digest_val % 11 == 0 and pol_score != "insufficient":
            if model_id == "claude-haiku-5-5":
                crit_score = 1
            elif model_id == "claude-sonnet-5-5":
                crit_score = 2

        def _comp(score: int | str, kws: tuple[str, ...], pos_reason: str) -> dict[str, Any]:
            if score == "insufficient":
                return {
                    "score": "insufficient",
                    "insufficient_evidence": True,
                    "reason_code": "MISSING_MEDIA_OR_LINK_CONTEXT" if has_unseen_media else "MISSING_PARENT_OR_QUOTE_CONTEXT",
                    "evidence_span": "",
                }
            if score == 0:
                return {
                    "score": 0,
                    "insufficient_evidence": False,
                    "reason_code": "PROMPT_INJECTION_IGNORED" if injection_detected else "ABSENT_IN_TEXT",
                    "evidence_span": "",
                }
            return {
                "score": int(score),
                "insufficient_evidence": False,
                "reason_code": "PROMPT_INJECTION_IGNORED" if injection_detected else pos_reason,
                "evidence_span": cls._extract_verbatim_span(post_text, kws),
            }

        any_insuff = any(s == "insufficient" for s in (pol_score, outrage_score, crit_score, bait_score, spam_score))
        missing_kind = (
            "media_visual"
            if (any_insuff and has_unseen_media)
            else ("reply_parent" if (any_insuff or is_reply_without_parent) else "none")
        )
        topic = (
            "international_conflict"
            if ("ukraine" in lower or "putin" in lower)
            else ("politics_policy" if has_pol else ("science_tech" if "camera" in lower else "none_or_everyday"))
        )

        return {
            "politicalness": _comp(pol_score, pol_kws, "DIRECT_POLICY_OR_CIVIC_REF"),
            "outraged": _comp(outrage_score, outrage_kws, "EXPRESSED_INDIGNATION"),
            "critical": _comp(
                crit_score,
                crit_kws,
                "NONPOLITICAL_CRITICISM" if pol_score == 0 else "DIRECT_NEGATIVE_EVALUATION",
            ),
            "engagement_bait": _comp(bait_score, bait_kws, "INTERACTION_HARVESTING_CUE"),
            "spam": _comp(spam_score, spam_kws, "COMMERCIAL_OR_SCAM_SOLICITATION"),
            "topic": topic,
            "context_dependent": bool(any_insuff or is_reply_without_parent),
            "missing_context_kind": missing_kind,
            "stance_ambiguity": "quoted_or_reported" if signals.get("is_quote") else "none",
        }

    def create_batch(
        self,
        requests: list[AnthropicBatchRequest],
        *,
        client_submission_id: str,
    ) -> BatchStatusSnapshot:
        self._load_or_init()

        if self.state.get("rate_limit_next_create"):
            retry_s = int(self.state["rate_limit_next_create"])
            self.state["rate_limit_next_create"] = None
            self._save()
            raise ProviderRateLimitError(f"Simulated 429 Rate Limit; Retry-After: {retry_s}", retry_s)

        if self.state.get("timeout_before_accept_once"):
            self.state["timeout_before_accept_once"] = False
            self._save()
            raise TimeoutError("Simulated network timeout before provider accepted batch")

        req_dicts: list[dict[str, Any]] = []
        for r in requests:
            if isinstance(r, dict):
                req_dicts.append(r)
            else:
                req_dicts.append(dict(r))

        for r in req_dicts:
            model_id = r["params"]["model"]
            self.verify_model_available(model_id)

        batch_hash = canonical_sha256([r["custom_id"] for r in req_dicts])[:16]
        provider_batch_id = f"msgbatch_sim_{batch_hash}_{len(self.state['batches']) + 1}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        results_list: list[dict[str, Any]] = []
        counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}

        for r in req_dicts:
            cid = str(r["custom_id"])
            params = r["params"]
            model_id = str(params["model"])
            user_msg = str(params["messages"][0]["content"])
            f_spec = dict(self.fault_rules.get(cid) or {})
            mode = f_spec.get("outcome", "succeeded")

            if mode == "omit_from_results":
                continue
            if mode == "errored_transient":
                counts["errored"] += 1
                results_list.append({
                    "custom_id": cid,
                    "simulated": True,
                    "result": {
                        "type": "errored",
                        "error": {"type": "overloaded_error", "message": "Simulated transient server overload"},
                    },
                })
                continue
            if mode == "errored_invalid_request":
                counts["errored"] += 1
                results_list.append({
                    "custom_id": cid,
                    "simulated": True,
                    "result": {
                        "type": "errored",
                        "error": {"type": "invalid_request_error", "message": "Simulated invalid request parameter"},
                    },
                })
                continue
            if mode == "expired":
                counts["expired"] += 1
                results_list.append({
                    "custom_id": cid,
                    "simulated": True,
                    "result": {"type": "expired"},
                })
                continue
            if mode == "canceled":
                counts["canceled"] += 1
                results_list.append({
                    "custom_id": cid,
                    "simulated": True,
                    "result": {"type": "canceled"},
                })
                continue

            # Otherwise result.type == "succeeded" (which includes stop_reason end_turn, refusal, max_tokens, or invalid schema)
            counts["succeeded"] += 1
            in_toks = int(f_spec.get("input_tokens") or max(450, len(user_msg) // 3 + 320))
            out_toks = int(f_spec.get("output_tokens") or (380 if "opus" in model_id else 220))
            returned_model = str(f_spec.get("returned_model") or model_id)

            if mode == "refusal":
                stop_reason = "refusal"
                stop_details = {"type": "refusal", "category": "safety_policy_refusal", "explanation": "Simulated refusal"}
                text_content = "I cannot fulfill this request."
            elif mode == "truncated":
                stop_reason = "max_tokens"
                stop_details = None
                text_content = '{"politicalness": {"score": 1, "insufficient_evidence": false'
            elif mode == "invalid_schema":
                stop_reason = "end_turn"
                stop_details = None
                text_content = '{"politicalness": "not_an_object", "heat": 0.9}'
            elif mode == "bad_evidence_span":
                stop_reason = "end_turn"
                stop_details = None
                sim_out = self._simulate_rubric_output(
                    custom_id=cid,
                    model_id=model_id,
                    user_envelope_json=user_msg,
                    fault_spec=f_spec,
                )
                sim_out["critical"]["score"] = 2
                sim_out["critical"]["insufficient_evidence"] = False
                sim_out["critical"]["reason_code"] = "DIRECT_NEGATIVE_EVALUATION"
                sim_out["critical"]["evidence_span"] = "FABRICATED_SPAN_NOT_IN_POST_99999"
                text_content = json.dumps(sim_out, sort_keys=True)
            else:
                stop_reason = "end_turn"
                stop_details = None
                sim_out = self._simulate_rubric_output(
                    custom_id=cid,
                    model_id=model_id,
                    user_envelope_json=user_msg,
                    fault_spec=f_spec,
                )
                text_content = json.dumps(sim_out, sort_keys=True)

            results_list.append({
                "custom_id": cid,
                "simulated": True,
                "not_for_model_training": True,
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": f"msg_sim_{cid[:24]}",
                        "type": "message",
                        "role": "assistant",
                        "model": returned_model,
                        "stop_reason": stop_reason,
                        "stop_details": stop_details,
                        "content": [{"type": "text", "text": text_content}],
                        "usage": {
                            "input_tokens": in_toks,
                            "output_tokens": out_toks,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        },
                    },
                },
            })

        if self.state.get("out_of_order_results", True) and len(results_list) > 1:
            results_list = list(reversed(results_list))

        self.state["batches"][provider_batch_id] = {
            "provider_batch_id": provider_batch_id,
            "client_submission_id": client_submission_id,
            "processing_status": "ended",
            "request_counts": counts,
            "created_at": now,
            "ended_at": now,
            "custom_ids": [r["custom_id"] for r in req_dicts],
            "results": results_list,
        }
        self._save()

        if self.state.get("crash_after_accept_once"):
            self.state["crash_after_accept_once"] = False
            self._save()
            raise ConnectionError("Simulated crash after provider accepted batch but before client saved batch ID")

        return self.retrieve_batch(provider_batch_id)

    def retrieve_batch(self, provider_batch_id: str) -> BatchStatusSnapshot:
        self._load_or_init()
        if provider_batch_id not in self.state["batches"]:
            raise KeyError(f"Batch {provider_batch_id} not found in fake provider")
        b = self.state["batches"][provider_batch_id]
        return BatchStatusSnapshot(
            provider_batch_id=b["provider_batch_id"],
            processing_status=b["processing_status"],
            request_counts=dict(b["request_counts"]),
            created_at=b["created_at"],
            ended_at=b.get("ended_at"),
            custom_ids_sample=tuple(b.get("custom_ids", [])),
        )

    def list_batches(self, limit: int = 20) -> list[BatchStatusSnapshot]:
        self._load_or_init()
        snaps: list[BatchStatusSnapshot] = []
        for bid in sorted(self.state["batches"].keys(), reverse=True)[:limit]:
            snaps.append(self.retrieve_batch(bid))
        return snaps

    def cancel_batch(self, provider_batch_id: str) -> BatchStatusSnapshot:
        self._load_or_init()
        if provider_batch_id not in self.state["batches"]:
            raise KeyError(f"Batch {provider_batch_id} not found")
        b = self.state["batches"][provider_batch_id]
        # Partial completion on cancel: keep first half succeeded, mark second half canceled
        results = b["results"]
        mid = max(1, len(results) // 2) if len(results) > 1 else 0
        new_results = []
        succeeded = errored = canceled = expired = 0
        for idx, item in enumerate(results):
            if idx >= mid:
                new_results.append({
                    "custom_id": item["custom_id"],
                    "simulated": True,
                    "result": {"type": "canceled"},
                })
                canceled += 1
            else:
                new_results.append(item)
                rtype = item["result"]["type"]
                if rtype == "succeeded":
                    succeeded += 1
                elif rtype == "errored":
                    errored += 1
                elif rtype == "expired":
                    expired += 1
        b["results"] = new_results
        b["processing_status"] = "ended"
        b["request_counts"] = {
            "processing": 0,
            "succeeded": succeeded,
            "errored": errored,
            "canceled": canceled,
            "expired": expired,
        }
        self._save()
        return self.retrieve_batch(provider_batch_id)

    def iter_batch_results(self, provider_batch_id: str) -> Iterator[dict[str, Any]]:
        self._load_or_init()
        if provider_batch_id not in self.state["batches"]:
            raise KeyError(f"Batch {provider_batch_id} not found")
        for item in self.state["batches"][provider_batch_id]["results"]:
            yield copy.deepcopy(item)


SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA: dict[str, str] = {
    "true": (
        "Unscorable without unseen external material (such as a bare URL/link, "
        "bare @mention, unattached image/video with no substantive prose, or "
        "unhydrated reply/quote)."
    ),
    "false": (
        "Scorable from the substantive prose shown above (including 0 when the "
        "post has self-contained readable prose where this attribute is absent)."
    ),
}

_SYSTEMONE_URL_RE = re.compile(
    r"(?:https?:/+\S+|(?:www\.|[a-zA-Z0-9-]+\.(?:com|org|net|io|gov|edu|me|co|be|ly|social|app|tv|news|blog|la|gg|xyz|info|uk|de|fr|jp|br))(?:/\S*)?)",
    re.IGNORECASE,
)
_SYSTEMONE_HANDLE_RE = re.compile(r"@[a-zA-Z0-9._-]+")
_SYSTEMONE_CREDIT_RE = re.compile(r"[🎥📸📹📷🎬]\s*\S+")
_SYSTEMONE_DATE_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?\b")
_SYSTEMONE_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F]{2,}")


def _analyze_systemone_text_substance(
    text: str,
    context_texts: list[str],
    structural_signals: dict[str, Any],
) -> dict[str, Any]:
    """Compute non-leaking structural diagnostics for bare links, @mentions, and unattached media."""
    combined = " ".join([text, *context_texts])
    has_url_in_text = bool(_SYSTEMONE_URL_RE.search(combined))
    has_handle_in_text = bool(_SYSTEMONE_HANDLE_RE.search(combined))
    cleaned = _SYSTEMONE_URL_RE.sub(" ", combined)
    cleaned = _SYSTEMONE_HANDLE_RE.sub(" ", cleaned)
    cleaned = _SYSTEMONE_CREDIT_RE.sub(" ", cleaned)
    cleaned = _SYSTEMONE_DATE_RE.sub(" ", cleaned)
    words = _SYSTEMONE_WORD_RE.findall(cleaned)
    has_media = bool(
        int(structural_signals.get("image_count") or 0) > 0
        or structural_signals.get("has_video")
        or structural_signals.get("embed_modality") in ("images", "video", "recordWithMedia")
    )
    has_ext = bool(structural_signals.get("has_external_link") or has_url_in_text)
    return {
        "media_visual_included": False,
        "external_link_body_included": False,
        "substantive_prose_word_count": len(words),
        "is_link_or_mention_only": len(words) == 0 and (has_ext or has_handle_in_text),
        "is_media_without_substantive_prose": len(words) == 0 and has_media,
    }


def build_systemone_clef_request(
    row: dict[str, Any],
    *,
    config: HarnessConfig,
    model_name: str = "Cloudflare/clef-flash",
    observation_mode: str = "text_only",
    corpus_by_uri: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Jev / Cloudflare Clef `/v1/systemone` bounded-decision request for the v3 rubric.

    Decomposes each component into an ordered 0..3 `score` question plus an explicit
    insufficiency `noul` check with non-colliding `criteria` options, alongside `choice`
    questions for topic, missing_context_kind, and stance_ambiguity.
    """
    obs = resolve_post_observation(row, corpus_by_uri=corpus_by_uri, observation_mode=observation_mode)
    state_envelope = copy.deepcopy(obs["envelope"])
    state_envelope["structural_signals"].update(
        _analyze_systemone_text_substance(
            obs["text"],
            obs["context_texts"],
            state_envelope["structural_signals"],
        )
    )

    questions: dict[str, dict[str, Any]] = {}
    for comp in COMPONENT_ORDER:
        spec = config.rubric["components"][comp]
        boundary_rules = " ".join(spec.get("boundary_rules") or [])
        score_instr = f"{spec['definition']} {boundary_rules} Rate only from observable text.".strip()
        questions[f"{comp}_score"] = {
            "type": "score",
            "instructions": score_instr,
            "criteria": [
                spec["anchors"]["0"],
                spec["anchors"]["1"],
                spec["anchors"]["2"],
                spec["anchors"]["3"],
            ],
        }
        questions[f"{comp}_insufficient"] = {
            "type": "noul",
            "instructions": (
                f"Is there insufficient evidence in the visible text to determine '{comp}' "
                "(because the post is a bare URL/link, bare @mention, image/video with no substantive prose, "
                "or unhydrated reply/quote requiring unseen parent, quote, media, link, or repetition history)?"
            ),
            "criteria": dict(SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA),
        }

    questions["topic"] = {
        "type": "choice",
        "instructions": "Select the primary topic category of the post.",
        "criteria": {t: f"Topic category: {t}" for t in config.rubric["topics"]},
    }
    questions["context_dependent"] = {
        "type": "noul",
        "instructions": "Does accurate interpretation depend on missing parent, quote, media, or link context?",
        "criteria": dict(SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA),
    }
    questions["missing_context_kind"] = {
        "type": "choice",
        "instructions": "If context is missing, what kind of context is missing?",
        "criteria": {k: f"Missing context kind: {k}" for k in config.rubric["missing_context_kinds"]},
    }
    questions["stance_ambiguity"] = {
        "type": "choice",
        "instructions": "Does the post quote, report, or satirize a stance rather than directly endorsing it?",
        "criteria": {s: f"Stance ambiguity: {s}" for s in config.rubric["stance_ambiguities"]},
    }

    payload = {
        "model": model_name,
        "state": state_envelope,
        "questions": questions,
    }
    assert_no_credentials(payload, "systemone_request")
    return payload


def parse_systemone_clef_response(
    response: dict[str, Any],
    *,
    insufficient_threshold: float = 0.50,
) -> dict[str, Any]:
    """Parse a Jev / Cloudflare Clef `/v1/systemone` response into v3 ordinal/continuous targets + calibration metrics.

    For each component in `COMPONENT_ORDER`:
    - Reads `<comp>_score` (`type == "score"`) with Brier-calibrated probabilities `{"0": p0, "1": p1, "2": p2, "3": p3}`,
      expected ordinal score `E[y] = sum(k * p_k)` in `[0.0, 3.0]`, normalized continuous score `E[y] / 3.0` in `[0.0, 1.0]`,
      MAP ordinal class `argmax_k p_k` in `{0, 1, 2, 3}`, confidence `max_k p_k`, and normalized Shannon entropy.
    - Reads `<comp>_insufficient` (`type == "noul"`) probability `noul` in `[0.0, 1.0]`.
    - Sets `ordinal_score = "insufficient"` when `noul >= insufficient_threshold` (without coercing to 0).
    """
    import math

    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("Invalid SystemOne response: missing 'answers' dict")
    answers = response["answers"]

    components_out: dict[str, dict[str, Any]] = {}
    trainer_continuous_0_1: dict[str, float | None] = {}
    ordinal_scores: dict[str, int | str] = {}
    any_insufficient = False

    for comp in COMPONENT_ORDER:
        score_ans = answers.get(f"{comp}_score")
        insuff_ans = answers.get(f"{comp}_insufficient")
        if not isinstance(score_ans, dict) or score_ans.get("type") != "score":
            raise ValueError(f"Missing or invalid score answer for '{comp}_score'")
        if not isinstance(insuff_ans, dict) or insuff_ans.get("type") != "noul":
            raise ValueError(f"Missing or invalid noul answer for '{comp}_insufficient'")

        probs_raw = score_ans.get("probabilities") or {}
        probs = {str(k): float(probs_raw.get(str(k), 0.0)) for k in range(4)}
        prob_sum = sum(probs.values())
        if prob_sum > 0:
            probs = {k: v / prob_sum for k, v in probs.items()}

        expected_ordinal_0_3 = sum(k * probs[str(k)] for k in range(4))
        continuous_0_1 = round(expected_ordinal_0_3 / 3.0, 6)
        map_level = max(range(4), key=lambda k: probs[str(k)])
        confidence = round(float(score_ans.get("confidence", probs[str(map_level)])), 6)

        # Normalized Shannon entropy over 4 levels (0 = point mass, 1 = uniform)
        entropy_bits = -sum(p * math.log2(p) for p in probs.values() if p > 1e-12)
        normalized_entropy = round(entropy_bits / 2.0, 6)

        noul_prob = float(insuff_ans.get("noul", 0.0))
        is_insufficient = noul_prob >= insufficient_threshold
        if is_insufficient:
            any_insufficient = True
            ordinal_val: int | str = "insufficient"
            trainer_val: float | None = None
        else:
            ordinal_val = int(map_level)
            trainer_val = continuous_0_1

        ordinal_scores[comp] = ordinal_val
        trainer_continuous_0_1[comp] = trainer_val
        components_out[comp] = {
            "score": ordinal_val,
            "ordinal_score": ordinal_val,
            "map_ordinal_0_3": int(map_level),
            "expected_ordinal_0_3": round(expected_ordinal_0_3, 6),
            "continuous_0_1": continuous_0_1,
            "normalized_0_to_1": trainer_val,
            "trainer_target_0_1": trainer_val,
            "insufficient_evidence": is_insufficient,
            "insufficient_noul_prob": round(noul_prob, 6),
            "confidence": confidence,
            "normalized_entropy": normalized_entropy,
            "probabilities": {k: round(v, 6) for k, v in probs.items()},
        }

    topic_ans = answers.get("topic") or {}
    context_dep_ans = answers.get("context_dependent") or {}
    missing_kind_ans = answers.get("missing_context_kind") or {}
    stance_ans = answers.get("stance_ambiguity") or {}

    context_dep_noul = float(context_dep_ans.get("noul", 0.0))
    context_dependent = context_dep_noul >= insufficient_threshold or any_insufficient
    missing_context_kind = str(missing_kind_ans.get("choice", "none"))
    if any_insufficient and missing_context_kind == "none":
        missing_context_kind = "reply_parent"

    out: dict[str, Any] = {
        "model": response.get("model", "Cloudflare/clef-flash"),
        "usage": response.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        "components": components_out,
        "ordinal_scores": ordinal_scores,
        "trainer_continuous_0_1": trainer_continuous_0_1,
        "has_insufficient_component": any_insufficient,
        "topic": str(topic_ans.get("choice", "other")),
        "topic_confidence": float(topic_ans.get("confidence", 0.0)),
        "context_dependent": context_dependent,
        "context_dependent_noul_prob": round(context_dep_noul, 6),
        "missing_context_kind": missing_context_kind,
        "missing_context_confidence": float(missing_kind_ans.get("confidence", 0.0)),
        "stance_ambiguity": str(stance_ans.get("choice", "none")),
        "stance_ambiguity_confidence": float(stance_ans.get("confidence", 0.0)),
    }
    for comp, comp_dict in components_out.items():
        out[comp] = comp_dict
    return out

