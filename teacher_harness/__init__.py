"""Catbird Anthropic Teacher Harness package."""

from __future__ import annotations

from .config import (
    BudgetConstraintError,
    ConfigurationMismatchError,
    CredentialLeakError,
    HarnessConfig,
    assert_no_credentials,
    canonical_json_bytes,
    canonical_sha256,
    format_usd,
    scrub_environment_snapshot,
)
from .db import (
    ArtifactIntegrityError,
    HarnessDB,
    ProcessLockError,
    acquire_writer_lock,
    utc_now_iso,
)
from .export import (
    ORDINAL_MAPPING_V1,
    TrainerExportValidationError,
    export_annotation_bank_and_trainer_targets,
    map_ordinal_to_training_score,
)
from .moderation import (
    fold_and_join_moderation_labels,
    iter_packed_cbor_labels,
    load_moderation_events,
)
from .pipeline import (
    run_collect,
    run_plan,
    run_prepare,
    run_reconcile,
    run_report,
    run_submit,
)
from .routing import evaluate_post_disagreement, route_review_candidates
from .schemas import (
    COMPONENT_ORDER,
    ComponentAssessment,
    SchemaValidationError,
    TeacherAnnotationOutput,
    build_batch_request_for_post,
    build_system_prompt,
    estimate_input_tokens_offline,
    make_custom_id,
    resolve_post_observation,
    validate_teacher_output_json,
    verify_prompt_isolation,
)
from .splits import (
    SplitLeakageError,
    near_duplicate_fingerprint,
    prepare_and_freeze_corpus,
)
from .transport import (
    AnthropicTransport,
    AuthorizationGateError,
    BatchStatusSnapshot,
    FakeAnthropicTransport,
    OfficialAnthropicSDKTransport,
    ProviderRateLimitError,
    build_systemone_clef_request,
    parse_systemone_clef_response,
)

__all__ = [
    "AnthropicTransport",
    "ArtifactIntegrityError",
    "AuthorizationGateError",
    "BatchStatusSnapshot",
    "BudgetConstraintError",
    "COMPONENT_ORDER",
    "ComponentAssessment",
    "ConfigurationMismatchError",
    "CredentialLeakError",
    "FakeAnthropicTransport",
    "HarnessConfig",
    "HarnessDB",
    "ORDINAL_MAPPING_V1",
    "OfficialAnthropicSDKTransport",
    "ProcessLockError",
    "ProviderRateLimitError",
    "SchemaValidationError",
    "SplitLeakageError",
    "TeacherAnnotationOutput",
    "TrainerExportValidationError",
    "acquire_writer_lock",
    "assert_no_credentials",
    "build_batch_request_for_post",
    "build_system_prompt",
    "build_systemone_clef_request",
    "canonical_json_bytes",
    "canonical_sha256",
    "estimate_input_tokens_offline",
    "evaluate_post_disagreement",
    "export_annotation_bank_and_trainer_targets",
    "fold_and_join_moderation_labels",
    "format_usd",
    "iter_packed_cbor_labels",
    "load_moderation_events",
    "make_custom_id",
    "map_ordinal_to_training_score",
    "near_duplicate_fingerprint",
    "parse_systemone_clef_response",
    "prepare_and_freeze_corpus",
    "resolve_post_observation",
    "route_review_candidates",
    "run_collect",
    "run_plan",
    "run_prepare",
    "run_reconcile",
    "run_report",
    "run_submit",
    "scrub_environment_snapshot",
    "utc_now_iso",
    "validate_teacher_output_json",
    "verify_prompt_isolation",
]
