"""Index-scoped semantic extraction planning for batch and targeted retries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from src.analysis.constants import ANTHROPIC_BATCH_PRICING, DEFAULT_MODELS, OPENAI_PRICING
from src.analysis.dimensions import (
    DimensionProfile,
    get_default_dimension_registry,
    load_dimension_profile,
)
from src.analysis.schemas import DimensionedExtraction, ExtractionResult
from src.analysis.semantic_prompts import SEMANTIC_PROMPT_VERSION, build_pass_definitions
from src.config import Config
from src.indexing.structured_store import StructuredStore
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import get_logger
from src.zotero.models import Author, PaperMetadata

logger = get_logger(__name__)

SEMANTIC_BATCH_STATE_DIRNAME = ".semantic_batches"
SEMANTIC_BATCH_SCHEMA_VERSION = "1.0.0"
SEMANTIC_BATCH_MODE = "batch_api"
_DEFAULT_INDEX_TIMESTAMP = "2020-01-01T00:00:00"


@dataclass
class ResolvedSemanticPaper:
    """One paper selected for semantic extraction."""

    paper: PaperMetadata
    text: str
    text_source: str
    raw_char_count: int
    raw_word_count: int
    llm_char_count: int


@dataclass
class SemanticBatchPlan:
    """Concrete batch plan for one index/profile/provider run."""

    index_dir: Path
    provider: str
    mode: str
    model: str
    reasoning_effort: str | None
    profile: DimensionProfile
    prompt_version: str
    pass_definitions: list[tuple[str, list[tuple[str, str]]]]
    selected: list[ResolvedSemanticPaper]
    requested_paper_ids: list[str]
    missing_paper_ids: list[str]
    skipped_existing: list[str]
    skipped_missing_text: list[str]
    skipped_excluded: list[str]
    include_existing: bool
    allow_abstract_fallback: bool
    live_text_fallback: bool
    estimated_cost: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe summary of the plan."""

        text_source_counts = Counter(item.text_source for item in self.selected)
        return {
            "index_dir": str(self.index_dir),
            "provider": self.provider,
            "mode": self.mode,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "profile_id": self.profile.profile_id,
            "profile_version": self.profile.version,
            "profile_fingerprint": self.profile.fingerprint,
            "prompt_version": self.prompt_version,
            "requested_count": len(self.requested_paper_ids),
            "selected_count": len(self.selected),
            "selected_papers": [
                {
                    "paper_id": item.paper.paper_id,
                    "title": item.paper.title,
                    "text_source": item.text_source,
                    "raw_char_count": item.raw_char_count,
                    "raw_word_count": item.raw_word_count,
                    "llm_char_count": item.llm_char_count,
                }
                for item in self.selected
            ],
            "selected_paper_ids": [item.paper.paper_id for item in self.selected],
            "text_source_counts": dict(text_source_counts),
            "missing_paper_ids": list(self.missing_paper_ids),
            "skipped_existing": list(self.skipped_existing),
            "skipped_missing_text": list(self.skipped_missing_text),
            "skipped_excluded": list(self.skipped_excluded),
            "include_existing": self.include_existing,
            "allow_abstract_fallback": self.allow_abstract_fallback,
            "live_text_fallback": self.live_text_fallback,
            "estimated_cost": dict(self.estimated_cost),
        }


class BatchClientProtocol(Protocol):
    """Minimal interface shared by provider-specific semantic batch clients."""

    def create_batch_requests(
        self,
        papers: list[PaperMetadata],
        text_getter: Callable[[PaperMetadata], str],
        pass_definitions: list[tuple[str, list[tuple[str, str]]]] | None = None,
    ) -> list[Any]: ...

    def submit_batch(self, requests: list[Any], *, persist_state: bool = True) -> str: ...

    def get_batch_status(self, batch_id: str) -> Any: ...

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        max_wait: int = 86400,
        progress_callback: Callable[..., Any] | None = None,
    ) -> Any: ...

    def get_results(
        self,
        batch_id: str,
        *,
        pass_definitions: list[tuple[str, list[tuple[str, str]]]] | None = None,
        profile: DimensionProfile | None = None,
        prompt_version: str = SEMANTIC_PROMPT_VERSION,
    ) -> Iterator[ExtractionResult]: ...


def semantic_batch_state_dir(index_dir: Path) -> Path:
    """Return the per-index state directory for semantic batch jobs."""

    state_dir = index_dir / SEMANTIC_BATCH_STATE_DIRNAME
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def resolve_dimension_profile(
    *,
    index_dir: Path,
    config: Config,
    reference: str | None = None,
) -> DimensionProfile:
    """Resolve a profile id/path, defaulting to the index snapshot profile."""

    if reference is None:
        return DimensionProfile(**StructuredStore(index_dir).load_dimension_profile())

    candidate = Path(reference)
    if candidate.exists():
        return load_dimension_profile(candidate)

    registry = get_default_dimension_registry()
    return registry.get_profile(reference)


def plan_semantic_batch(
    *,
    index_dir: Path,
    config: Config,
    provider: str,
    paper_ids: list[str] | None = None,
    exclude_paper_ids: list[str] | None = None,
    limit: int | None = None,
    include_existing: bool = False,
    allow_abstract_fallback: bool = False,
    live_text_fallback: bool = False,
    prompt_version: str = SEMANTIC_PROMPT_VERSION,
    profile_reference: str | None = None,
) -> SemanticBatchPlan:
    """Build an index-scoped semantic batch plan."""

    store = StructuredStore(index_dir)
    papers = store.load_papers()
    existing = store.load_extractions()
    profile = resolve_dimension_profile(
        index_dir=index_dir,
        config=config,
        reference=profile_reference,
    )
    pass_definitions = build_pass_definitions(profile)
    provider_settings = config.extraction.get_provider_settings(provider)
    model = provider_settings.model or DEFAULT_MODELS[provider]
    mode = SEMANTIC_BATCH_MODE
    reasoning_effort = provider_settings.reasoning_effort
    requested_ids = list(dict.fromkeys(paper_ids or list(papers)))
    excluded_ids = set(exclude_paper_ids or [])

    selected: list[ResolvedSemanticPaper] = []
    missing_paper_ids: list[str] = []
    skipped_existing: list[str] = []
    skipped_missing_text: list[str] = []
    skipped_excluded: list[str] = []

    for paper_id in requested_ids:
        if limit is not None and len(selected) >= limit:
            break
        if paper_id not in papers:
            missing_paper_ids.append(paper_id)
            continue
        if paper_id in excluded_ids:
            skipped_excluded.append(paper_id)
            continue
        record = existing.get(paper_id)
        if (
            not include_existing
            and record is not None
            and _record_matches_target(
                record=record,
                profile=profile,
                prompt_version=prompt_version,
            )
        ):
            skipped_existing.append(paper_id)
            continue
        paper = _paper_from_index_dict(paper_id, papers[paper_id])
        resolved = _resolve_text_for_paper(
            store=store,
            paper=paper,
            config=config,
            allow_abstract_fallback=allow_abstract_fallback,
            live_text_fallback=live_text_fallback,
        )
        if resolved is None:
            skipped_missing_text.append(paper_id)
            continue
        selected.append(resolved)

    llm_lengths = [item.llm_char_count for item in selected]
    avg_text_length = int(sum(llm_lengths) / len(llm_lengths)) if llm_lengths else 0
    estimated_cost = estimate_semantic_batch_cost(
        provider=provider,
        model=model,
        num_papers=len(selected),
        avg_text_length=avg_text_length,
        num_passes=len(pass_definitions),
    )

    return SemanticBatchPlan(
        index_dir=index_dir,
        provider=provider,
        mode=mode,
        model=model,
        reasoning_effort=reasoning_effort,
        profile=profile,
        prompt_version=prompt_version,
        pass_definitions=pass_definitions,
        selected=selected,
        requested_paper_ids=requested_ids,
        missing_paper_ids=missing_paper_ids,
        skipped_existing=skipped_existing,
        skipped_missing_text=skipped_missing_text,
        skipped_excluded=skipped_excluded,
        include_existing=include_existing,
        allow_abstract_fallback=allow_abstract_fallback,
        live_text_fallback=live_text_fallback,
        estimated_cost=estimated_cost,
    )


def submit_semantic_batch(
    plan: SemanticBatchPlan,
    *,
    config: Config,
) -> dict[str, Any]:
    """Submit a planned semantic batch and persist its manifest."""

    if not plan.selected:
        raise ValueError("No papers selected for batch submission")

    client = create_batch_client(
        provider=plan.provider,
        model=plan.model,
        reasoning_effort=plan.reasoning_effort,
        batch_dir=semantic_batch_state_dir(plan.index_dir),
    )
    text_lookup = {item.paper.paper_id: item.text for item in plan.selected}
    requests = client.create_batch_requests(
        [item.paper for item in plan.selected],
        text_getter=lambda paper: text_lookup[paper.paper_id],
        pass_definitions=plan.pass_definitions,
    )
    batch_id = client.submit_batch(requests, persist_state=False)
    manifest = build_batch_manifest(plan=plan, batch_id=batch_id)
    save_batch_manifest(plan.index_dir, manifest)
    return manifest


def build_batch_manifest(
    *,
    plan: SemanticBatchPlan,
    batch_id: str,
) -> dict[str, Any]:
    """Build the persistent manifest for one submitted batch."""

    return {
        "schema_version": SEMANTIC_BATCH_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(),
        "index_dir": str(plan.index_dir),
        "provider": plan.provider,
        "mode": plan.mode,
        "model": plan.model,
        "reasoning_effort": plan.reasoning_effort,
        "prompt_version": plan.prompt_version,
        "profile_snapshot": plan.profile.model_dump(mode="json"),
        "pass_definitions": serialize_pass_definitions(plan.pass_definitions),
        "paper_ids": [item.paper.paper_id for item in plan.selected],
        "text_sources": {
            item.paper.paper_id: {
                "text_source": item.text_source,
                "raw_char_count": item.raw_char_count,
                "raw_word_count": item.raw_word_count,
                "llm_char_count": item.llm_char_count,
            }
            for item in plan.selected
        },
        "total_requests": len(plan.selected) * len(plan.pass_definitions),
        "passes_per_paper": len(plan.pass_definitions),
        "estimated_cost": dict(plan.estimated_cost),
        "include_existing": plan.include_existing,
        "allow_abstract_fallback": plan.allow_abstract_fallback,
        "live_text_fallback": plan.live_text_fallback,
        "requested_paper_ids": list(plan.requested_paper_ids),
        "missing_paper_ids": list(plan.missing_paper_ids),
        "skipped_existing": list(plan.skipped_existing),
        "skipped_missing_text": list(plan.skipped_missing_text),
        "skipped_excluded": list(plan.skipped_excluded),
    }


def load_batch_manifest(index_dir: Path, batch_id: str) -> dict[str, Any]:
    """Load one batch manifest from the index-scoped state directory."""

    manifest_path = semantic_batch_state_dir(index_dir) / f"{batch_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")
    manifest = safe_read_json(manifest_path, default={})
    if not isinstance(manifest, dict) or manifest.get("batch_id") != batch_id:
        raise ValueError(f"Invalid batch manifest: {manifest_path}")
    return manifest


def save_batch_manifest(index_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write one batch manifest to the index-scoped state directory."""

    batch_id = str(manifest.get("batch_id") or "").strip()
    if not batch_id:
        raise ValueError("Batch manifest is missing batch_id")
    manifest_path = semantic_batch_state_dir(index_dir) / f"{batch_id}.json"
    safe_write_json(manifest_path, manifest)
    return manifest_path


def list_batch_manifests(
    index_dir: Path,
    *,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """List stored manifests for an index, optionally filtered by provider."""

    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(semantic_batch_state_dir(index_dir).glob("*.json")):
        manifest = safe_read_json(manifest_path, default={})
        if not isinstance(manifest, dict) or "batch_id" not in manifest:
            continue
        if provider and manifest.get("provider") != provider:
            continue
        manifests.append(manifest)
    return manifests


def get_semantic_batch_status(
    *,
    index_dir: Path,
    batch_id: str,
    config: Config,
) -> tuple[dict[str, Any], Any]:
    """Load one manifest and fetch its remote provider status."""

    manifest = load_batch_manifest(index_dir, batch_id)
    client = create_batch_client(
        provider=str(manifest["provider"]),
        model=str(manifest["model"]),
        reasoning_effort=manifest.get("reasoning_effort"),
        batch_dir=semantic_batch_state_dir(index_dir),
    )
    status = client.get_batch_status(batch_id)
    manifest["last_status"] = status.status
    manifest["last_checked_at"] = datetime.now().isoformat()
    save_batch_manifest(index_dir, manifest)
    return manifest, status


def wait_for_semantic_batch(
    *,
    index_dir: Path,
    batch_id: str,
    config: Config,
    poll_interval: int = 60,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """Wait for a submitted batch to complete and persist final status."""

    manifest = load_batch_manifest(index_dir, batch_id)
    client = create_batch_client(
        provider=str(manifest["provider"]),
        model=str(manifest["model"]),
        reasoning_effort=manifest.get("reasoning_effort"),
        batch_dir=semantic_batch_state_dir(index_dir),
    )
    status = client.wait_for_batch(
        batch_id,
        poll_interval=poll_interval,
        progress_callback=progress_callback,
    )
    manifest["last_status"] = status.status
    manifest["last_checked_at"] = datetime.now().isoformat()
    manifest["completed_at"] = datetime.now().isoformat()
    save_batch_manifest(index_dir, manifest)
    return manifest, status


def collect_semantic_batch(
    *,
    index_dir: Path,
    batch_id: str,
    config: Config,
    logger_override: Logger | None = None,
) -> dict[str, Any]:
    """Collect one batch into semantic_analyses.json without embeddings/similarity."""

    active_logger = logger_override or logger
    manifest = load_batch_manifest(index_dir, batch_id)
    profile = DimensionProfile(**manifest["profile_snapshot"])
    _ensure_index_profile_compatible(index_dir=index_dir, profile=profile)

    client = create_batch_client(
        provider=str(manifest["provider"]),
        model=str(manifest["model"]),
        reasoning_effort=manifest.get("reasoning_effort"),
        batch_dir=semantic_batch_state_dir(index_dir),
    )
    store = StructuredStore(index_dir)
    extractions = dict(store.load_extractions())
    pass_definitions = deserialize_pass_definitions(manifest["pass_definitions"])

    added = 0
    failed = 0
    failed_ids: list[str] = []

    for result in client.get_results(
        batch_id,
        pass_definitions=pass_definitions,
        profile=profile,
        prompt_version=str(manifest["prompt_version"]),
    ):
        if result.success and result.extraction:
            extractions[result.paper_id] = {
                "paper_id": result.paper_id,
                "extraction": result.extraction.to_index_dict(),
                "timestamp": datetime.now().isoformat(),
                "model": result.model_used,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "batch_id": batch_id,
                "provider": manifest["provider"],
                "mode": manifest["mode"],
            }
            added += 1
        else:
            failed += 1
            failed_ids.append(result.paper_id)
            active_logger.warning(
                "Semantic batch %s failed for %s: %s",
                batch_id,
                result.paper_id,
                result.error,
            )

    store.save_extractions(extractions)
    store.save_dimension_profile(profile.model_dump(mode="json"))

    from src.indexing.pipeline import generate_summary

    generate_summary(index_dir, active_logger)

    manifest["last_status"] = "collected"
    manifest["last_checked_at"] = datetime.now().isoformat()
    manifest["collected_at"] = datetime.now().isoformat()
    manifest["collected_count"] = added
    manifest["failed_count"] = failed
    manifest["failed_paper_ids"] = failed_ids
    save_batch_manifest(index_dir, manifest)

    return {
        "batch_id": batch_id,
        "provider": manifest["provider"],
        "model": manifest["model"],
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "added": added,
        "failed": failed,
        "failed_paper_ids": failed_ids,
        "total_extractions": len(extractions),
    }


def list_pending_semantic_batches(
    *,
    index_dir: Path,
    config: Config,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Return non-terminal manifest/status pairs for an index."""

    pending: list[dict[str, Any]] = []
    for manifest in list_batch_manifests(index_dir, provider=provider):
        client = create_batch_client(
            provider=str(manifest["provider"]),
            model=str(manifest["model"]),
            reasoning_effort=manifest.get("reasoning_effort"),
            batch_dir=semantic_batch_state_dir(index_dir),
        )
        status = client.get_batch_status(str(manifest["batch_id"]))
        manifest["last_status"] = status.status
        manifest["last_checked_at"] = datetime.now().isoformat()
        save_batch_manifest(index_dir, manifest)
        if is_terminal_batch_status(provider=str(manifest["provider"]), status=status.status):
            continue
        pending.append(
            {
                "batch_id": manifest["batch_id"],
                "provider": manifest["provider"],
                "model": manifest["model"],
                "status": status.status,
                "completed_requests": status.completed_requests,
                "total_requests": status.total_requests,
                "paper_count": len(manifest.get("paper_ids", [])),
                "profile_id": manifest.get("profile_snapshot", {}).get("profile_id"),
            }
        )
    return pending


def retry_semantic_papers(
    *,
    index_dir: Path,
    config: Config,
    logger_override: Logger | None,
    paper_ids: list[str],
    provider: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    parallel: int | None = None,
    no_cache: bool = False,
    profile: DimensionProfile | None = None,
) -> dict[str, Any]:
    """Re-run semantic extraction for selected papers using stored fulltext snapshots."""

    active_logger = logger_override or logger
    store = StructuredStore(index_dir)
    papers = store.load_papers()
    extractions = dict(store.load_extractions())
    selected_ids = list(dict.fromkeys(paper_ids))
    missing_ids = [paper_id for paper_id in selected_ids if paper_id not in papers]
    if missing_ids:
        raise ValueError(f"Unknown paper_ids: {', '.join(missing_ids)}")

    selected_papers = [
        _paper_from_index_dict(paper_id, papers[paper_id]) for paper_id in selected_ids
    ]
    text_snapshots: dict[str, dict[str, object]] = {}
    missing_snapshots: list[str] = []
    for paper in selected_papers:
        snapshot = store.load_text_snapshot(paper.paper_id)
        if snapshot is None:
            missing_snapshots.append(paper.paper_id)
            continue
        text_snapshots[paper.paper_id] = snapshot

    if missing_snapshots:
        raise ValueError("Missing fulltext snapshots for: " + ", ".join(sorted(missing_snapshots)))

    from src.indexing.pipeline import (
        build_section_extractor,
        configure_extraction_runtime,
        generate_summary,
    )

    runtime_args = SimpleNamespace(
        provider=provider,
        mode=mode,
        model=model,
        parallel=parallel,
        no_cache=no_cache,
        use_subscription=False,
    )
    _provider, runtime_mode, cache_dir, parallel_workers, use_cache = configure_extraction_runtime(
        runtime_args,
        config,
        active_logger,
    )
    extractor = build_section_extractor(
        runtime_args,
        config,
        cache_dir=cache_dir,
        mode=runtime_mode,
        parallel_workers=parallel_workers,
        use_cache=use_cache,
    )

    updated = 0
    failed = 0
    failed_ids: list[str] = []
    for result in extractor.extract_batch(
        selected_papers,
        text_snapshots=text_snapshots,
    ):
        if result.success and result.extraction and not result.pass_errors:
            extractions[result.paper_id] = {
                "paper_id": result.paper_id,
                "extraction": result.extraction.to_index_dict(),
                "timestamp": result.timestamp.isoformat(),
                "model": result.model_used,
                "duration": result.duration_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "extraction_method": result.extraction_method,
                "planned_extraction_intent": result.planned_extraction_intent,
                "planned_profile_key": result.planned_profile_key,
                "actual_profile_key": result.actual_profile_key,
                "extraction_routing_overridden": result.extraction_routing_overridden,
                "extraction_override_reason": result.extraction_override_reason,
                "extraction_diagnostics": dict(result.extraction_diagnostics),
            }
            updated += 1
        else:
            failed += 1
            failed_ids.append(result.paper_id)

    store.save_extractions(extractions)
    if profile is not None and _should_save_retry_profile_snapshot(
        store=store,
        profile=profile,
        selected_ids=selected_ids,
        total_papers=len(papers),
    ):
        store.save_dimension_profile(profile.model_dump(mode="json"))
    generate_summary(index_dir, active_logger)
    return {
        "requested_paper_ids": selected_ids,
        "updated": updated,
        "failed": failed,
        "failed_paper_ids": failed_ids,
        "total_extractions": len(extractions),
    }


def _should_save_retry_profile_snapshot(
    *,
    store: StructuredStore,
    profile: DimensionProfile,
    selected_ids: list[str],
    total_papers: int,
) -> bool:
    """Return whether a targeted retry should update dimension_profile.json.

    Partial-scope retries are allowed during staged profile backfills where the
    index intentionally contains a mix of old and new extractions. In that
    state, flipping the index-wide active profile snapshot would break resume
    logic for the main backfill, so only persist the new profile when the index
    is already on that profile or the retry scope covers the entire corpus.
    """

    current_profile = DimensionProfile(**store.load_dimension_profile())
    if current_profile.fingerprint == profile.fingerprint:
        return True
    if len(selected_ids) == total_papers:
        return True
    logger.info(
        "Leaving dimension profile snapshot at %s during targeted retry because "
        "the retry scope is partial (%d/%d papers) and the active index profile "
        "still differs from %s",
        current_profile.profile_id,
        len(selected_ids),
        total_papers,
        profile.profile_id,
    )
    return False


def estimate_semantic_batch_cost(
    *,
    provider: str,
    model: str,
    num_papers: int,
    avg_text_length: int,
    num_passes: int,
) -> dict[str, Any]:
    """Estimate batch cost without requiring live API credentials."""

    input_tokens_per_pass = avg_text_length // 4 + 500 if avg_text_length else 0
    output_tokens_per_pass = 2000
    total_input = input_tokens_per_pass * num_papers * num_passes
    total_output = output_tokens_per_pass * num_papers * num_passes

    if provider == "anthropic":
        input_cost_per_million, output_cost_per_million = ANTHROPIC_BATCH_PRICING.get(
            model,
            (2.50, 12.50),
        )
    elif provider == "openai":
        std_input, std_output = OPENAI_PRICING.get(model, (2.50, 15.0))
        input_cost_per_million = std_input * 0.5
        output_cost_per_million = std_output * 0.5
    else:
        raise ValueError(f"Unsupported batch provider: {provider}")

    input_cost = (total_input / 1_000_000) * input_cost_per_million
    output_cost = (total_output / 1_000_000) * output_cost_per_million
    return {
        "num_papers": num_papers,
        "passes_per_paper": num_passes,
        "total_requests": num_papers * num_passes,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost": round(input_cost + output_cost, 2),
        "discount": "50% (batch API)",
        "model": model,
        "pricing": f"${input_cost_per_million}/MTok in, ${output_cost_per_million}/MTok out",
    }


def serialize_pass_definitions(
    pass_definitions: list[tuple[str, list[tuple[str, str]]]],
) -> list[dict[str, Any]]:
    """Convert tuple-based pass definitions into JSON-safe payloads."""

    return [
        {
            "label": label,
            "questions": [
                {"field_name": field_name, "question": question_text}
                for field_name, question_text in questions
            ],
        }
        for label, questions in pass_definitions
    ]


def deserialize_pass_definitions(
    payload: list[dict[str, Any]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Convert JSON-safe pass definitions into the tuple form expected by clients."""

    deserialized: list[tuple[str, list[tuple[str, str]]]] = []
    for entry in payload:
        label = str(entry.get("label") or "").strip()
        questions = [
            (str(item.get("field_name") or "").strip(), str(item.get("question") or "").strip())
            for item in entry.get("questions", [])
            if str(item.get("field_name") or "").strip()
        ]
        deserialized.append((label, questions))
    return deserialized


def create_batch_client(
    *,
    provider: str,
    model: str,
    reasoning_effort: str | None,
    batch_dir: Path,
) -> BatchClientProtocol:
    """Create a provider-specific batch client bound to one state directory."""

    if provider == "openai":
        from src.analysis.openai_batch_client import OpenAIBatchClient

        return OpenAIBatchClient(
            model=model,
            reasoning_effort=reasoning_effort,
            batch_dir=batch_dir,
        )
    if provider == "anthropic":
        from src.analysis.batch_client import BatchExtractionClient

        return BatchExtractionClient(
            model=model,
            batch_dir=batch_dir,
        )
    raise ValueError(f"Batch API not supported for provider: {provider}")


def is_terminal_batch_status(*, provider: str, status: str) -> bool:
    """Return whether a provider-specific batch status is terminal."""

    if provider == "anthropic":
        return status in {"ended", "canceled", "expired"}
    if provider == "openai":
        return status in {"completed", "failed", "expired", "cancelled"}
    return False


def _record_matches_target(
    *,
    record: dict[str, Any],
    profile: DimensionProfile,
    prompt_version: str,
) -> bool:
    """Return whether an extraction already matches the requested profile/prompt."""

    try:
        extraction = DimensionedExtraction.from_record(record)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Existing semantic extraction record is unreadable; scheduling re-extraction: %s",
            exc,
        )
        return False
    return (
        extraction.profile_fingerprint == profile.fingerprint
        and extraction.prompt_version == prompt_version
    )


def _resolve_text_for_paper(
    *,
    store: StructuredStore,
    paper: PaperMetadata,
    config: Config,
    allow_abstract_fallback: bool,
    live_text_fallback: bool,
) -> ResolvedSemanticPaper | None:
    """Resolve canonical text for one paper, preferring stored fulltext snapshots."""

    from src.extraction.text_cleaner import TextCleaner

    text_cleaner = TextCleaner()
    snapshot = store.load_text_snapshot(paper.paper_id)
    if snapshot is not None:
        snapshot_text = str(snapshot.get("text") or "")
        if snapshot_text:
            llm_text = text_cleaner.truncate_for_llm(snapshot_text)
            return ResolvedSemanticPaper(
                paper=paper,
                text=llm_text,
                text_source="fulltext_snapshot",
                raw_char_count=len(snapshot_text),
                raw_word_count=len(snapshot_text.split()),
                llm_char_count=len(llm_text),
            )

    abstract = str(paper.abstract or "").strip()
    if allow_abstract_fallback and abstract:
        llm_text = text_cleaner.truncate_for_llm(abstract)
        return ResolvedSemanticPaper(
            paper=paper,
            text=llm_text,
            text_source="abstract",
            raw_char_count=len(abstract),
            raw_word_count=len(abstract.split()),
            llm_char_count=len(llm_text),
        )

    if live_text_fallback:
        live_text = _extract_live_text(config=config, paper=paper)
        if live_text:
            llm_text = text_cleaner.truncate_for_llm(live_text)
            return ResolvedSemanticPaper(
                paper=paper,
                text=llm_text,
                text_source="live_extraction",
                raw_char_count=len(live_text),
                raw_word_count=len(live_text.split()),
                llm_char_count=len(llm_text),
            )

    return None


def _extract_live_text(*, config: Config, paper: PaperMetadata) -> str | None:
    """Run the extraction cascade for one paper when explicitly requested."""

    source_path = paper.source_path or paper.pdf_path
    if source_path is None or not Path(source_path).exists():
        return None

    from src.extraction.cascade import ExtractionCascade
    from src.extraction.opendataloader_extractor import build_hybrid_config_from_processing
    from src.extraction.pdf_extractor import PDFExtractor
    from src.extraction.text_cleaner import TextCleaner

    pdf_extractor = PDFExtractor(
        enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
    )
    text_cleaner = TextCleaner()
    hybrid_config = build_hybrid_config_from_processing(config.processing)
    cascade = ExtractionCascade(
        pdf_extractor=pdf_extractor,
        enable_arxiv=config.processing.arxiv_enabled,
        enable_opendataloader=config.processing.opendataloader_enabled,
        enable_marker=config.processing.marker_enabled,
        opendataloader_mode=config.processing.opendataloader_mode,
        opendataloader_hybrid=hybrid_config,
        opendataloader_hybrid_fallback=config.processing.opendataloader_hybrid_fallback,
    )
    result = cascade.extract_text(
        Path(source_path),
        doi=paper.doi,
        url=paper.url,
    )
    if not result.text:
        return None
    if result.is_markdown:
        return result.text
    return text_cleaner.clean(result.text)


def _ensure_index_profile_compatible(
    *,
    index_dir: Path,
    profile: DimensionProfile,
    paper_ids: list[str] | None = None,
) -> None:
    """Reject incompatible writes into one index.

    When ``paper_ids`` is provided, only those extraction records are checked.
    This allows targeted retries during a staged profile backfill where the
    index as a whole is intentionally mixed until the full corpus completes.
    """

    store = StructuredStore(index_dir)
    existing = store.load_extractions()
    mismatched_ids: list[str] = []
    relevant_ids = set(paper_ids) if paper_ids else None
    for paper_id, record in existing.items():
        if relevant_ids is not None and paper_id not in relevant_ids:
            continue
        try:
            extraction = DimensionedExtraction.from_record(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot validate existing semantic extraction for {paper_id!r}: {exc}"
            ) from exc
        if extraction.profile_fingerprint and extraction.profile_fingerprint != profile.fingerprint:
            mismatched_ids.append(paper_id)
            if len(mismatched_ids) >= 5:
                break
    if mismatched_ids:
        sample = ", ".join(mismatched_ids[:5])
        raise ValueError(
            "Index contains semantic analyses from a different dimension profile. "
            f"Target profile: {profile.profile_id}@{profile.fingerprint[:12]}; "
            f"mismatched papers include: {sample}"
        )


def _paper_from_index_dict(paper_id: str, paper_dict: dict[str, Any]) -> PaperMetadata:
    """Reconstruct PaperMetadata from one papers.json record."""

    authors = []
    for author_data in paper_dict.get("authors", []):
        if isinstance(author_data, dict):
            authors.append(Author(**author_data))
    return PaperMetadata(
        paper_id=paper_id,
        zotero_key=paper_dict.get("zotero_key", paper_id.split("_")[0]),
        zotero_item_id=paper_dict.get("zotero_item_id", 0),
        item_type=paper_dict.get("item_type", "journalArticle"),
        title=paper_dict.get("title", "Unknown"),
        authors=authors,
        publication_year=paper_dict.get("publication_year"),
        publication_date=paper_dict.get("publication_date"),
        journal=paper_dict.get("journal"),
        doi=paper_dict.get("doi"),
        abstract=paper_dict.get("abstract"),
        url=paper_dict.get("url"),
        collections=paper_dict.get("collections", []),
        tags=paper_dict.get("tags", []),
        pdf_path=paper_dict.get("pdf_path"),
        pdf_attachment_key=paper_dict.get("pdf_attachment_key"),
        source_path=paper_dict.get("source_path"),
        source_attachment_key=paper_dict.get("source_attachment_key"),
        source_media_type=paper_dict.get("source_media_type"),
        date_added=_coerce_index_datetime(paper_dict.get("date_added")),
        date_modified=_coerce_index_datetime(paper_dict.get("date_modified")),
        indexed_at=paper_dict.get("indexed_at"),
    )


def _coerce_index_datetime(value: Any) -> datetime:
    """Parse timestamp fields from papers.json into datetimes."""

    if isinstance(value, datetime):
        return value

    raw_value = str(value or _DEFAULT_INDEX_TIMESTAMP).strip()
    normalized = raw_value.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.fromisoformat(_DEFAULT_INDEX_TIMESTAMP)
