#!/usr/bin/env python
"""Manage portable semantic dimension profiles and storage."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.dimensions import (
    DimensionDefinition,
    DimensionProfile,
    DimensionRegistry,
    get_default_dimension_registry,
    get_dimension_map,
    load_dimension_profile,
)
from src.analysis.coverage import score_coverage
from src.analysis.llm_factory import create_llm_client
from src.analysis.schemas import DimensionedExtraction, SemanticAnalysis
from src.config import Config
from src.indexing.orchestrator import (
    EXTRACTION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    IndexManifest,
    fingerprint_payload,
)
from src.indexing.pipeline import (
    build_section_extractor,
    compute_similarity_pairs,
    configure_extraction_runtime,
    generate_summary,
    load_reusable_text_snapshots,
    run_embedding_generation,
)
from src.indexing.raptor_pipeline import generate_scoped_raptor_summaries
from src.indexing.structured_store import StructuredStore
from src.utils.checkpoint import CheckpointManager
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging
from src.zotero.models import Author, PaperMetadata

PROPOSALS_FILENAME = "dimension_proposals.json"
MIGRATED_EXTRACTION_SCHEMA_VERSION = "2.0.0"
BACKFILL_CHECKPOINT_ROOT = ".dimension_backfill"
BACKFILL_STAGED_EXTRACTIONS_FILENAME = "semantic_analyses.backfill.json"

SUGGESTION_HEURISTICS = [
    {
        "dimension_id": "stakeholders",
        "label": "Stakeholders",
        "question": "Which stakeholders, communities, or user groups are centered or affected?",
        "keywords": ["stakeholder", "community", "user", "participant", "actor"],
        "section": "impact",
        "roles": ["audience"],
        "rationale": "The corpus frequently names stakeholder groups that are not isolated in a dedicated dimension.",
    },
    {
        "dimension_id": "geographic_scope",
        "label": "Geographic Scope",
        "question": "What geographic, regional, or place-based scope shapes the claims and evidence?",
        "keywords": ["country", "region", "geographic", "local", "global", "city"],
        "section": "scholarly",
        "roles": [],
        "rationale": "The corpus repeatedly signals geographic framing that could help compare otherwise distant topics.",
    },
    {
        "dimension_id": "evaluation_metrics",
        "label": "Evaluation Metrics",
        "question": "What metrics, benchmarks, or evaluation criteria are used to judge success?",
        "keywords": ["benchmark", "metric", "accuracy", "precision", "recall", "f1", "evaluation"],
        "section": "methodology",
        "roles": [],
        "rationale": "Many papers discuss explicit metrics, but the active profile may not expose them as a reusable axis.",
    },
    {
        "dimension_id": "operational_constraints",
        "label": "Operational Constraints",
        "question": "What operational, implementation, or resource constraints shape the work?",
        "keywords": ["constraint", "deployment", "implementation", "resource", "cost", "budget"],
        "section": "impact",
        "roles": [],
        "rationale": "Operational limits appear often enough to warrant a dedicated comparison dimension.",
    },
    {
        "dimension_id": "risk_profile",
        "label": "Risk Profile",
        "question": "What concrete risks, harms, or failure modes are identified or implied?",
        "keywords": ["risk", "harm", "failure", "hazard", "unsafe", "safety"],
        "section": "impact",
        "roles": [],
        "rationale": "Risk language is common across the sampled corpus and may bridge technical and policy-oriented work.",
    },
]

SUGGESTION_SYSTEM_PROMPT = """\
You are designing new semantic analysis dimensions for a literature review index.
Propose dimensions that are genuinely missing from the active profile, recur across
multiple papers, and improve comparison or bridge discovery across the corpus.

Rules:
1. Do not rename, duplicate, or trivially rephrase existing dimensions.
2. Prefer portable analytic axes that can be extracted from many papers.
3. Ground proposals in the provided corpus evidence and similarity-neighbor context.
4. Use stable snake_case dimension IDs.
5. Return ONLY valid JSON matching the requested schema."""


def _strip_json_fence(text: str) -> str:
    """Remove common markdown fencing before JSON parsing."""

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _slugify_dimension_id(value: str) -> str:
    """Normalize candidate dimension IDs into stable snake_case."""

    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _truncate_text(value: str | None, max_chars: int = 480) -> str:
    """Compact long evidence text for prompts and proposal payloads."""

    if not value:
        return ""
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _resolve_llm_runtime(
    args: argparse.Namespace,
    config: Config,
) -> tuple[str, str, str]:
    """Resolve provider, mode, and model for semantic suggestion generation."""

    provider = getattr(args, "provider", None) or config.extraction.provider
    provider_settings = config.extraction.get_provider_settings(provider)
    mode = (
        getattr(args, "mode", None)
        or provider_settings.mode
        or config.extraction.mode
    )
    model = (
        getattr(args, "model", None)
        or provider_settings.model
        or config.extraction.model
    )
    return provider, mode, model


def _call_raw_llm_prompt(
    prompt: str,
    provider: str,
    mode: str,
    model: str,
) -> str:
    """Execute a raw prompt against an LLM client and return response text."""

    client = create_llm_client(
        provider=provider,
        mode=mode,
        model=model,
        max_tokens=8192,
        timeout=180,
        reasoning_effort="high",
    )

    if mode == "api" and hasattr(client, "_call_api"):
        response = client._call_api(prompt)
        if isinstance(response, tuple):
            return str(response[0])
        return str(response)

    if mode == "cli" and hasattr(client, "_call_cli"):
        response = client._call_cli(prompt)
        if isinstance(response, tuple):
            return str(response[0])
        return str(response)

    if hasattr(client, "_call_llm"):
        return str(client._call_llm(prompt))

    response_text, success, error = client.raw_query(prompt)
    if not success or not response_text:
        raise RuntimeError(error or "LLM returned no response text")
    return response_text


def _sample_suggestion_ids(
    papers: dict[str, dict],
    extractions: dict[str, dict],
    similarity_pairs: dict[str, list[dict]],
    sample_size: int,
    requested_ids: list[str] | None = None,
) -> list[str]:
    """Choose high-signal papers for dimension suggestion analysis."""

    if requested_ids:
        return [paper_id for paper_id in requested_ids if paper_id in papers and paper_id in extractions]

    scored: list[tuple[float, float, float, str]] = []
    for paper_id, record in extractions.items():
        if paper_id not in papers:
            continue
        extraction = record.get("extraction", record) if isinstance(record, dict) else {}
        dims = get_dimension_map(record)
        bridge_score = sum(
            1.0
            for key in [
                "disciplines_bridged",
                "cross_domain_insights",
                "policy_recommendations",
                "power_dynamics",
                "stp_cas_linkage",
            ]
            if dims.get(key)
        )
        coverage = float(extraction.get("dimension_coverage", 0.0) or 0.0)
        neighbor_score = float(len(similarity_pairs.get(paper_id, [])))
        scored.append((bridge_score, coverage, neighbor_score, paper_id))

    if not scored:
        return []

    ordered = [paper_id for *_scores, paper_id in sorted(scored, reverse=True)]
    return ordered[:sample_size]


def _build_paper_briefs(
    paper_ids: list[str],
    papers: dict[str, dict],
    extractions: dict[str, dict],
) -> list[dict[str, Any]]:
    """Build compact paper summaries for suggestion prompts and outputs."""

    briefs: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        paper = papers.get(paper_id, {})
        dims = get_dimension_map(extractions.get(paper_id, {}))
        briefs.append(
            {
                "paper_id": paper_id,
                "title": paper.get("title", "Unknown"),
                "year": paper.get("publication_year"),
                "abstract": _truncate_text(paper.get("abstract"), max_chars=700),
                "signals": {
                    key: _truncate_text(dims.get(key))
                    for key in [
                        "field",
                        "disciplines_bridged",
                        "cross_domain_insights",
                        "deployment_gap",
                        "power_dynamics",
                        "policy_recommendations",
                        "stp_cas_linkage",
                        "intervention_leverage_point",
                    ]
                    if dims.get(key)
                },
            }
        )
    return briefs


def _build_similarity_context(
    paper_ids: list[str],
    papers: dict[str, dict],
    extractions: dict[str, dict],
    similarity_pairs: dict[str, list[dict]],
    neighbor_count: int,
    max_pairs: int,
) -> list[dict[str, Any]]:
    """Build vectorstore-derived bridge examples from similarity neighbors."""

    contexts: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        source_paper = papers.get(paper_id, {})
        source_dims = get_dimension_map(extractions.get(paper_id, {}))
        source_field = source_dims.get("field")
        source_bridge = source_dims.get("disciplines_bridged") or source_dims.get("stp_cas_linkage")
        for neighbor in similarity_pairs.get(paper_id, [])[:neighbor_count]:
            neighbor_id = neighbor.get("similar_paper_id")
            if not neighbor_id or neighbor_id not in papers:
                continue
            neighbor_dims = get_dimension_map(extractions.get(neighbor_id, {}))
            neighbor_field = neighbor_dims.get("field")
            contexts.append(
                {
                    "source_paper_id": paper_id,
                    "source_title": source_paper.get("title", "Unknown"),
                    "source_field": source_field,
                    "source_bridge_signal": _truncate_text(source_bridge, max_chars=220),
                    "neighbor_paper_id": neighbor_id,
                    "neighbor_title": papers[neighbor_id].get("title", "Unknown"),
                    "neighbor_field": neighbor_field,
                    "neighbor_bridge_signal": _truncate_text(
                        neighbor_dims.get("disciplines_bridged")
                        or neighbor_dims.get("stp_cas_linkage"),
                        max_chars=220,
                    ),
                    "similarity_score": neighbor.get("similarity_score"),
                }
            )
    ranked = sorted(
        contexts,
        key=lambda item: (
            item["source_field"] != item["neighbor_field"],
            item.get("similarity_score", 0.0) or 0.0,
        ),
        reverse=True,
    )
    return ranked[:max_pairs]


def _build_suggestion_prompt(
    profile: DimensionProfile,
    paper_briefs: list[dict[str, Any]],
    similarity_context: list[dict[str, Any]],
    max_proposals: int,
) -> str:
    """Compose the semantic suggestion prompt for LLM generation."""

    existing_dimensions = "\n".join(
        f"- {dimension.id} [{dimension.section}]: {dimension.question}"
        for dimension in profile.ordered_dimensions
    )
    sample_payload = json.dumps(
        {
            "paper_samples": paper_briefs,
            "similarity_bridges": similarity_context,
        },
        indent=2,
        ensure_ascii=False,
    )
    sections = ", ".join(section.id for section in profile.ordered_sections)

    return f"""{SUGGESTION_SYSTEM_PROMPT}

ACTIVE PROFILE:
- profile_id: {profile.profile_id}
- version: {profile.version}
- title: {profile.title}
- available_sections: {sections}

EXISTING DIMENSIONS:
{existing_dimensions}

CORPUS EVIDENCE:
{sample_payload}

Return a JSON object in this exact shape:
{{
  "proposals": [
    {{
      "dimension_id": "snake_case_id",
      "label": "Readable Label",
      "question": "Extraction question phrased for analysts",
      "rationale": "Why this recurring axis adds value beyond the existing profile",
      "suggested_section": "one of the available section ids",
      "suggested_roles": ["optional_role_alias"],
      "confidence": 0.0,
      "example_papers": ["paper_id_1", "paper_id_2"],
      "bridge_value": "How this helps compare or bridge disparate papers"
    }}
  ]
}}

Constraints:
- Propose at most {max_proposals} dimensions.
- Prefer dimensions that recur across multiple sampled papers or similarity bridges.
- Do not repeat ids already present in the active profile.
- Confidence must be between 0 and 1."""


def _normalize_example_papers(
    raw_examples: Any,
    papers: dict[str, dict],
) -> list[dict[str, str]]:
    """Normalize proposal example-paper references into structured entries."""

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    candidates = raw_examples if isinstance(raw_examples, list) else []
    for candidate in candidates:
        if isinstance(candidate, dict):
            paper_id = str(candidate.get("paper_id") or "").strip()
            title = str(candidate.get("title") or papers.get(paper_id, {}).get("title") or "Unknown")
        else:
            paper_id = str(candidate).strip()
            title = str(papers.get(paper_id, {}).get("title") or "Unknown")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        normalized.append({"paper_id": paper_id, "title": title})
    return normalized


def _normalize_llm_proposals(
    raw_payload: dict[str, Any],
    profile: DimensionProfile,
    papers: dict[str, dict],
) -> list[dict[str, Any]]:
    """Validate and normalize semantic LLM proposals into the CLI payload shape."""

    existing_ids = set(profile.dimension_map)
    section_ids = {section.id for section in profile.sections}
    raw_proposals = raw_payload.get("proposals", [])
    if not isinstance(raw_proposals, list):
        raise ValueError("LLM suggestion payload is missing a proposals list")

    normalized: list[dict[str, Any]] = []
    for proposal in raw_proposals:
        if not isinstance(proposal, dict):
            continue
        dimension_id = _slugify_dimension_id(
            str(proposal.get("dimension_id") or proposal.get("label") or "")
        )
        if not dimension_id or dimension_id in existing_ids:
            continue
        question = str(proposal.get("question") or "").strip()
        label = str(proposal.get("label") or dimension_id.replace("_", " ").title()).strip()
        if not question:
            continue

        suggested_section = str(proposal.get("suggested_section") or "").strip()
        if suggested_section not in section_ids:
            suggested_section = profile.ordered_sections[-1].id

        try:
            confidence = float(proposal.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        normalized.append(
            {
                "dimension_id": dimension_id,
                "label": label,
                "question": question,
                "rationale": str(proposal.get("rationale") or "").strip(),
                "suggested_section": suggested_section,
                "suggested_roles": [
                    str(role).strip()
                    for role in proposal.get("suggested_roles", [])
                    if str(role).strip()
                ],
                "confidence": max(0.0, min(1.0, confidence)),
                "example_papers": _normalize_example_papers(
                    proposal.get("example_papers", []),
                    papers,
                ),
                "bridge_value": str(proposal.get("bridge_value") or "").strip(),
                "proposal_sources": ["semantic_llm"],
            }
        )
    return normalized


def _merge_dimension_proposals(*proposal_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge proposal lists by dimension id while preserving evidence."""

    merged: dict[str, dict[str, Any]] = {}
    for proposals in proposal_lists:
        for proposal in proposals:
            proposal_id = proposal["dimension_id"]
            if proposal_id not in merged:
                merged[proposal_id] = {
                    **proposal,
                    "proposal_sources": list(proposal.get("proposal_sources", [])),
                    "example_papers": list(proposal.get("example_papers", [])),
                }
                continue

            current = merged[proposal_id]
            current["confidence"] = max(
                float(current.get("confidence", 0.0)),
                float(proposal.get("confidence", 0.0)),
            )
            if not current.get("rationale") and proposal.get("rationale"):
                current["rationale"] = proposal["rationale"]
            if not current.get("bridge_value") and proposal.get("bridge_value"):
                current["bridge_value"] = proposal["bridge_value"]
            if len(proposal.get("question", "")) > len(current.get("question", "")):
                current["question"] = proposal["question"]
            if len(proposal.get("label", "")) > len(current.get("label", "")):
                current["label"] = proposal["label"]

            existing_sources = set(current.get("proposal_sources", []))
            for source in proposal.get("proposal_sources", []):
                if source not in existing_sources:
                    current.setdefault("proposal_sources", []).append(source)
                    existing_sources.add(source)

            seen_examples = {item["paper_id"] for item in current.get("example_papers", [])}
            for example in proposal.get("example_papers", []):
                if example["paper_id"] not in seen_examples:
                    current.setdefault("example_papers", []).append(example)
                    seen_examples.add(example["paper_id"])

    return sorted(
        merged.values(),
        key=lambda proposal: (
            "semantic_llm" in proposal.get("proposal_sources", []),
            float(proposal.get("confidence", 0.0)),
            len(proposal.get("example_papers", [])),
            proposal["dimension_id"],
        ),
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Portable semantic dimension profile tooling",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser(
        "migrate-store",
        help="Migrate semantic_analyses.json to canonical map-based storage",
    )
    migrate.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
    )
    migrate.add_argument("--dry-run", action="store_true")
    migrate.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional explicit backup directory",
    )

    diff = subparsers.add_parser(
        "diff",
        help="Compare two dimension profiles",
    )
    diff.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
        help="Index directory used when --old-profile is omitted",
    )
    diff.add_argument(
        "--old-profile",
        type=str,
        default=None,
        help="Profile id or path. Defaults to the index snapshot profile.",
    )
    diff.add_argument(
        "--new-profile",
        type=str,
        required=True,
        help="Profile id or path to compare against",
    )
    diff.add_argument("--output", type=Path, default=None)
    diff.add_argument("--json", action="store_true", dest="json_output")

    suggest = subparsers.add_parser(
        "suggest",
        help="Generate heuristic dimension proposals from the corpus",
    )
    suggest.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
    )
    suggest.add_argument(
        "--paper",
        action="append",
        default=[],
        help="Restrict suggestion generation to these paper_ids (repeatable)",
    )
    suggest.add_argument("--sample-size", type=int, default=None)
    suggest.add_argument("--min-hits", type=int, default=3)
    suggest.add_argument("--max-proposals", type=int, default=None)
    suggest.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google", "ollama", "llamacpp"],
        default=None,
    )
    suggest.add_argument("--mode", choices=["api", "cli"], default=None)
    suggest.add_argument("--model", type=str, default=None)
    suggest.add_argument("--heuristic-only", action="store_true")
    suggest.add_argument("--output", type=Path, default=None)

    approve = subparsers.add_parser(
        "approve",
        help="Apply approved proposals to a profile file",
    )
    approve.add_argument("--profile", type=Path, required=True)
    approve.add_argument("--proposals", type=Path, required=True)
    approve.add_argument(
        "--dimension-id",
        action="append",
        default=[],
        help="Proposal dimension id to approve (repeatable)",
    )
    approve.add_argument("--all", action="store_true")
    approve.add_argument("--output", type=Path, default=None)

    backfill = subparsers.add_parser(
        "backfill",
        help="Backfill an index against a new dimension profile",
    )
    backfill.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
    )
    backfill.add_argument(
        "--dimension-profile",
        type=Path,
        required=True,
        help="Path to the target profile YAML/JSON",
    )
    backfill.add_argument(
        "--paper",
        action="append",
        default=[],
        help="Restrict backfill to these paper_ids (repeatable)",
    )
    backfill.add_argument("--dry-run", action="store_true")
    backfill.add_argument("--skip-embeddings", action="store_true")
    backfill.add_argument("--skip-similarity", action="store_true")
    backfill.add_argument("--provider", choices=["anthropic", "openai", "google"], default=None)
    backfill.add_argument("--mode", choices=["api", "cli"], default=None)
    backfill.add_argument("--model", type=str, default=None)
    backfill.add_argument("--parallel", type=int, default=None)
    backfill.add_argument("--no-cache", action="store_true")
    backfill.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted backfill for the same target profile and paper scope",
    )
    backfill.add_argument(
        "--fulltext-only",
        action="store_true",
        help="Capture canonical full-text snapshots without calling the semantic analysis provider",
    )
    backfill.add_argument(
        "--refresh-text",
        action="store_true",
        help="Re-run source text extraction for the targeted papers instead of reusing stored full-text snapshots",
    )

    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    if getattr(args, "sample_size", None):
        config.dimensions.suggestion_sample_size = args.sample_size
    return config


def _apply_dimension_profile_override(config: Config, profile_path: Path) -> DimensionProfile:
    profile = load_dimension_profile(profile_path)
    existing_paths = [Path(path) for path in config.dimensions.profile_paths]
    if profile_path not in existing_paths:
        config.dimensions.profile_paths = [*existing_paths, profile_path]
    config.dimensions.active_profile = profile.profile_id
    config.configure_dimension_registry()
    return profile


def _paper_from_index_dict(paper_id: str, paper_dict: dict[str, Any]) -> PaperMetadata:
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
        date_added=paper_dict.get("date_added") or "2020-01-01T00:00:00",
        date_modified=paper_dict.get("date_modified") or "2020-01-01T00:00:00",
    )


def _resolve_profile_reference(reference: str | None, index_dir: Path) -> DimensionProfile:
    if reference is None:
        store = StructuredStore(index_dir)
        return DimensionProfile(**store.load_dimension_profile())

    candidate = Path(reference)
    if candidate.exists():
        return load_dimension_profile(candidate)

    registry = get_default_dimension_registry()
    return registry.get_profile(reference)


def _bump_patch_version(version: str) -> str:
    parts = version.split(".")
    while len(parts) < 3:
        parts.append("0")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
    except ValueError:
        parts.append("1")
    return ".".join(parts[:3])


def _make_backup(path: Path, backup_dir: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = backup_dir or path.parent / f"backup_{timestamp}"
    target_dir.mkdir(parents=True, exist_ok=True)
    backup_path = target_dir / path.name
    shutil.copy2(path, backup_path)
    return backup_path


def _load_extractions_payload(index_dir: Path) -> tuple[dict[str, Any], dict[str, dict]]:
    raw_data = safe_read_json(index_dir / "semantic_analyses.json", default={})
    if isinstance(raw_data, dict) and "extractions" in raw_data:
        return raw_data, raw_data.get("extractions", {})
    if isinstance(raw_data, dict):
        return {}, raw_data
    raise ValueError("Unsupported semantic_analyses.json structure")


def _backfill_checkpoint_dir(index_dir: Path, profile: DimensionProfile) -> Path:
    """Return the checkpoint directory for a staged backfill run."""

    fingerprint = profile.fingerprint[:12]
    return index_dir / BACKFILL_CHECKPOINT_ROOT / f"{profile.profile_id}_{fingerprint}"


def _staged_backfill_extractions_path(checkpoint_dir: Path) -> Path:
    """Return the staged extraction payload path for a backfill checkpoint."""

    return checkpoint_dir / BACKFILL_STAGED_EXTRACTIONS_FILENAME


def _load_staged_backfill_extractions(checkpoint_dir: Path) -> dict[str, dict]:
    """Load staged extraction updates from a backfill checkpoint."""

    return safe_read_json(
        _staged_backfill_extractions_path(checkpoint_dir),
        default={},
    )


def _save_staged_backfill_extractions(
    checkpoint_dir: Path,
    staged_extractions: dict[str, dict],
) -> None:
    """Persist staged extraction updates for a resumable backfill."""

    safe_write_json(
        _staged_backfill_extractions_path(checkpoint_dir),
        staged_extractions,
    )


def _checkpoint_request_metadata(
    *,
    target_ids: list[str],
    old_profile: DimensionProfile,
    target_profile: DimensionProfile,
    changed_sections: list[str],
    reextract_sections: list[str],
    partial_scope: bool,
    refresh_text: bool,
    fulltext_only: bool,
) -> dict[str, Any]:
    """Build the request fingerprint stored alongside backfill checkpoints."""

    return {
        "paper_ids": list(target_ids),
        "old_profile_id": old_profile.profile_id,
        "old_profile_fingerprint": old_profile.fingerprint,
        "target_profile_id": target_profile.profile_id,
        "target_profile_fingerprint": target_profile.fingerprint,
        "changed_sections": list(changed_sections),
        "reextract_sections": list(reextract_sections),
        "partial_scope": partial_scope,
        "refresh_text": refresh_text,
        "fulltext_only": fulltext_only,
    }


def _checkpoint_matches_request(
    state_metadata: dict[str, Any],
    request_metadata: dict[str, Any],
) -> bool:
    """Return whether a stored checkpoint matches the current backfill request."""

    comparable_keys = [
        "paper_ids",
        "old_profile_id",
        "old_profile_fingerprint",
        "target_profile_id",
        "target_profile_fingerprint",
        "changed_sections",
        "reextract_sections",
        "partial_scope",
        "refresh_text",
        "fulltext_only",
    ]
    return all(
        state_metadata.get(key) == request_metadata.get(key)
        for key in comparable_keys
    )


def _section_dimension_ids(
    profile: DimensionProfile,
    section_ids: list[str],
) -> list[str]:
    """Return ordered dimension IDs covered by the selected sections."""

    selected = set(section_ids)
    return [
        dimension.id
        for section in profile.ordered_sections
        if section.id in selected
        for dimension in profile.dimensions_for_section(section.id)
    ]


def _retarget_extraction(
    extraction: DimensionedExtraction,
    target_profile: DimensionProfile,
    dimensions: dict[str, str | None] | None = None,
) -> DimensionedExtraction:
    """Retarget an extraction to a profile and recompute profile-driven coverage."""

    retargeted = DimensionedExtraction(
        paper_id=extraction.paper_id,
        profile_id=target_profile.profile_id,
        profile_version=target_profile.version,
        profile_fingerprint=target_profile.fingerprint,
        prompt_version=extraction.prompt_version,
        extraction_model=extraction.extraction_model,
        extracted_at=extraction.extracted_at,
        dimensions=dict(dimensions or extraction.dimensions),
        dimension_coverage=extraction.dimension_coverage,
        coverage_flags=list(extraction.coverage_flags),
    )
    coverage = score_coverage(SemanticAnalysis(**retargeted.to_index_dict()))
    retargeted.dimension_coverage = coverage.coverage
    retargeted.coverage_flags = list(coverage.flags)
    return retargeted


def _merge_reextracted_dimensions(
    old_extraction: DimensionedExtraction,
    new_extraction: DimensionedExtraction,
    target_profile: DimensionProfile,
    section_ids: list[str],
) -> DimensionedExtraction:
    """Merge only the dimensions touched by a scoped re-extraction."""

    targeted_dimension_ids = _section_dimension_ids(target_profile, section_ids)
    merged_dimensions = dict(old_extraction.dimensions)
    for dimension_id in targeted_dimension_ids:
        merged_dimensions[dimension_id] = new_extraction.dimensions.get(dimension_id)
    return _retarget_extraction(
        old_extraction,
        target_profile=target_profile,
        dimensions=merged_dimensions,
    )


def _extraction_manifest_info(
    args: argparse.Namespace,
    config: Config,
    profile: DimensionProfile,
) -> dict[str, Any]:
    """Build manifest extraction metadata for a profile backfill run."""

    provider = getattr(args, "provider", None) or config.extraction.provider
    provider_settings = config.extraction.get_provider_settings(provider)
    payload = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "provider": provider,
        "mode": getattr(args, "mode", None)
        or provider_settings.mode
        or config.extraction.mode,
        "model": getattr(args, "model", None)
        or provider_settings.model
        or config.extraction.model,
        "summary_model": getattr(args, "summary_model", None),
        "methodology_model": getattr(args, "methodology_model", None),
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_fingerprint": profile.fingerprint,
    }
    return {**payload, "fingerprint": fingerprint_payload(payload)}


def _update_manifest_profile_snapshot(
    index_dir: Path,
    args: argparse.Namespace,
    config: Config,
    profile: DimensionProfile,
) -> None:
    """Update manifest extraction metadata after a successful backfill."""

    manifest_path = index_dir / MANIFEST_FILENAME
    manifest, _error = IndexManifest.load(manifest_path)
    if manifest is None:
        return

    manifest.extraction = _extraction_manifest_info(args, config, profile)
    manifest.last_run = {
        **manifest.last_run,
        "dimension_backfill": {
            "completed_at": datetime.now().isoformat(),
            "profile_id": profile.profile_id,
            "profile_version": profile.version,
            "profile_fingerprint": profile.fingerprint,
        },
    }
    manifest.save(manifest_path)


def migrate_store(args: argparse.Namespace, logger) -> int:
    """Rewrite extraction storage into canonical dimension-map form."""

    store = StructuredStore(args.index_dir)
    extractions_file = args.index_dir / "semantic_analyses.json"
    if not extractions_file.exists():
        logger.error("Missing extraction store: %s", extractions_file)
        return 1

    wrapper, extractions = _load_extractions_payload(args.index_dir)
    profile_snapshot = store.load_dimension_profile()
    profile = DimensionProfile(**profile_snapshot)
    registry = DimensionRegistry()
    registry.register_profile(profile)
    registry.set_active_profile(profile.profile_id)

    migrated: dict[str, dict] = {}
    changed = 0
    for paper_id, record in extractions.items():
        canonical = DimensionedExtraction.from_record(record)
        wrapped = dict(record) if isinstance(record, dict) else {"paper_id": paper_id}
        wrapped["paper_id"] = paper_id
        wrapped["extraction"] = canonical.to_index_dict()
        migrated[paper_id] = wrapped
        original_extraction = record.get("extraction", record) if isinstance(record, dict) else {}
        if not isinstance(original_extraction, dict) or "dimensions" not in original_extraction:
            changed += 1

    logger.info("Prepared %d extraction records; %d require migration", len(migrated), changed)
    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "index_dir": str(args.index_dir),
                    "paper_count": len(migrated),
                    "records_requiring_migration": changed,
                    "profile_id": profile.profile_id,
                },
                sort_keys=False,
                allow_unicode=False,
            ).strip()
        )
        return 0

    backup_path = _make_backup(extractions_file, args.backup_dir)
    logger.info("Backed up extraction store to %s", backup_path)

    output_wrapper = dict(wrapper) if wrapper else {}
    output_wrapper["schema_version"] = MIGRATED_EXTRACTION_SCHEMA_VERSION
    output_wrapper["generated_at"] = datetime.now().isoformat()
    output_wrapper["extraction_count"] = len(migrated)
    output_wrapper["extractions"] = migrated

    safe_write_json(extractions_file, output_wrapper)
    store.save_dimension_profile(profile.model_dump(mode="json"))
    logger.info("Migrated %d extraction records", len(migrated))
    return 0


def diff_profiles(args: argparse.Namespace) -> int:
    """Print or save a diff between two profiles."""

    old_profile = _resolve_profile_reference(args.old_profile, args.index_dir)
    new_profile = _resolve_profile_reference(args.new_profile, args.index_dir)
    registry = DimensionRegistry()
    registry.register_profile(old_profile)
    registry.register_profile(new_profile)
    diff = registry.diff_profiles(old_profile.profile_id, new_profile.profile_id)

    payload = diff.model_dump(mode="json")
    if args.output:
        safe_write_json(args.output, payload)
        print(args.output)
        return 0

    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Profile diff: {old_profile.profile_id} -> {new_profile.profile_id}")
    if not diff.entries:
        print("No changes")
        return 0
    for entry in diff.entries:
        label = entry.dimension_id
        if entry.status == "added":
            print(f"ADDED     {label} -> {entry.new_section}")
        elif entry.status == "disabled":
            print(f"DISABLED  {label} (was {entry.old_section})")
        elif entry.status == "reworded":
            print(f"REWORDED  {label} ({entry.old_section} -> {entry.new_section})")
        else:
            replaced = ", ".join(entry.replaces or [])
            print(f"REPLACED  {label} <- {replaced or 'n/a'}")
    return 0


def suggest_dimensions(args: argparse.Namespace, config: Config) -> int:
    """Generate profile extension proposals from corpus evidence and heuristics."""

    store = StructuredStore(args.index_dir)
    papers = store.load_papers()
    extractions = store.load_extractions()
    profile = DimensionProfile(**store.load_dimension_profile())
    similarity_pairs = store.load_similarity_pairs()

    sample_size = args.sample_size or config.dimensions.suggestion_sample_size
    max_proposals = args.max_proposals or config.dimensions.suggestion_max_proposals
    neighbor_count = config.dimensions.suggestion_neighbor_count
    sampled_ids = _sample_suggestion_ids(
        papers=papers,
        extractions=extractions,
        similarity_pairs=similarity_pairs,
        sample_size=sample_size,
        requested_ids=args.paper,
    )
    existing_ids = set(profile.dimension_map)
    section_ids = {section.id for section in profile.sections}
    heuristic_proposals: list[dict[str, Any]] = []

    for heuristic in SUGGESTION_HEURISTICS:
        if heuristic["dimension_id"] in existing_ids:
            continue
        hits: list[dict[str, str]] = []
        keywords = heuristic["keywords"]
        for paper_id in sampled_ids:
            extraction_record = extractions.get(paper_id, {})
            paper = papers.get(paper_id, {})
            search_text = " ".join(
                part
                for part in [
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    " ".join(value for value in get_dimension_map(extraction_record).values() if value),
                ]
                if part
            ).lower()
            if any(keyword in search_text for keyword in keywords):
                hits.append(
                    {
                        "paper_id": paper_id,
                        "title": paper.get("title", "Unknown"),
                    }
                )

        if len(hits) < args.min_hits:
            continue

        suggested_section = heuristic["section"]
        if suggested_section not in section_ids:
            suggested_section = profile.ordered_sections[-1].id

        heuristic_proposals.append(
            {
                "dimension_id": heuristic["dimension_id"],
                "label": heuristic["label"],
                "question": heuristic["question"],
                "rationale": heuristic["rationale"],
                "suggested_section": suggested_section,
                "suggested_roles": heuristic["roles"],
                "confidence": round(min(0.95, len(hits) / max(len(sampled_ids), 1)), 3),
                "keyword_hits": len(hits),
                "example_papers": hits[:5],
                "proposal_sources": ["heuristic"],
            }
        )

    semantic_proposals: list[dict[str, Any]] = []
    suggestion_runtime: dict[str, str | None] = {
        "provider": None,
        "mode": None,
        "model": None,
    }
    if (
        config.dimensions.suggestion_use_llm
        and not args.heuristic_only
        and sampled_ids
    ):
        provider, mode, model = _resolve_llm_runtime(args, config)
        suggestion_runtime = {"provider": provider, "mode": mode, "model": model}
        paper_briefs = _build_paper_briefs(sampled_ids[: min(len(sampled_ids), 8)], papers, extractions)
        similarity_context = _build_similarity_context(
            paper_ids=sampled_ids[: min(len(sampled_ids), 8)],
            papers=papers,
            extractions=extractions,
            similarity_pairs=similarity_pairs,
            neighbor_count=neighbor_count,
            max_pairs=max_proposals * 2,
        )
        prompt = _build_suggestion_prompt(
            profile=profile,
            paper_briefs=paper_briefs,
            similarity_context=similarity_context,
            max_proposals=max_proposals,
        )
        try:
            raw_response = _call_raw_llm_prompt(
                prompt=prompt,
                provider=provider,
                mode=mode,
                model=model,
            )
            semantic_payload = json.loads(_strip_json_fence(raw_response))
            semantic_proposals = _normalize_llm_proposals(
                raw_payload=semantic_payload,
                profile=profile,
                papers=papers,
            )
        except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
            print(
                f"Warning: semantic suggestion generation failed; "
                f"falling back to heuristic proposals only ({exc})",
                file=sys.stderr,
            )

    proposals = _merge_dimension_proposals(semantic_proposals, heuristic_proposals)
    proposals = proposals[:max_proposals]

    output_path = args.output or (args.index_dir / PROPOSALS_FILENAME)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "profile_id": profile.profile_id,
        "sample_size": len(sampled_ids),
        "sampled_paper_ids": sampled_ids,
        "proposal_count": len(proposals),
        "heuristic_candidate_count": len(heuristic_proposals),
        "semantic_candidate_count": len(semantic_proposals),
        "semantic_runtime": suggestion_runtime,
        "proposals": proposals,
    }
    safe_write_json(output_path, payload)
    print(output_path)
    return 0


def approve_dimensions(args: argparse.Namespace) -> int:
    """Append approved proposals to a profile and write a new profile file."""

    profile = load_dimension_profile(args.profile)
    proposals_payload = safe_read_json(args.proposals, default={})
    proposals = proposals_payload.get("proposals", [])
    if not isinstance(proposals, list):
        raise ValueError("Proposal file is missing a proposals list")

    if not args.all and not args.dimension_id:
        raise ValueError("Use --all or provide at least one --dimension-id")

    allowed_ids = None if args.all else set(args.dimension_id)
    selected = [
        proposal
        for proposal in proposals
        if allowed_ids is None or proposal.get("dimension_id") in allowed_ids
    ]
    if not selected:
        raise ValueError("No proposals matched the requested ids")

    next_order = max((dimension.order for dimension in profile.dimensions), default=0) + 1
    existing_ids = set(profile.dimension_map)
    for proposal in selected:
        dimension_id = proposal["dimension_id"]
        if dimension_id in existing_ids:
            continue
        profile.dimensions.append(
            DimensionDefinition(
                id=dimension_id,
                label=proposal["label"],
                question=proposal["question"],
                section=proposal["suggested_section"],
                order=next_order,
                aliases=[dimension_id],
                roles=proposal.get("suggested_roles", []),
            )
        )
        next_order += 1
        existing_ids.add(dimension_id)

    profile.version = _bump_patch_version(profile.version)
    output_path = args.output or args.profile.with_name(
        f"{args.profile.stem}.approved{args.profile.suffix or '.yaml'}"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            profile.model_dump(mode="json"),
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
    print(output_path)
    return 0


def backfill_dimensions(args: argparse.Namespace, config: Config, logger) -> int:
    """Backfill an index against a new profile by re-extracting targeted papers."""

    target_profile = _apply_dimension_profile_override(config, args.dimension_profile)
    store = StructuredStore(args.index_dir)
    old_profile = DimensionProfile(**store.load_dimension_profile())

    registry = DimensionRegistry()
    registry.register_profile(old_profile)
    registry.register_profile(target_profile)
    diff = registry.diff_profiles(old_profile.profile_id, target_profile.profile_id)
    changed_sections = sorted(
        {
            entry.new_section or entry.old_section
            for entry in diff.entries
            if entry.status in {"added", "reworded", "replaced", "disabled"}
            and (entry.new_section or entry.old_section)
        }
    )
    reextract_sections = sorted(
        {
            entry.new_section or entry.old_section
            for entry in diff.entries
            if entry.status in {"added", "reworded", "replaced"}
            and (entry.new_section or entry.old_section)
        }
    )
    disable_only = bool(diff.entries) and not reextract_sections

    papers = store.load_papers()
    extractions = store.load_extractions()
    target_ids = sorted(set(args.paper) if args.paper else set(extractions))
    partial_scope = bool(args.paper)
    missing_papers = [paper_id for paper_id in target_ids if paper_id not in papers]
    if missing_papers:
        logger.warning("Ignoring %d paper ids missing from papers.json", len(missing_papers))
        target_ids = [paper_id for paper_id in target_ids if paper_id in papers]

    request_metadata = _checkpoint_request_metadata(
        target_ids=target_ids,
        old_profile=old_profile,
        target_profile=target_profile,
        changed_sections=changed_sections,
        reextract_sections=reextract_sections,
        partial_scope=partial_scope,
        refresh_text=bool(getattr(args, "refresh_text", False)),
        fulltext_only=bool(getattr(args, "fulltext_only", False)),
    )
    checkpoint_dir = _backfill_checkpoint_dir(args.index_dir, target_profile)
    checkpoint_exists = checkpoint_dir.exists()

    if args.dry_run:
        payload = {
            "index_dir": str(args.index_dir),
            "paper_count": len(target_ids),
            "old_profile": old_profile.profile_id,
            "new_profile": target_profile.profile_id,
            "changed_sections": changed_sections,
            "reextract_sections": reextract_sections,
            "disable_only": disable_only,
            "refresh_text": bool(getattr(args, "refresh_text", False)),
            "fulltext_only": bool(getattr(args, "fulltext_only", False)),
            "resume": bool(getattr(args, "resume", False)),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_exists": checkpoint_exists,
            "diff": diff.model_dump(mode="json"),
        }
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).strip())
        return 0

    resume_requested = bool(getattr(args, "resume", False))
    fulltext_only = bool(getattr(args, "fulltext_only", False))

    if checkpoint_exists and not resume_requested:
        logger.info("Removing stale backfill checkpoint at %s", checkpoint_dir)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    checkpoint_mgr = CheckpointManager(checkpoint_dir, checkpoint_id="dimension_backfill")
    checkpoint_state = checkpoint_mgr.load() if resume_requested else None
    if checkpoint_state is not None:
        if not _checkpoint_matches_request(checkpoint_state.metadata, request_metadata):
            raise ValueError(
                "Existing backfill checkpoint does not match the current request. "
                "Delete the checkpoint or rerun without --resume."
            )
    else:
        checkpoint_mgr.initialize(
            total_items=len(target_ids),
            metadata=request_metadata,
        )

    staged_extractions = _load_staged_backfill_extractions(checkpoint_dir)
    updated_extractions = dict(extractions)
    updated_extractions.update(staged_extractions)
    requested_target_ids = list(target_ids)

    completed_ids = set(checkpoint_mgr.state.processed_ids if checkpoint_mgr.state else [])
    failed_checkpoint_ids = set(checkpoint_mgr.get_failed_ids()) if checkpoint_mgr.state else set()
    if completed_ids or failed_checkpoint_ids:
        logger.info(
            "Backfill checkpoint: %d completed, %d failed, %d remaining",
            len(completed_ids),
            len(failed_checkpoint_ids),
            max(0, len(target_ids) - len(completed_ids)),
        )

    target_ids = [paper_id for paper_id in target_ids if paper_id not in completed_ids]
    failed_ids: list[str] = []
    paper_models = [_paper_from_index_dict(paper_id, papers[paper_id]) for paper_id in target_ids]

    if reextract_sections or fulltext_only:
        provider, mode, cache_dir, parallel_workers, use_cache = configure_extraction_runtime(
            args,
            config,
            logger,
        )
        extractor = build_section_extractor(
            args=args,
            config=config,
            cache_dir=cache_dir,
            mode=mode,
            parallel_workers=parallel_workers,
            use_cache=use_cache,
        )
        reusable_text_snapshots = load_reusable_text_snapshots(
            store,
            paper_models,
            refresh_text=bool(getattr(args, "refresh_text", False)),
        )

        extract_batch_kwargs: dict[str, Any] = {}
        if not fulltext_only:
            extract_batch_kwargs["section_ids"] = reextract_sections
        signature = inspect.signature(extractor.extract_batch)
        if "text_snapshots" in signature.parameters:
            extract_batch_kwargs["text_snapshots"] = reusable_text_snapshots
        if "skip_llm" in signature.parameters:
            extract_batch_kwargs["skip_llm"] = fulltext_only

        processed_since_flush = 0
        try:
            for result in extractor.extract_batch(
                paper_models,
                **extract_batch_kwargs,
            ):
                processed_since_flush += 1
                snapshot = result.text_snapshot or {}
                snapshot_text = snapshot.get("text") if isinstance(snapshot, dict) else None
                if (
                    isinstance(snapshot_text, str)
                    and snapshot_text
                    and bool(snapshot.get("should_persist", True))
                ):
                    metadata = {
                        key: value
                        for key, value in snapshot.items()
                        if key not in {"text", "should_persist"}
                    }
                    store.save_text_snapshot(
                        result.paper_id,
                        snapshot_text,
                        metadata=metadata,
                        persist_manifest=False,
                    )

                if fulltext_only:
                    if result.success:
                        checkpoint_mgr.complete_item(result.paper_id, success=True)
                    else:
                        logger.warning(
                            "Full-text capture failed for %s: %s",
                            result.paper_id,
                            result.error,
                        )
                        failed_ids.append(result.paper_id)
                        checkpoint_mgr.complete_item(
                            result.paper_id,
                            success=False,
                            error=Exception(result.error or "Unknown error"),
                        )
                else:
                    if not result.success or not result.extraction:
                        logger.warning("Backfill failed for %s: %s", result.paper_id, result.error)
                        failed_ids.append(result.paper_id)
                        checkpoint_mgr.complete_item(
                            result.paper_id,
                            success=False,
                            error=Exception(result.error or "Unknown error"),
                        )
                    elif result.pass_errors:
                        logger.warning(
                            "Backfill failed for %s due to incomplete targeted section extraction: %s",
                            result.paper_id,
                            "; ".join(result.pass_errors),
                        )
                        failed_ids.append(result.paper_id)
                        checkpoint_mgr.complete_item(
                            result.paper_id,
                            success=False,
                            error=Exception("; ".join(result.pass_errors)),
                        )
                    else:
                        existing_record = updated_extractions.get(
                            result.paper_id,
                            {"paper_id": result.paper_id},
                        )
                        old_extraction = DimensionedExtraction.from_record(existing_record)
                        new_extraction = DimensionedExtraction.from_record(result.extraction)
                        merged = _merge_reextracted_dimensions(
                            old_extraction=old_extraction,
                            new_extraction=new_extraction,
                            target_profile=target_profile,
                            section_ids=reextract_sections,
                        )
                        updated_extractions[result.paper_id] = {
                            **existing_record,
                            "paper_id": result.paper_id,
                            "extraction": merged.to_index_dict(),
                            "timestamp": result.timestamp.isoformat(),
                            "model": result.model_used,
                            "duration": result.duration_seconds,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                        }
                        staged_extractions[result.paper_id] = updated_extractions[result.paper_id]
                        checkpoint_mgr.complete_item(result.paper_id, success=True)

                if processed_since_flush % 10 == 0:
                    _save_staged_backfill_extractions(checkpoint_dir, staged_extractions)
                    checkpoint_mgr.save()
                    store.flush_fulltext_manifest()
        except KeyboardInterrupt:
            checkpoint_mgr.save()
            _save_staged_backfill_extractions(checkpoint_dir, staged_extractions)
            store.flush_fulltext_manifest()
            logger.warning(
                "Backfill interrupted by user. Resume with --resume. Checkpoint saved at %s",
                checkpoint_dir,
            )
            return 130
        finally:
            checkpoint_mgr.save()
            _save_staged_backfill_extractions(checkpoint_dir, staged_extractions)
            store.flush_fulltext_manifest()

    if failed_ids:
        logger.error(
            "Aborting backfill; %d paper(s) failed. Resume with --resume. Checkpoint: %s",
            len(failed_ids),
            checkpoint_dir,
        )
        return 1

    if fulltext_only:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        logger.info(
            "Captured canonical full-text snapshots for %d paper(s)%s",
            len(requested_target_ids),
            " using stored snapshots where available"
            if not args.refresh_text
            else " with refreshed source extraction",
        )
        return 0

    for paper_id in requested_target_ids:
        existing_record = updated_extractions.get(paper_id)
        if not existing_record:
            continue
        extraction = DimensionedExtraction.from_record(existing_record)
        retargeted = _retarget_extraction(extraction, target_profile)
        updated_extractions[paper_id] = {
            **existing_record,
            "paper_id": paper_id,
            "extraction": retargeted.to_index_dict(),
        }

    if not args.skip_embeddings:
        scoped_papers = [
            _paper_from_index_dict(paper_id, papers[paper_id])
            for paper_id in requested_target_ids
            if paper_id in updated_extractions
        ]
        scoped_extractions = {
            paper_id: record
            for paper_id, record in updated_extractions.items()
            if paper_id in {paper.paper_id for paper in scoped_papers}
        }
        raptor_summaries = generate_scoped_raptor_summaries(
            papers=scoped_papers,
            extractions=scoped_extractions,
            index_dir=args.index_dir,
            mode="template",
            force=True,
        )
        run_embedding_generation(
            papers=scoped_papers,
            extractions=updated_extractions,
            index_dir=args.index_dir,
            embedding_model=config.embeddings.model,
            rebuild=False,
            logger=logger,
            embedding_backend=config.embeddings.backend,
            ollama_base_url=config.embeddings.ollama_base_url,
            query_prefix=config.embeddings.query_prefix,
            document_prefix=config.embeddings.document_prefix,
            embedding_batch_size=config.embeddings.batch_size,
            raptor_summaries=raptor_summaries,
            delete_paper_ids=[],
        )
        if not args.skip_similarity:
            compute_similarity_pairs(
                index_dir=args.index_dir,
                embedding_model=config.embeddings.model,
                top_n=20,
                logger=logger,
            )

    store.save_extractions(updated_extractions)
    if partial_scope:
        logger.info(
            "Partial backfill completed for %d paper(s); leaving the index-level "
            "profile snapshot unchanged until a full-corpus backfill succeeds",
            len(requested_target_ids),
        )
    else:
        store.save_dimension_profile(target_profile.model_dump(mode="json"))
        _update_manifest_profile_snapshot(args.index_dir, args, config, target_profile)

    generate_summary(args.index_dir, logger)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    logger.info(
        "Backfilled %d papers from profile %s to %s (%d changed sections, %d re-extracted)",
        len(requested_target_ids),
        old_profile.profile_id,
        target_profile.profile_id,
        len(changed_sections),
        len(reextract_sections),
    )
    return 0


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")

    if args.command == "migrate-store":
        return migrate_store(args, logger)

    if args.command == "diff":
        return diff_profiles(args)

    if args.command == "approve":
        return approve_dimensions(args)

    config = _load_config(args)
    if args.command == "suggest":
        return suggest_dimensions(args, config)
    if args.command == "backfill":
        return backfill_dimensions(args, config, logger)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
