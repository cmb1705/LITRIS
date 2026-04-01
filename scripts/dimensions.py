#!/usr/bin/env python
"""Manage portable semantic dimension profiles and storage."""

from __future__ import annotations

import argparse
import json
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
from src.analysis.schemas import DimensionedExtraction
from src.config import Config
from src.indexing.pipeline import (
    build_section_extractor,
    compute_similarity_pairs,
    configure_extraction_runtime,
    generate_summary,
    run_embedding_generation,
)
from src.indexing.raptor_pipeline import generate_scoped_raptor_summaries
from src.indexing.structured_store import StructuredStore
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging
from src.zotero.models import Author, PaperMetadata

PROPOSALS_FILENAME = "dimension_proposals.json"
MIGRATED_EXTRACTION_SCHEMA_VERSION = "2.0.0"

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
    suggest.add_argument("--sample-size", type=int, default=None)
    suggest.add_argument("--min-hits", type=int, default=3)
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
    """Generate heuristic dimension proposals from stored papers/extractions."""

    store = StructuredStore(args.index_dir)
    papers = store.load_papers()
    extractions = store.load_extractions()
    profile = DimensionProfile(**store.load_dimension_profile())

    sample_size = args.sample_size or config.dimensions.suggestion_sample_size
    sampled_ids = sorted(extractions)[:sample_size]
    existing_ids = set(profile.dimension_map)
    section_ids = {section.id for section in profile.sections}
    proposals: list[dict[str, Any]] = []

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

        proposals.append(
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
            }
        )

    output_path = args.output or (args.index_dir / PROPOSALS_FILENAME)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "profile_id": profile.profile_id,
        "sample_size": len(sampled_ids),
        "proposal_count": len(proposals),
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
    touched_sections = sorted(
        {
            entry.new_section or entry.old_section
            for entry in diff.entries
            if entry.status in {"added", "reworded", "replaced", "disabled"}
            and (entry.new_section or entry.old_section)
        }
    )

    papers = store.load_papers()
    extractions = store.load_extractions()
    target_ids = sorted(set(args.paper) if args.paper else set(extractions))
    missing_papers = [paper_id for paper_id in target_ids if paper_id not in papers]
    if missing_papers:
        logger.warning("Ignoring %d paper ids missing from papers.json", len(missing_papers))
        target_ids = [paper_id for paper_id in target_ids if paper_id in papers]

    if args.dry_run:
        payload = {
            "index_dir": str(args.index_dir),
            "paper_count": len(target_ids),
            "old_profile": old_profile.profile_id,
            "new_profile": target_profile.profile_id,
            "changed_sections": touched_sections,
            "diff": diff.model_dump(mode="json"),
        }
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).strip())
        return 0

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

    paper_models = [_paper_from_index_dict(paper_id, papers[paper_id]) for paper_id in target_ids]
    updated_extractions = dict(extractions)
    for result in extractor.extract_batch(paper_models):
        if not result.success or not result.extraction:
            logger.warning("Backfill failed for %s: %s", result.paper_id, result.error)
            continue

        existing_record = updated_extractions.get(result.paper_id, {"paper_id": result.paper_id})
        old_extraction = DimensionedExtraction.from_record(existing_record)
        new_extraction = result.extraction.to_dimensioned_extraction()
        merged_dimensions = {**old_extraction.dimensions, **new_extraction.dimensions}
        merged = DimensionedExtraction(
            paper_id=result.paper_id,
            profile_id=new_extraction.profile_id,
            profile_version=new_extraction.profile_version,
            profile_fingerprint=new_extraction.profile_fingerprint,
            prompt_version=new_extraction.prompt_version,
            extraction_model=new_extraction.extraction_model,
            extracted_at=new_extraction.extracted_at,
            dimensions=merged_dimensions,
            dimension_coverage=new_extraction.dimension_coverage,
            coverage_flags=new_extraction.coverage_flags,
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

    store.save_dimension_profile(target_profile.model_dump(mode="json"))
    store.save_extractions(updated_extractions)

    if not args.skip_embeddings:
        scoped_papers = [
            _paper_from_index_dict(paper_id, papers[paper_id])
            for paper_id in target_ids
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

    generate_summary(args.index_dir, logger)
    logger.info(
        "Backfilled %d papers from profile %s to %s",
        len(target_ids),
        old_profile.profile_id,
        target_profile.profile_id,
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
