"""Unified index orchestration for full builds, updates, and maintenance flows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm import tqdm

from src.analysis.classification_store import (
    ClassificationIndex,
    ClassificationRecord,
    ClassificationStore,
)
from src.analysis.dimensions import get_default_dimension_registry
from src.analysis.extraction_intent_classifier import (
    EXTRACTION_INTENT_SCHEMA_VERSION,
    ExtractionPlanItem,
    build_extraction_plan,
    classify_extraction_intent,
    group_extraction_plan_by_profile,
    resolve_hybrid_profile,
    summarize_intents,
)
from src.analysis.schemas import SemanticAnalysis
from src.analysis.semantic_prompts import SEMANTIC_PROMPT_VERSION
from src.config import Config
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.indexing.embeddings import CHUNK_TYPES
from src.indexing.pipeline import (
    build_section_extractor,
    compute_similarity_pairs,
    configure_extraction_runtime,
    generate_summary,
    load_reusable_text_snapshots,
    run_embedding_generation,
    run_extraction,
    run_gap_fill,
    update_classification_extraction_methods,
    verify_cli_authentication,
    write_skipped_report,
)
from src.indexing.raptor_pipeline import (
    RAPTOR_SCHEMA_VERSION,
    generate_scoped_raptor_summaries,
    prune_raptor_cache,
    rebuild_raptor_cache_from_store,
)
from src.indexing.structured_store import StructuredStore
from src.indexing.vector_store import VectorStore
from src.references.base import BaseReferenceDB
from src.references.factory import create_reference_db
from src.utils.checkpoint import CheckpointManager
from src.utils.deduplication import analyze_doi_overlap, extract_existing_dois, filter_by_doi
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import LogContext
from src.zotero.models import PaperMetadata

MANIFEST_FILENAME = "index_manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"
CHUNK_SCHEMA_VERSION = "2.0.0"
SIMILARITY_SCHEMA_VERSION = "1.0.0"
EXTRACTION_SCHEMA_VERSION = SEMANTIC_PROMPT_VERSION
CLASSIFICATION_POLICY_VERSION = "1.1.0"

STAGE_NAMES = [
    "classification",
    "extraction",
    "embeddings",
    "raptor",
    "similarity",
    "summary",
]
LIVE_INDEX_ARTIFACTS = [
    "papers.json",
    "semantic_analyses.json",
    "dimension_profile.json",
    "fulltext",
    "fulltext_manifest.json",
    "metadata.json",
    "summary.json",
    "similarity_pairs.json",
    "raptor_summaries.json",
    MANIFEST_FILENAME,
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def fingerprint_payload(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible payload."""
    normalized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class PaperSnapshot:
    """Source-state snapshot used for attachment-aware change detection."""

    paper_id: str
    zotero_key: str
    pdf_attachment_key: str | None
    date_modified: str | None
    pdf_path: str | None
    pdf_size: int | None
    pdf_mtime_ns: int | None
    source_path: str | None
    source_size: int | None
    source_mtime_ns: int | None
    source_media_type: str | None
    source_fingerprint: str
    extractable: bool | None = None
    indexed: bool | None = None

    @classmethod
    def from_paper(cls, paper: PaperMetadata) -> PaperSnapshot:
        pdf_size = None
        pdf_mtime_ns = None
        source_size = None
        source_mtime_ns = None
        if paper.pdf_path and Path(paper.pdf_path).exists():
            try:
                stat = Path(paper.pdf_path).stat()
                pdf_size = int(stat.st_size)
                pdf_mtime_ns = int(stat.st_mtime_ns)
            except OSError:
                pdf_size = None
                pdf_mtime_ns = None
        if paper.source_path and Path(paper.source_path).exists():
            try:
                source_stat = Path(paper.source_path).stat()
                source_size = int(source_stat.st_size)
                source_mtime_ns = int(source_stat.st_mtime_ns)
            except OSError:
                source_size = None
                source_mtime_ns = None

        payload = {
            "paper_id": paper.paper_id,
            "zotero_key": paper.zotero_key,
            "pdf_attachment_key": paper.pdf_attachment_key,
            "item_type": paper.item_type,
            "title": paper.title,
            "authors": [author.model_dump() for author in paper.authors],
            "publication_year": paper.publication_year,
            "publication_date": paper.publication_date,
            "journal": paper.journal,
            "volume": paper.volume,
            "issue": paper.issue,
            "pages": paper.pages,
            "doi": paper.doi,
            "isbn": paper.isbn,
            "issn": paper.issn,
            "abstract": paper.abstract,
            "url": paper.url,
            "collections": paper.collections,
            "tags": paper.tags,
            "pdf_path": str(paper.pdf_path) if paper.pdf_path else None,
            "pdf_size": pdf_size,
            "pdf_mtime_ns": pdf_mtime_ns,
            "source_path": str(paper.source_path) if paper.source_path else None,
            "source_size": source_size,
            "source_mtime_ns": source_mtime_ns,
            "source_media_type": paper.source_media_type,
            "date_added": paper.date_added.isoformat() if paper.date_added else None,
            "date_modified": paper.date_modified.isoformat() if paper.date_modified else None,
        }
        return cls(
            paper_id=paper.paper_id,
            zotero_key=paper.zotero_key,
            pdf_attachment_key=paper.pdf_attachment_key,
            date_modified=paper.date_modified.isoformat() if paper.date_modified else None,
            pdf_path=str(paper.pdf_path) if paper.pdf_path else None,
            pdf_size=pdf_size,
            pdf_mtime_ns=pdf_mtime_ns,
            source_path=str(paper.source_path) if paper.source_path else None,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
            source_media_type=paper.source_media_type,
            source_fingerprint=fingerprint_payload(payload),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperSnapshot:
        return cls(
            paper_id=data["paper_id"],
            zotero_key=data.get("zotero_key", ""),
            pdf_attachment_key=data.get("pdf_attachment_key"),
            date_modified=data.get("date_modified"),
            pdf_path=data.get("pdf_path"),
            pdf_size=data.get("pdf_size"),
            pdf_mtime_ns=data.get("pdf_mtime_ns"),
            source_path=data.get("source_path"),
            source_size=data.get("source_size"),
            source_mtime_ns=data.get("source_mtime_ns"),
            source_media_type=data.get("source_media_type"),
            source_fingerprint=data.get("source_fingerprint", ""),
            extractable=data.get("extractable"),
            indexed=data.get("indexed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeSet:
    """Attachment-aware change categories for a sync run."""

    new_items: list[str] = field(default_factory=list)
    modified_items: list[str] = field(default_factory=list)
    deleted_items: list[str] = field(default_factory=list)
    unchanged_items: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_items or self.modified_items or self.deleted_items)

    @property
    def touched_items(self) -> list[str]:
        ordered = []
        seen = set()
        for item_id in [*self.new_items, *self.modified_items, *self.deleted_items]:
            if item_id not in seen:
                seen.add(item_id)
                ordered.append(item_id)
        return ordered

    def summary(self) -> str:
        parts = []
        if self.new_items:
            parts.append(f"{len(self.new_items)} new")
        if self.modified_items:
            parts.append(f"{len(self.modified_items)} modified")
        if self.deleted_items:
            parts.append(f"{len(self.deleted_items)} deleted")
        return ", ".join(parts) if parts else "No changes detected"


@dataclass
class IndexArtifactSnapshot:
    """Filesystem snapshot of live index artifacts for rollback."""

    index_dir: Path
    backup_dir: Path
    artifact_names: list[str]

    @classmethod
    def capture(cls, index_dir: Path, artifact_names: list[str]) -> IndexArtifactSnapshot:
        backup_dir = Path(tempfile.mkdtemp(prefix="index_snapshot_", dir=index_dir))
        snapshot = cls(index_dir=index_dir, backup_dir=backup_dir, artifact_names=artifact_names)
        for name in artifact_names:
            source = index_dir / name
            target = backup_dir / name
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        return snapshot

    def restore(self) -> None:
        for name in self.artifact_names:
            source = self.backup_dir / name
            target = self.index_dir / name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)

    def cleanup(self) -> None:
        shutil.rmtree(self.backup_dir, ignore_errors=True)


@dataclass
class PendingStageWork:
    """Pending work flags persisted in the manifest."""

    paper_ids: list[str] = field(default_factory=list)
    all: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PendingStageWork:
        data = data or {}
        return cls(
            paper_ids=list(data.get("paper_ids", [])),
            all=bool(data.get("all", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"paper_ids": sorted(set(self.paper_ids)), "all": self.all}

    def extend(self, paper_ids: list[str]) -> None:
        current = set(self.paper_ids)
        current.update(paper_ids)
        self.paper_ids = sorted(current)

    def clear(self) -> None:
        self.paper_ids = []
        self.all = False

    def has_work(self) -> bool:
        return self.all or bool(self.paper_ids)


@dataclass
class IndexManifest:
    """Operational manifest used to plan safe builds versus updates."""

    schema_version: str = MANIFEST_SCHEMA_VERSION
    source_scope: dict[str, Any] = field(default_factory=dict)
    classification_policy: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    chunk_schema_version: str = CHUNK_SCHEMA_VERSION
    raptor_schema_version: str = RAPTOR_SCHEMA_VERSION
    similarity_schema_version: str = SIMILARITY_SCHEMA_VERSION
    stage_health: dict[str, str] = field(
        default_factory=lambda: dict.fromkeys(STAGE_NAMES, "unknown")
    )
    pending_work: dict[str, PendingStageWork] = field(
        default_factory=lambda: {
            "extraction": PendingStageWork(),
            "embeddings": PendingStageWork(),
            "raptor": PendingStageWork(),
            "similarity": PendingStageWork(),
        }
    )
    paper_snapshots: dict[str, PaperSnapshot] = field(default_factory=dict)
    last_full_build: str | None = None
    last_successful_sync: str | None = None
    last_run: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> tuple[IndexManifest | None, str | None]:
        if not path.exists():
            return None, "missing"
        raw = safe_read_json(path, default=None)
        if not isinstance(raw, dict):
            return None, "corrupt"
        try:
            pending_work = {
                name: PendingStageWork.from_dict(raw.get("pending_work", {}).get(name))
                for name in ("extraction", "embeddings", "raptor", "similarity")
            }
            snapshots = {
                paper_id: PaperSnapshot.from_dict(snapshot_data)
                for paper_id, snapshot_data in raw.get("paper_snapshots", {}).items()
            }
            return (
                cls(
                    schema_version=raw.get("schema_version", MANIFEST_SCHEMA_VERSION),
                    source_scope=raw.get("source_scope", {}),
                    classification_policy=raw.get("classification_policy", {}),
                    extraction=raw.get("extraction", {}),
                    embedding=raw.get("embedding", {}),
                    chunk_schema_version=raw.get("chunk_schema_version", CHUNK_SCHEMA_VERSION),
                    raptor_schema_version=raw.get("raptor_schema_version", RAPTOR_SCHEMA_VERSION),
                    similarity_schema_version=raw.get(
                        "similarity_schema_version", SIMILARITY_SCHEMA_VERSION
                    ),
                    stage_health=raw.get(
                        "stage_health",
                        dict.fromkeys(STAGE_NAMES, "unknown"),
                    ),
                    pending_work=pending_work,
                    paper_snapshots=snapshots,
                    last_full_build=raw.get("last_full_build"),
                    last_successful_sync=raw.get("last_successful_sync"),
                    last_run=raw.get("last_run", {}),
                ),
                None,
            )
        except Exception:
            return None, "corrupt"

    def save(self, path: Path) -> None:
        safe_write_json(
            path,
            {
                "schema_version": self.schema_version,
                "source_scope": self.source_scope,
                "classification_policy": self.classification_policy,
                "extraction": self.extraction,
                "embedding": self.embedding,
                "chunk_schema_version": self.chunk_schema_version,
                "raptor_schema_version": self.raptor_schema_version,
                "similarity_schema_version": self.similarity_schema_version,
                "stage_health": self.stage_health,
                "pending_work": {
                    name: pending.to_dict() for name, pending in self.pending_work.items()
                },
                "paper_snapshots": {
                    paper_id: snapshot.to_dict()
                    for paper_id, snapshot in self.paper_snapshots.items()
                },
                "last_full_build": self.last_full_build,
                "last_successful_sync": self.last_successful_sync,
                "last_run": self.last_run,
            },
        )


@dataclass
class SyncPlan:
    """Resolved sync decision and run scope."""

    requested_mode: str
    resolved_mode: str
    reasons: list[str]
    change_set: ChangeSet
    advisories: list[str] = field(default_factory=list)
    forced_paper_ids: list[str] = field(default_factory=list)
    pending_work: dict[str, PendingStageWork] = field(default_factory=dict)
    clear_vector_store: bool = False
    full_rebuild: bool = False
    requires_full_extraction: bool = False
    noop: bool = False

    def has_pending_work(self) -> bool:
        return any(work.has_work() for work in self.pending_work.values())


def detect_snapshot_changes(
    previous_snapshots: dict[str, PaperSnapshot],
    current_snapshots: dict[str, PaperSnapshot],
    forced_paper_ids: set[str] | None = None,
) -> ChangeSet:
    """Detect new, modified, deleted, and unchanged papers from snapshots."""
    forced_paper_ids = forced_paper_ids or set()
    previous_ids = set(previous_snapshots)
    current_ids = set(current_snapshots)

    new_items = sorted(current_ids - previous_ids)
    deleted_items = sorted(previous_ids - current_ids)
    modified_items = []
    unchanged_items = []

    for paper_id in sorted(previous_ids & current_ids):
        if paper_id in forced_paper_ids:
            modified_items.append(paper_id)
            continue
        if (
            previous_snapshots[paper_id].source_fingerprint
            != current_snapshots[paper_id].source_fingerprint
        ):
            modified_items.append(paper_id)
        else:
            unchanged_items.append(paper_id)

    for paper_id in sorted(forced_paper_ids & current_ids):
        if paper_id not in new_items and paper_id not in modified_items:
            modified_items.append(paper_id)

    return ChangeSet(
        new_items=new_items,
        modified_items=sorted(modified_items),
        deleted_items=deleted_items,
        unchanged_items=sorted(unchanged_items),
    )


class IndexOrchestrator:
    """Single control plane for index build, update, and maintenance flows."""

    def __init__(
        self,
        project_root: Path,
        logger,
        *,
        index_dir: Path | None = None,
        config: Config | None = None,
    ):
        self.project_root = Path(project_root)
        if index_dir is not None:
            self.index_dir = Path(index_dir)
        elif config is not None:
            self.index_dir = config.get_index_path(self.project_root)
        else:
            self.index_dir = self.project_root / "data" / "index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.store = StructuredStore(self.index_dir)
        self.manifest_path = self.index_dir / MANIFEST_FILENAME

    @staticmethod
    def _format_extraction_config(extraction_info: dict[str, Any]) -> str:
        provider = extraction_info.get("provider") or "unknown-provider"
        mode = extraction_info.get("mode") or "default-mode"
        model = extraction_info.get("model") or "<provider default>"
        return f"{provider}/{mode}/{model}"

    def _build_extraction_drift_advisory(
        self,
        previous_info: dict[str, Any],
        requested_info: dict[str, Any],
    ) -> str:
        previous_label = self._format_extraction_config(previous_info)
        requested_label = self._format_extraction_config(requested_info)
        return (
            "Extraction defaults differ from the existing corpus "
            f"(baseline: {previous_label}; requested: {requested_label}). "
            "Auto/update preserves existing extractions and applies the current "
            "provider only to new, modified, pending, or explicitly targeted papers. "
            "Use --sync-mode full to re-extract unchanged papers."
        )

    def _log_plan_advisories(self, plan: SyncPlan) -> None:
        for advisory in plan.advisories:
            self.logger.warning(advisory)

    def run(self, args: argparse.Namespace, config: Config) -> int:
        start_time = datetime.now()
        start_perf = perf_counter()

        source = getattr(args, "source", "zotero")
        ref_db = self._create_reference_db(source, args, config)
        current_papers, papers_without_pdf = self._load_source_papers(ref_db, args)
        if getattr(args, "dedupe_by_doi", False):
            existing_dois = extract_existing_dois(self.index_dir)
            if existing_dois:
                before = len(current_papers)
                current_papers, doi_duplicates = filter_by_doi(current_papers, existing_dois)
                if doi_duplicates:
                    self.logger.info(
                        "DOI deduplication: %d papers skipped (matching DOIs in existing index)",
                        len(doi_duplicates),
                    )
                if before != len(current_papers):
                    self.logger.info(
                        "Source scope after DOI dedupe: %d papers",
                        len(current_papers),
                    )

        if getattr(args, "show_skipped", False):
            self._show_skipped(papers_without_pdf)
            return 0

        class_store = ClassificationStore(
            index_dir=self.index_dir,
            min_publication_words=config.processing.min_publication_words,
            min_publication_pages=config.processing.min_publication_pages,
        )

        if getattr(args, "classify_only", False):
            return self._run_classify_only(args, config, class_store, current_papers)

        checkpoint_mgr = CheckpointManager(self.index_dir, checkpoint_id="extraction")
        if getattr(args, "reset_checkpoint", False):
            checkpoint_mgr.reset()
            self.logger.info("Checkpoint reset, starting fresh")

        if getattr(args, "show_failed", False):
            self._show_failed(checkpoint_mgr)
            return 0

        current_snapshots = {
            paper.paper_id: PaperSnapshot.from_paper(paper) for paper in current_papers
        }
        current_papers_by_id = {paper.paper_id: paper for paper in current_papers}
        existing_papers = self.store.load_papers()
        existing_extractions = self.store.load_extractions()
        class_index = class_store.load()

        plan = self.plan_sync(
            args=args,
            config=config,
            ref_db=ref_db,
            current_snapshots=current_snapshots,
            existing_papers=existing_papers,
            existing_extractions=existing_extractions,
            class_index=class_index,
        )

        if (
            getattr(args, "explain_plan", False)
            or getattr(args, "dry_run", False)
            or getattr(args, "detect_only", False)
        ):
            self._print_plan(plan, current_snapshots)

        if getattr(args, "detect_only", False):
            return 0 if plan.resolved_mode != "invalid" else 1

        if plan.resolved_mode == "invalid":
            for reason in plan.reasons:
                self.logger.error(reason)
            return 1

        self._log_plan_advisories(plan)

        classification_scope_ids = self._classification_scope_ids(
            args=args,
            plan=plan,
            current_papers_by_id=current_papers_by_id,
            class_index=class_index,
            config=config,
        )

        if plan.noop and not classification_scope_ids and not plan.change_set.deleted_items:
            self.logger.info("No changes detected and no pending work")
            return 0

        manifest, _ = IndexManifest.load(self.manifest_path)
        manifest = manifest or IndexManifest()
        manifest.source_scope = self._source_scope_info(ref_db, args)
        manifest.classification_policy = self._classification_policy_info(args, config)
        manifest.extraction = self._extraction_info(args, config)
        manifest.embedding = self._embedding_info(config)
        manifest.chunk_schema_version = CHUNK_SCHEMA_VERSION
        manifest.raptor_schema_version = RAPTOR_SCHEMA_VERSION
        manifest.similarity_schema_version = SIMILARITY_SCHEMA_VERSION
        manifest.last_run = {
            "requested_mode": plan.requested_mode,
            "resolved_mode": plan.resolved_mode,
            "started_at": start_time.isoformat(),
            "reasons": plan.reasons,
            "advisories": plan.advisories,
        }

        stage_times: dict[str, float] = {}
        run_failed_ids: list[str] = []

        if classification_scope_ids or plan.change_set.deleted_items:
            with self._time_stage("classification", stage_times):
                self._update_classification_index(
                    class_store=class_store,
                    class_index=class_index,
                    papers=[
                        current_papers_by_id[paper_id] for paper_id in classification_scope_ids
                    ],
                    deleted_paper_ids=plan.change_set.deleted_items,
                    config=config,
                    force_reclassify=getattr(args, "reclassify", False),
                    refresh_text=getattr(args, "refresh_text", False),
                )
                manifest.stage_health["classification"] = "ok"

        eligible_ids = self._eligible_paper_ids(
            current_snapshots=current_snapshots,
            class_index=class_index,
            index_all=getattr(args, "index_all", False),
        )
        desired_index_ids = set(eligible_ids)
        removed_from_index_ids = {
            paper_id
            for paper_id in (plan.change_set.new_items + plan.change_set.modified_items)
            if paper_id in current_snapshots and paper_id not in desired_index_ids
        }

        if getattr(args, "show_doi_overlap", False):
            self._show_doi_overlap(current_papers, desired_index_ids)
            return 0

        extraction_required_ids = self._extraction_required_ids(
            args=args,
            plan=plan,
            desired_index_ids=desired_index_ids,
            existing_extractions=existing_extractions,
        )
        extraction_plan_items = self._build_extraction_plan_items(
            extraction_required_ids=extraction_required_ids,
            class_index=class_index,
            config=config,
        )
        if getattr(args, "explain_plan", False) or getattr(args, "dry_run", False):
            grouped_plan = group_extraction_plan_by_profile(extraction_plan_items)
            print("\nExtraction routing plan:")
            for line in self._format_extraction_plan_lines(
                grouped_plan,
                current_papers_by_id=current_papers_by_id,
            ):
                print(f"  {line}" if not line.startswith("  ") else line)
        if getattr(args, "dry_run", False):
            return 0 if plan.resolved_mode != "invalid" else 1
        provider, mode, cache_dir, parallel_workers, use_cache = configure_extraction_runtime(
            args, config, self.logger
        )
        if (
            mode == "cli"
            and extraction_required_ids
            and not getattr(args, "skip_extraction", False)
        ):
            if provider == "anthropic":
                verify_cli_authentication()
            elif provider == "openai" and not shutil.which("codex"):
                self.logger.error("Codex CLI is required for OpenAI CLI mode")
                return 1

        if (
            mode == "batch_api"
            and extraction_required_ids
            and not getattr(args, "skip_extraction", False)
        ):
            print("\n" + "=" * 60)
            print("Batch API Mode")
            print("=" * 60)
            print(
                "\nUse scripts/dimensions.py batch submit --index-dir <index_dir> "
                "for async semantic extraction, then rerun build_index.py."
            )
            return 0

        return self._execute_plan(
            args=args,
            config=config,
            ref_db=ref_db,
            class_store=class_store,
            class_index=class_index,
            current_papers=current_papers,
            current_papers_by_id=current_papers_by_id,
            current_snapshots=current_snapshots,
            existing_papers=existing_papers,
            existing_extractions=existing_extractions,
            checkpoint_mgr=checkpoint_mgr,
            plan=plan,
            manifest=manifest,
            desired_index_ids=desired_index_ids,
            removed_from_index_ids=removed_from_index_ids,
            extraction_required_ids=extraction_required_ids,
            extraction_plan_items=extraction_plan_items,
            provider=provider,
            mode=mode,
            cache_dir=cache_dir,
            parallel_workers=parallel_workers,
            use_cache=use_cache,
            stage_times=stage_times,
            run_failed_ids=run_failed_ids,
            start_perf=start_perf,
        )

    def _execute_plan(
        self,
        args: argparse.Namespace,
        config: Config,
        ref_db: BaseReferenceDB,
        class_store: ClassificationStore,
        class_index: ClassificationIndex,
        current_papers: list[PaperMetadata],
        current_papers_by_id: dict[str, PaperMetadata],
        current_snapshots: dict[str, PaperSnapshot],
        existing_papers: dict[str, dict],
        existing_extractions: dict[str, dict],
        checkpoint_mgr: CheckpointManager,
        plan: SyncPlan,
        manifest: IndexManifest,
        desired_index_ids: set[str],
        removed_from_index_ids: set[str],
        extraction_required_ids: set[str],
        extraction_plan_items: list[ExtractionPlanItem],
        provider: str,
        mode: str,
        cache_dir: Path,
        parallel_workers: int,
        use_cache: bool,
        stage_times: dict[str, float],
        run_failed_ids: list[str],
        start_perf: float,
    ) -> int:
        extraction_requirements = self._describe_extraction_requirements(
            args=args,
            plan=plan,
            desired_index_ids=desired_index_ids,
            existing_extractions=existing_extractions,
            current_papers_by_id=current_papers_by_id,
        )

        if getattr(args, "skip_extraction", False) and extraction_required_ids:
            self.logger.error(
                "--skip-extraction is unsafe for %d scoped papers that need fresh extraction",
                len(extraction_required_ids),
            )
            for line in self._format_extraction_requirement_lines(extraction_requirements):
                self.logger.error("  %s", line)
            return 1

        extraction_results: list = []
        updated_extractions = dict(existing_extractions)
        extraction_interrupted = False
        pause_requested = False
        extraction_pending_ids: list[str] = []
        extractor = None

        extraction_candidate_ids = sorted(extraction_required_ids)
        extraction_plan_by_id = {item.paper_id: item for item in extraction_plan_items}
        extraction_candidate_papers = [
            current_papers_by_id[paper_id]
            for paper_id in extraction_candidate_ids
            if paper_id in current_papers_by_id
        ]
        extraction_candidate_papers = self._apply_retry_resume_limit_skip(
            args=args,
            papers=extraction_candidate_papers,
            checkpoint_mgr=checkpoint_mgr,
        )

        if getattr(args, "estimate_cost", False):
            grouped_candidate_plan: dict[str, list[PaperMetadata]] = {}
            for paper in extraction_candidate_papers:
                plan_item = extraction_plan_by_id.get(paper.paper_id)
                profile_key = plan_item.profile.profile_key if plan_item is not None else "fast"
                grouped_candidate_plan.setdefault(profile_key, []).append(paper)
            estimate = None
            for _profile_key, papers_for_profile in grouped_candidate_plan.items():
                plan_item = extraction_plan_by_id.get(papers_for_profile[0].paper_id)
                extractor = build_section_extractor(
                    args=args,
                    config=config,
                    cache_dir=cache_dir,
                    mode=mode,
                    parallel_workers=parallel_workers,
                    use_cache=use_cache,
                    run_control_path=self.index_dir / "run_control.json",
                    planned_profile=plan_item.profile if plan_item else None,
                )
                profile_estimate = extractor.estimate_batch_cost(papers_for_profile)
                if estimate is None:
                    estimate = profile_estimate
                else:
                    estimate["papers_with_pdf"] += profile_estimate.get("papers_with_pdf", 0)
                    estimate["papers_to_extract"] += profile_estimate.get("papers_to_extract", 0)
                    estimate["papers_cached"] = estimate.get(
                        "papers_cached", 0
                    ) + profile_estimate.get("papers_cached", 0)
                    estimate["estimated_total_cost"] += profile_estimate.get(
                        "estimated_total_cost", 0.0
                    )
            estimate = estimate or {
                "papers_with_pdf": 0,
                "papers_to_extract": 0,
                "estimated_total_cost": 0.0,
            }
            self._print_cost_estimate(estimate)
            return 0

        paper_dicts = dict(existing_papers)
        if extraction_candidate_papers and not getattr(args, "skip_extraction", False):
            with self._time_stage("extraction", stage_times):
                if not checkpoint_mgr.load():
                    checkpoint_mgr.initialize(
                        total_items=len(extraction_candidate_papers),
                        metadata={"started_at": datetime.now().isoformat()},
                    )
                source_text_snapshots = ref_db.load_source_text_snapshots(
                    extraction_candidate_papers
                )
                reusable_text_snapshots = load_reusable_text_snapshots(
                    self.store,
                    extraction_candidate_papers,
                    refresh_text=getattr(args, "refresh_text", False),
                )
                merged_text_snapshots = dict(source_text_snapshots)
                merged_text_snapshots.update(reusable_text_snapshots)
                grouped_plan = group_extraction_plan_by_profile(
                    [
                        extraction_plan_by_id[paper.paper_id]
                        for paper in extraction_candidate_papers
                        if paper.paper_id in extraction_plan_by_id
                    ]
                )
                grouped_plan_items = list(grouped_plan.items())
                cleared_cache = False
                for group_index, (profile_key, plan_items) in enumerate(grouped_plan_items):
                    group_papers = [
                        current_papers_by_id[item.paper_id]
                        for item in plan_items
                        if item.paper_id in current_papers_by_id
                    ]
                    extractor = build_section_extractor(
                        args=args,
                        config=config,
                        cache_dir=cache_dir,
                        mode=mode,
                        parallel_workers=parallel_workers,
                        use_cache=use_cache,
                        run_control_path=self.index_dir / "run_control.json",
                        planned_profile=plan_items[0].profile,
                    )
                    if getattr(args, "clear_cache", False) and not cleared_cache:
                        cleared = extractor.clear_cache()
                        self.logger.info(
                            "Cleared %d cached extractions before profile %s",
                            cleared,
                            profile_key,
                        )
                        cleared_cache = True
                    group_snapshots = {
                        paper.paper_id: merged_text_snapshots[paper.paper_id]
                        for paper in group_papers
                        if paper.paper_id in merged_text_snapshots
                    }
                    group_plan_map = {item.paper_id: item for item in plan_items}
                    extraction_run = run_extraction(
                        group_papers,
                        extractor,
                        self.index_dir,
                        paper_dicts,
                        updated_extractions,
                        checkpoint_mgr,
                        self.logger,
                        text_snapshots=group_snapshots,
                        extraction_plans=group_plan_map,
                    )
                    paper_dicts = extraction_run.paper_dicts
                    updated_extractions = extraction_run.extractions
                    extraction_results.extend(extraction_run.results)
                    extraction_interrupted = extraction_run.interrupted
                    pause_requested = extraction_run.pause_requested
                    extraction_pending_ids.extend(extraction_run.pending_ids)
                    checkpoint_mgr.save()
                    update_classification_extraction_methods(
                        class_index,
                        extraction_run.results,
                    )
                    class_store.save(class_index)
                    if extraction_interrupted:
                        remaining_plan_items = [
                            item
                            for _, items in grouped_plan_items[group_index + 1 :]
                            for item in items
                        ]
                        extraction_pending_ids.extend(
                            item.paper_id for item in remaining_plan_items
                        )
                        break
                if getattr(args, "gap_fill", False) and not extraction_interrupted:
                    run_gap_fill(
                        results=extraction_results,
                        existing_extractions=updated_extractions,
                        papers_to_extract=extraction_candidate_papers,
                        primary_provider=provider,
                        gap_fill_provider=getattr(args, "gap_fill_provider", None),
                        threshold=getattr(args, "gap_fill_threshold", 0.85),
                        mode=mode,
                        config=config,
                        index_dir=self.index_dir,
                        paper_dicts=paper_dicts,
                        checkpoint_mgr=checkpoint_mgr,
                        logger=self.logger,
                    )
                write_skipped_report(
                    extraction_results,
                    paper_dicts,
                    self.project_root / "data" / "out" / "experiments" / "skipped_items",
                    self.logger,
                )
                manifest.stage_health["extraction"] = "pending" if extraction_interrupted else "ok"

        failed_extraction_ids = {
            result.paper_id for result in extraction_results if not result.success
        }
        run_failed_ids.extend(sorted(failed_extraction_ids))

        scoped_current_ids = self._scoped_current_ids(
            args=args,
            plan=plan,
            desired_index_ids=desired_index_ids,
        )
        if plan.resolved_mode == "full" and not getattr(args, "paper", []):
            deleted_from_store: set[str] = set()
            final_papers = {
                paper_id: current_papers_by_id[paper_id].to_index_dict()
                for paper_id in sorted(desired_index_ids)
                if paper_id not in failed_extraction_ids
            }
            final_extractions = {
                paper_id: updated_extractions[paper_id]
                for paper_id in sorted(final_papers)
                if paper_id in updated_extractions
            }
        else:
            final_papers = dict(existing_papers)
            final_extractions = dict(existing_extractions)
            deleted_from_store = set(plan.change_set.deleted_items) | removed_from_index_ids
            for paper_id in deleted_from_store:
                final_papers.pop(paper_id, None)
                final_extractions.pop(paper_id, None)
            safe_apply_ids = {
                paper_id
                for paper_id in scoped_current_ids
                if paper_id not in failed_extraction_ids
                and (paper_id not in extraction_required_ids or paper_id in updated_extractions)
            }
            for paper_id in safe_apply_ids:
                final_papers[paper_id] = current_papers_by_id[paper_id].to_index_dict()
                if paper_id in updated_extractions:
                    final_extractions[paper_id] = updated_extractions[paper_id]

        embedding_scope_ids = self._embedding_scope_ids(
            args=args,
            plan=plan,
            final_papers=final_papers,
            final_extractions=final_extractions,
            failed_extraction_ids=failed_extraction_ids,
        )
        delete_embedding_ids = sorted(set(plan.change_set.deleted_items) | removed_from_index_ids)

        artifact_snapshot = IndexArtifactSnapshot.capture(
            self.index_dir,
            LIVE_INDEX_ARTIFACTS,
        )
        staged_chroma_dir: Path | None = None
        try:
            self.store.save_papers(final_papers)
            self.store.save_dimension_profile(
                get_default_dimension_registry().active_profile.model_dump(mode="json")
            )
            self.store.save_extractions(final_extractions)
            for paper_id in deleted_from_store:
                self.store.delete_text_snapshot(paper_id, persist_manifest=False)
            self.store.flush_fulltext_manifest()
            valid_index_ids = set(final_papers)
            prune_raptor_cache(self.index_dir, valid_index_ids)

            if extraction_interrupted:
                pending_embeddings = PendingStageWork()
                if plan.clear_vector_store:
                    pending_embeddings.all = True
                else:
                    pending_embeddings.extend(sorted(embedding_scope_ids))

                manifest.pending_work["extraction"].clear()
                manifest.pending_work["extraction"].extend(extraction_pending_ids)
                manifest.pending_work["embeddings"] = pending_embeddings
                manifest.pending_work["raptor"] = PendingStageWork.from_dict(
                    pending_embeddings.to_dict()
                )
                if (
                    pending_embeddings.has_work()
                    or bool(delete_embedding_ids)
                    or manifest.pending_work["similarity"].has_work()
                    or plan.clear_vector_store
                ):
                    manifest.pending_work["similarity"].all = True
                    manifest.stage_health["similarity"] = "pending"
                manifest.stage_health["embeddings"] = "pending"
                manifest.stage_health["raptor"] = "pending"
                manifest.paper_snapshots = self._resolved_manifest_snapshots(
                    previous_snapshots=manifest.paper_snapshots,
                    current_snapshots=current_snapshots,
                    class_index=class_index,
                    indexed_ids=set(final_papers),
                    preserved_current_ids=set(extraction_pending_ids),
                    preserved_deleted_ids=set(delete_embedding_ids),
                )
                completed_at = datetime.now()
                manifest.last_run = {
                    **manifest.last_run,
                    "interrupted": True,
                    "pause_requested": pause_requested,
                    "completed_at": completed_at.isoformat(),
                    "duration_seconds": round(perf_counter() - start_perf, 3),
                    "stage_durations": stage_times,
                    "counts": {
                        "source_papers": len(current_papers),
                        "indexed_papers": len(final_papers),
                        "extractions": len(final_extractions),
                        "changes_new": len(plan.change_set.new_items),
                        "changes_modified": len(plan.change_set.modified_items),
                        "changes_deleted": len(plan.change_set.deleted_items),
                        "failed_extractions": len(run_failed_ids),
                        "pending_extractions": len(extraction_pending_ids),
                    },
                }
                manifest.save(self.manifest_path)
                class_store.save(class_index)
                if pause_requested and extractor is not None:
                    extractor.clear_pause_request()
                self.logger.warning(
                    "Extraction %s; saved partial index state and %d pending paper(s) for resume",
                    "paused" if pause_requested else "interrupted",
                    len(extraction_pending_ids),
                )
                return 1
            if extraction_candidate_papers:
                manifest.pending_work["extraction"].clear()

            if getattr(args, "skip_embeddings", False):
                with self._time_stage("embeddings", stage_times):
                    self._clear_or_delete_vectors_for_pending(
                        plan=plan,
                        delete_paper_ids=delete_embedding_ids + sorted(embedding_scope_ids),
                    )
                    manifest.pending_work["embeddings"].all = plan.full_rebuild and not getattr(
                        args, "paper", []
                    )
                    if not manifest.pending_work["embeddings"].all:
                        manifest.pending_work["embeddings"].extend(sorted(embedding_scope_ids))
                    manifest.pending_work["raptor"] = PendingStageWork.from_dict(
                        manifest.pending_work["embeddings"].to_dict()
                    )
                    manifest.pending_work["similarity"].all = True
                    manifest.stage_health["embeddings"] = "pending"
                    manifest.stage_health["raptor"] = "pending"
            else:
                extraction_models = self._normalize_extractions(final_extractions)
                if embedding_scope_ids or delete_embedding_ids or plan.clear_vector_store:
                    with self._time_stage("raptor", stage_times):
                        raptor_papers = (
                            [
                                current_papers_by_id[paper_id]
                                for paper_id in sorted(valid_index_ids)
                                if paper_id in current_papers_by_id
                            ]
                            if plan.clear_vector_store
                            else [
                                current_papers_by_id[paper_id]
                                for paper_id in sorted(embedding_scope_ids)
                                if paper_id in current_papers_by_id
                            ]
                        )
                        raptor_scope_ids = {
                            paper.paper_id
                            for paper in raptor_papers
                            if paper.paper_id in extraction_models
                        }
                        raptor_summaries = generate_scoped_raptor_summaries(
                            papers=raptor_papers,
                            extractions={
                                paper_id: extraction_models[paper_id]
                                for paper_id in raptor_scope_ids
                            },
                            index_dir=self.index_dir,
                            mode="template",
                            force=True,
                        )
                        manifest.stage_health["raptor"] = "ok"
                    with self._time_stage("embeddings", stage_times):
                        if plan.clear_vector_store:
                            staged_chroma_dir = self._create_staged_chroma_dir()
                        embedding_run_metadata: dict[str, Any] = {}
                        run_embedding_generation(
                            papers=raptor_papers,
                            extractions=final_extractions,
                            index_dir=self.index_dir,
                            embedding_model=config.embeddings.model,
                            rebuild=plan.clear_vector_store,
                            logger=self.logger,
                            embedding_backend=config.embeddings.backend,
                            ollama_base_url=config.embeddings.ollama_base_url,
                            query_prefix=config.embeddings.query_prefix,
                            document_prefix=config.embeddings.document_prefix,
                            embedding_batch_size=config.embeddings.batch_size,
                            raptor_summaries=raptor_summaries,
                            delete_paper_ids=(
                                [] if plan.clear_vector_store else delete_embedding_ids
                            ),
                            vector_store_dir=staged_chroma_dir,
                            run_metadata=embedding_run_metadata,
                        )
                        manifest.last_run.update(embedding_run_metadata)
                        if staged_chroma_dir is not None:
                            self._commit_staged_chroma_dir(staged_chroma_dir)
                            staged_chroma_dir = None
                        manifest.pending_work["embeddings"].clear()
                        manifest.pending_work["raptor"].clear()
                        manifest.stage_health["embeddings"] = "ok"

            similarity_needed = (
                manifest.pending_work["similarity"].has_work()
                or bool(embedding_scope_ids)
                or bool(delete_embedding_ids)
                or plan.clear_vector_store
            )
            if getattr(args, "skip_similarity", False):
                manifest.pending_work["similarity"].all = (
                    similarity_needed or manifest.pending_work["similarity"].all
                )
                manifest.stage_health["similarity"] = "pending"
            elif similarity_needed and not getattr(args, "skip_embeddings", False):
                with self._time_stage("similarity", stage_times):
                    try:
                        compute_similarity_pairs(
                            index_dir=self.index_dir,
                            embedding_model=config.embeddings.model,
                            top_n=20,
                            embedding_backend=config.embeddings.backend,
                            ollama_base_url=config.embeddings.ollama_base_url,
                            query_prefix=config.embeddings.query_prefix,
                            document_prefix=config.embeddings.document_prefix,
                            embedding_batch_size=config.embeddings.batch_size,
                            logger=self.logger,
                        )
                    except Exception as exc:
                        self.logger.error(
                            "similarity stage failed; keeping index usable and marking similarity pending: %s",
                            exc,
                        )
                        manifest.pending_work["similarity"].all = True
                        manifest.stage_health["similarity"] = "pending"
                    else:
                        manifest.pending_work["similarity"].clear()
                        manifest.stage_health["similarity"] = "ok"

            with self._time_stage("summary", stage_times):
                try:
                    summary = generate_summary(self.index_dir, self.logger)
                except Exception as exc:
                    self.logger.error(
                        "summary stage failed; continuing with existing summary metadata: %s",
                        exc,
                    )
                    manifest.stage_health["summary"] = "failed"
                    summary = safe_read_json(self.index_dir / "summary.json", default={}) or {}
                else:
                    manifest.stage_health["summary"] = "ok"

            self._write_metadata(
                manifest=manifest,
                final_papers=final_papers,
                final_extractions=final_extractions,
                stage_times=stage_times,
                failed_paper_ids=run_failed_ids,
                summary=summary,
            )
        except Exception:
            if staged_chroma_dir is not None:
                shutil.rmtree(staged_chroma_dir, ignore_errors=True)
            artifact_snapshot.restore()
            self.store = StructuredStore(self.index_dir)
            self.logger.error("Rolled back live index artifacts after persist-stage failure")
            raise
        finally:
            artifact_snapshot.cleanup()

        manifest.paper_snapshots = self._resolved_manifest_snapshots(
            previous_snapshots=manifest.paper_snapshots,
            current_snapshots=current_snapshots,
            class_index=class_index,
            indexed_ids=set(final_papers),
        )

        completed_at = datetime.now()
        manifest.last_successful_sync = completed_at.isoformat()
        if plan.clear_vector_store or (
            plan.resolved_mode == "full" and not getattr(args, "paper", [])
        ):
            manifest.last_full_build = completed_at.isoformat()
        manifest.last_run = {
            **manifest.last_run,
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round(perf_counter() - start_perf, 3),
            "stage_durations": stage_times,
            "counts": {
                "source_papers": len(current_papers),
                "indexed_papers": len(final_papers),
                "extractions": len(final_extractions),
                "changes_new": len(plan.change_set.new_items),
                "changes_modified": len(plan.change_set.modified_items),
                "changes_deleted": len(plan.change_set.deleted_items),
                "failed_extractions": len(run_failed_ids),
            },
        }
        manifest.save(self.manifest_path)
        class_store.save(class_index)

        if run_failed_ids:
            return 1
        return 0

    def _snapshot_with_status(
        self,
        snapshot: PaperSnapshot,
        class_index: ClassificationIndex,
        indexed_ids: set[str],
    ) -> PaperSnapshot:
        """Return a snapshot copy annotated with extractable/indexed status."""

        resolved = PaperSnapshot.from_dict(snapshot.to_dict())
        record = class_index.papers.get(snapshot.paper_id)
        if record is not None:
            resolved.extractable = record.extractable
        resolved.indexed = snapshot.paper_id in indexed_ids
        return resolved

    def _resolved_manifest_snapshots(
        self,
        previous_snapshots: dict[str, PaperSnapshot],
        current_snapshots: dict[str, PaperSnapshot],
        class_index: ClassificationIndex,
        indexed_ids: set[str],
        preserved_current_ids: set[str] | None = None,
        preserved_deleted_ids: set[str] | None = None,
    ) -> dict[str, PaperSnapshot]:
        """Resolve the manifest snapshot view for the current run outcome."""

        preserved_current_ids = preserved_current_ids or set()
        preserved_deleted_ids = preserved_deleted_ids or set()
        resolved: dict[str, PaperSnapshot] = {}

        for paper_id in sorted(preserved_deleted_ids):
            previous = previous_snapshots.get(paper_id)
            if previous is not None:
                resolved[paper_id] = PaperSnapshot.from_dict(previous.to_dict())

        for paper_id in sorted(current_snapshots):
            if paper_id in preserved_current_ids:
                previous = previous_snapshots.get(paper_id)
                if previous is not None:
                    resolved[paper_id] = PaperSnapshot.from_dict(previous.to_dict())
                continue
            resolved[paper_id] = self._snapshot_with_status(
                current_snapshots[paper_id],
                class_index=class_index,
                indexed_ids=indexed_ids,
            )

        return resolved

    def refresh_derived_artifacts(
        self,
        config: Config,
        mode: str = "template",
        provider: str = "google",
        model: str | None = None,
        force: bool = False,
        skip_similarity: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Maintenance entrypoint for RAPTOR and similarity refresh."""
        paper_dicts = self.store.load_papers()
        extraction_models = self._normalize_extractions(self.store.load_extractions())
        manifest, _ = IndexManifest.load(self.manifest_path)
        manifest = manifest or IndexManifest()
        if dry_run:
            self.logger.info(
                "DRY RUN: would regenerate RAPTOR for %d papers and refresh similarity=%s",
                len(extraction_models),
                not skip_similarity,
            )
            return 0

        raptor_summaries = rebuild_raptor_cache_from_store(
            index_dir=self.index_dir,
            paper_dicts=paper_dicts,
            extractions=extraction_models,
            mode=mode,
            provider=provider,
            model=model,
            force=force,
        )
        raptor_papers = [
            current
            for current in (
                self._paper_from_index_dict(paper_id, paper_data)
                for paper_id, paper_data in paper_dicts.items()
            )
            if current.paper_id in extraction_models
        ]
        staged_chroma_dir = self._create_staged_chroma_dir()
        embedding_run_metadata: dict[str, Any] = {}
        run_embedding_generation(
            papers=raptor_papers,
            extractions=extraction_models,
            index_dir=self.index_dir,
            embedding_model=config.embeddings.model,
            rebuild=True,
            logger=self.logger,
            embedding_backend=config.embeddings.backend,
            ollama_base_url=config.embeddings.ollama_base_url,
            query_prefix=config.embeddings.query_prefix,
            document_prefix=config.embeddings.document_prefix,
            embedding_batch_size=config.embeddings.batch_size,
            raptor_summaries=raptor_summaries,
            delete_paper_ids=[],
            vector_store_dir=staged_chroma_dir,
            run_metadata=embedding_run_metadata,
        )
        self._commit_staged_chroma_dir(staged_chroma_dir)
        if not skip_similarity:
            compute_similarity_pairs(
                index_dir=self.index_dir,
                embedding_model=config.embeddings.model,
                top_n=20,
                embedding_backend=config.embeddings.backend,
                ollama_base_url=config.embeddings.ollama_base_url,
                query_prefix=config.embeddings.query_prefix,
                document_prefix=config.embeddings.document_prefix,
                embedding_batch_size=config.embeddings.batch_size,
                logger=self.logger,
            )
            manifest.pending_work["similarity"].clear()
            manifest.stage_health["similarity"] = "ok"
        else:
            manifest.pending_work["similarity"].all = True
            manifest.stage_health["similarity"] = "pending"
        manifest.pending_work["embeddings"].clear()
        manifest.pending_work["raptor"].clear()
        manifest.stage_health["embeddings"] = "ok"
        manifest.stage_health["raptor"] = "ok"
        manifest.embedding = self._embedding_info(config)
        manifest.chunk_schema_version = CHUNK_SCHEMA_VERSION
        manifest.raptor_schema_version = RAPTOR_SCHEMA_VERSION
        manifest.similarity_schema_version = SIMILARITY_SCHEMA_VERSION
        manifest.last_successful_sync = datetime.now().isoformat()
        manifest.last_run = {
            "requested_mode": "maintenance",
            "resolved_mode": "maintenance",
            "completed_at": manifest.last_successful_sync,
            "reasons": ["rebuild_raptor_similarity.py wrapper"],
            **embedding_run_metadata,
        }
        summary = generate_summary(self.index_dir, self.logger)
        self._write_metadata(
            manifest=manifest,
            final_papers=paper_dicts,
            final_extractions=self.store.load_extractions(),
            stage_times={},
            failed_paper_ids=[],
            summary=summary,
        )
        manifest.save(self.manifest_path)
        return 0

    def plan_sync(
        self,
        args: argparse.Namespace,
        config: Config,
        ref_db: BaseReferenceDB,
        current_snapshots: dict[str, PaperSnapshot],
        existing_papers: dict[str, dict],
        existing_extractions: dict[str, dict],
        class_index: ClassificationIndex,
    ) -> SyncPlan:
        requested_mode = self._requested_sync_mode(args)
        manifest, manifest_error = IndexManifest.load(self.manifest_path)
        reasons: list[str] = []
        advisories: list[str] = []

        if requested_mode == "update" and ref_db.provider != "zotero":
            return SyncPlan(
                requested_mode=requested_mode,
                resolved_mode="invalid",
                reasons=["Incremental update mode is only supported for Zotero sources in v1"],
                change_set=ChangeSet(),
                pending_work={},
            )

        if requested_mode == "auto" and ref_db.provider != "zotero":
            reasons.append("Non-Zotero sources default to full rebuild in auto mode")

        compatibility_reasons: list[str] = []
        requires_full_extraction = False
        if manifest_error == "missing":
            compatibility_reasons.append("Index manifest is missing")
        elif manifest_error == "corrupt":
            compatibility_reasons.append("Index manifest is corrupt")

        source_info = self._source_scope_info(ref_db, args)
        classification_info = self._classification_policy_info(args, config)
        extraction_info = self._extraction_info(args, config)
        embedding_info = self._embedding_info(config)

        if manifest is not None:
            if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
                compatibility_reasons.append("Index manifest schema version is incompatible")
            if manifest.source_scope.get("fingerprint") != source_info["fingerprint"]:
                compatibility_reasons.append("Source scope changed")
            if (
                manifest.classification_policy.get("fingerprint")
                != classification_info["fingerprint"]
            ):
                compatibility_reasons.append("Classification policy changed")
            if manifest.extraction.get("fingerprint") != extraction_info["fingerprint"]:
                advisories.append(
                    self._build_extraction_drift_advisory(
                        previous_info=manifest.extraction,
                        requested_info=extraction_info,
                    )
                )
            if manifest.embedding.get("fingerprint") != embedding_info["fingerprint"]:
                compatibility_reasons.append("Embedding configuration changed")
            if manifest.chunk_schema_version != CHUNK_SCHEMA_VERSION:
                compatibility_reasons.append("Chunk schema version changed")
            if manifest.raptor_schema_version != RAPTOR_SCHEMA_VERSION:
                compatibility_reasons.append("RAPTOR schema version changed")
            if manifest.similarity_schema_version != SIMILARITY_SCHEMA_VERSION:
                compatibility_reasons.append("Similarity schema version changed")
            if not self._structured_store_consistent(
                existing_papers, existing_extractions, manifest
            ):
                compatibility_reasons.append("Structured store and manifest are inconsistent")
                requires_full_extraction = True

            vector_reason = self._validate_vector_store()
            if vector_reason:
                compatibility_reasons.append(vector_reason)
            else:
                chunk_reason = self._validate_chunk_types()
                if chunk_reason:
                    compatibility_reasons.append(chunk_reason)

        target_current_ids, target_deleted_ids, target_unmatched = self._resolve_target_ids(
            target_keys=getattr(args, "paper", []) or [],
            current_snapshots=current_snapshots,
            manifest=manifest,
            existing_papers=existing_papers,
        )
        if target_unmatched:
            reasons.append(f"Unmatched target IDs: {', '.join(sorted(target_unmatched))}")

        previous_snapshots = manifest.paper_snapshots if manifest else {}
        change_set = detect_snapshot_changes(
            previous_snapshots=previous_snapshots,
            current_snapshots=current_snapshots,
            forced_paper_ids=set(target_current_ids),
        )
        if target_deleted_ids:
            deleted = set(change_set.deleted_items)
            deleted.update(target_deleted_ids)
            change_set.deleted_items = sorted(deleted)

        if getattr(args, "new_only", False):
            change_set.modified_items = []
            change_set.deleted_items = []
        if getattr(args, "delete_only", False):
            change_set.new_items = []
            change_set.modified_items = []

        pending_work = (
            manifest.pending_work
            if manifest
            else {
                "extraction": PendingStageWork(),
                "embeddings": PendingStageWork(),
                "raptor": PendingStageWork(),
                "similarity": PendingStageWork(),
            }
        )

        if requested_mode == "update" and compatibility_reasons:
            return SyncPlan(
                requested_mode=requested_mode,
                resolved_mode="invalid",
                reasons=[*compatibility_reasons, *reasons],
                change_set=change_set,
                advisories=advisories,
                forced_paper_ids=sorted(target_current_ids),
                pending_work=pending_work,
            )

        resolved_mode = requested_mode
        full_rebuild = False
        clear_vector_store = False
        noop = False

        if requested_mode == "full":
            resolved_mode = "full"
            full_rebuild = True
            clear_vector_store = not bool(getattr(args, "paper", []))
        elif compatibility_reasons or ref_db.provider != "zotero":
            resolved_mode = "full"
            full_rebuild = True
            clear_vector_store = not bool(getattr(args, "paper", []))
            reasons = [*compatibility_reasons, *reasons]
        else:
            resolved_mode = "update"
            if not change_set.has_changes and not any(
                work.has_work() for work in pending_work.values()
            ):
                noop = True

        if getattr(args, "paper", []) and requested_mode == "full":
            reasons.append("--paper with --sync-mode full runs a targeted refresh only")
            clear_vector_store = False

        return SyncPlan(
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            reasons=reasons,
            change_set=change_set,
            advisories=advisories,
            forced_paper_ids=sorted(target_current_ids),
            pending_work=pending_work,
            clear_vector_store=clear_vector_store,
            full_rebuild=full_rebuild,
            requires_full_extraction=requires_full_extraction,
            noop=noop,
        )

    def _create_reference_db(
        self,
        source: str,
        args: argparse.Namespace,
        config: Config,
    ) -> BaseReferenceDB:
        self.logger.info("Loading papers from %s source...", source)
        if source == "zotero":
            return create_reference_db(
                "zotero",
                db_path=config.get_zotero_db_path(),
                storage_path=config.get_storage_path(),
            )
        if not getattr(args, "source_path", None):
            raise ValueError(f"--source-path required for source '{source}'")
        source_kwarg_map = {
            "bibtex": "bibtex_path",
            "pdffolder": "folder_path",
            "endnote": "xml_path",
            "mendeley": "db_path",
            "paperpile": "bibtex_path",
        }
        kwarg_name = source_kwarg_map.get(source, "path")
        return create_reference_db(source, **{kwarg_name: args.source_path})

    def _load_source_papers(
        self,
        ref_db: BaseReferenceDB,
        args: argparse.Namespace,
    ) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
        with LogContext(self.logger, f"Loading papers from {ref_db.provider}"):
            all_papers = list(ref_db.get_all_papers())
            if getattr(args, "collection", None):
                if ref_db.provider != "zotero":
                    self.logger.warning(
                        "--collection filter is Zotero-specific; ignoring for source '%s'",
                        ref_db.provider,
                    )
                else:
                    all_papers = [
                        paper
                        for paper in all_papers
                        if any(args.collection in collection for collection in paper.collections)
                    ]
                    self.logger.info(
                        "Collection filter '%s': %d papers matched",
                        args.collection,
                        len(all_papers),
                    )
            papers = [paper for paper in all_papers if paper.has_extractable_source]
            papers_without_pdf = [paper for paper in all_papers if not paper.has_extractable_source]
            if papers_without_pdf:
                self.logger.info(
                    "Found %d papers, %d without local extractable sources (skipped)",
                    len(all_papers),
                    len(papers_without_pdf),
                )
            self.logger.info("Found %d papers with local extractable sources", len(papers))
        return papers, papers_without_pdf

    def _run_classify_only(
        self,
        args: argparse.Namespace,
        config: Config,
        class_store: ClassificationStore,
        current_papers: list[PaperMetadata],
    ) -> int:
        class_index = (
            class_store.load() if not getattr(args, "reclassify", False) else ClassificationIndex()
        )
        self._update_classification_index(
            class_store=class_store,
            class_index=class_index,
            papers=current_papers,
            deleted_paper_ids=[],
            config=config,
            force_reclassify=getattr(args, "reclassify", False),
            refresh_text=getattr(args, "refresh_text", False),
        )
        self._print_classification_report(class_index)
        return 0

    def _update_classification_index(
        self,
        class_store: ClassificationStore,
        class_index: ClassificationIndex,
        papers: list[PaperMetadata],
        deleted_paper_ids: list[str],
        config: Config,
        force_reclassify: bool = False,
        refresh_text: bool = False,
    ) -> None:
        pdf_extractor = PDFExtractor(
            enable_ocr=config.processing.ocr_enabled or config.processing.ocr_on_fail,
        )
        text_cleaner = TextCleaner()
        reusable_snapshots = load_reusable_text_snapshots(
            self.store,
            papers,
            refresh_text=refresh_text,
        )
        for paper in tqdm(papers, desc="Classifying"):
            existing_record = class_index.papers.get(paper.paper_id)
            if (
                not force_reclassify
                and existing_record is not None
                and not self._record_needs_reclassification(existing_record)
            ):
                continue
            text = None
            word_count = None
            page_count = None
            section_markers = None
            source_tier = "metadata_only"
            snapshot = reusable_snapshots.get(paper.paper_id)
            if snapshot is not None:
                snapshot_text = snapshot.get("text")
                if isinstance(snapshot_text, str) and snapshot_text.strip():
                    text = snapshot_text
                    stats = text_cleaner.get_stats(text)
                    word_count = stats.word_count
                    page_count = int(snapshot.get("page_count") or stats.page_count)
                    section_markers = int(
                        snapshot.get("section_markers") or text_cleaner.count_section_markers(text)
                    )
                    source_tier = "stored_snapshot"
            elif paper.pdf_path and Path(paper.pdf_path).exists():
                pdf_path = Path(paper.pdf_path)
                try:
                    raw_text, _ = pdf_extractor.extract_text_with_method(pdf_path)
                    text = text_cleaner.clean(raw_text)
                    stats = text_cleaner.get_stats(text)
                    word_count = stats.word_count
                    page_count = pdf_extractor.get_page_count(pdf_path)
                    section_markers = text_cleaner.count_section_markers(text)
                    source_tier = "cheap_text_pass"
                except Exception as exc:
                    self.logger.warning(
                        "Text extraction failed for %s: %s",
                        paper.title,
                        exc,
                    )
            record = class_store.classify_paper(
                paper,
                text=text,
                word_count=word_count,
                page_count=page_count,
                section_markers=section_markers,
            )
            intent_record = classify_extraction_intent(
                paper,
                text=text,
                word_count=word_count,
                page_count=page_count,
                section_markers=section_markers,
                source_tier=source_tier,
                classified_at=record.classified_at,
                allow_picture_enrichment=(
                    config.processing.opendataloader_hybrid_auto_picture_intents
                    or config.processing.opendataloader_hybrid_enrich_picture_description
                ),
            )
            profile = resolve_hybrid_profile(intent_record.intent, config.processing)
            record.extraction_intent = intent_record.intent.value
            record.intent_confidence = intent_record.confidence
            record.intent_reasons = list(intent_record.reasons)
            record.intent_signals = dict(intent_record.signals)
            record.intent_classified_at = intent_record.classified_at
            record.intent_source_tier = intent_record.source_tier
            record.hybrid_profile_key = profile.profile_key
            class_index.papers[paper.paper_id] = record
        for paper_id in deleted_paper_ids:
            class_index.papers.pop(paper_id, None)
        class_store.save(class_index)

    def _classification_scope_ids(
        self,
        args: argparse.Namespace,
        plan: SyncPlan,
        current_papers_by_id: dict[str, PaperMetadata],
        class_index: ClassificationIndex,
        config: Config,
    ) -> list[str]:
        if getattr(args, "paper", []):
            return sorted(
                paper_id for paper_id in plan.forced_paper_ids if paper_id in current_papers_by_id
            )
        if getattr(args, "reclassify", False):
            return sorted(current_papers_by_id.keys())
        if plan.full_rebuild or not class_index.papers:
            return sorted(current_papers_by_id.keys())
        scope = set(plan.change_set.new_items + plan.change_set.modified_items)
        scope.update(plan.forced_paper_ids)
        scope.update(
            paper_id
            for paper_id in current_papers_by_id
            if paper_id not in class_index.papers
            or self._record_needs_reclassification(class_index.papers[paper_id])
        )
        return sorted(scope)

    @staticmethod
    def _record_needs_reclassification(record: ClassificationRecord) -> bool:
        """Return whether a stored classification record lacks extraction intent."""
        return not bool(record.extraction_intent and record.hybrid_profile_key)

    def _eligible_paper_ids(
        self,
        current_snapshots: dict[str, PaperSnapshot],
        class_index: ClassificationIndex,
        index_all: bool,
    ) -> set[str]:
        if index_all:
            return set(current_snapshots)
        return {
            paper_id
            for paper_id in current_snapshots
            if paper_id in class_index.papers and class_index.papers[paper_id].extractable
        }

    def _scoped_current_ids(
        self,
        args: argparse.Namespace,
        plan: SyncPlan,
        desired_index_ids: set[str],
    ) -> set[str]:
        if getattr(args, "paper", []):
            return set(plan.forced_paper_ids) & set(desired_index_ids)
        if plan.resolved_mode == "full" and not getattr(args, "paper", []):
            return set(desired_index_ids)
        scoped_ids = set(plan.change_set.new_items + plan.change_set.modified_items)
        scoped_ids.update(plan.forced_paper_ids)
        pending_ids = set(plan.pending_work["extraction"].paper_ids)
        if plan.pending_work["extraction"].all:
            pending_ids = set(desired_index_ids)
        scoped_ids.update(pending_ids)
        return scoped_ids & set(desired_index_ids)

    def _extraction_required_ids(
        self,
        args: argparse.Namespace,
        plan: SyncPlan,
        desired_index_ids: set[str],
        existing_extractions: dict[str, dict],
    ) -> set[str]:
        if getattr(args, "paper", []):
            forced_ids = set(plan.forced_paper_ids) & set(desired_index_ids)
            pending_ids = set(plan.pending_work["extraction"].paper_ids) & forced_ids
            if plan.pending_work["extraction"].all:
                pending_ids = set(forced_ids)
            return forced_ids | pending_ids

        pending_ids = set(plan.pending_work["extraction"].paper_ids)
        if plan.pending_work["extraction"].all:
            pending_ids = set(desired_index_ids)

        if plan.resolved_mode == "full" and not getattr(args, "paper", []):
            if plan.requires_full_extraction:
                required = set(desired_index_ids)
            else:
                required = {
                    paper_id
                    for paper_id in desired_index_ids
                    if paper_id not in existing_extractions
                }
        else:
            required = set(plan.change_set.new_items + plan.change_set.modified_items)
            required.update(plan.forced_paper_ids)
            required &= set(desired_index_ids)
        required |= pending_ids & set(desired_index_ids)
        return required

    def _describe_extraction_requirements(
        self,
        args: argparse.Namespace,
        plan: SyncPlan,
        desired_index_ids: set[str],
        existing_extractions: dict[str, dict],
        current_papers_by_id: dict[str, PaperMetadata],
    ) -> list[dict[str, str]]:
        """Describe which papers require extraction and why."""
        required_ids = sorted(
            self._extraction_required_ids(
                args=args,
                plan=plan,
                desired_index_ids=desired_index_ids,
                existing_extractions=existing_extractions,
            )
        )
        pending_ids = set(plan.pending_work["extraction"].paper_ids)
        if plan.pending_work["extraction"].all:
            pending_ids = set(desired_index_ids)

        described: list[dict[str, str]] = []
        for paper_id in required_ids:
            paper = current_papers_by_id.get(paper_id)
            if paper_id in pending_ids:
                reason = "pending extraction work from a previous incomplete run"
            elif plan.resolved_mode == "full" and not getattr(args, "paper", []):
                if plan.requires_full_extraction:
                    reason = "full extraction required because extraction compatibility changed"
                else:
                    reason = "missing stored extraction for full rebuild"
            elif paper_id in plan.forced_paper_ids:
                reason = "explicitly targeted via --paper"
            elif paper_id in plan.change_set.new_items:
                reason = "new paper"
            elif paper_id in plan.change_set.modified_items:
                reason = "modified paper"
            else:
                reason = "fresh extraction required"

            described.append(
                {
                    "paper_id": paper_id,
                    "title": paper.title if paper else paper_id,
                    "zotero_key": paper.zotero_key if paper else "",
                    "reason": reason,
                }
            )
        return described

    def _build_extraction_plan_items(
        self,
        *,
        extraction_required_ids: set[str],
        class_index: ClassificationIndex,
        config: Config,
    ) -> list[ExtractionPlanItem]:
        """Build resolved extraction plan items for the current extraction scope."""
        return build_extraction_plan(
            paper_ids=sorted(extraction_required_ids),
            classification_records=class_index.papers,
            processing=config.processing,
        )

    @staticmethod
    def _format_extraction_plan_lines(
        grouped_plan: dict[str, list[ExtractionPlanItem]],
        current_papers_by_id: dict[str, PaperMetadata],
        limit_per_group: int = 5,
    ) -> list[str]:
        """Format grouped extraction plan details for logs and dry-runs."""
        lines: list[str] = []
        for profile_key, items in sorted(
            grouped_plan.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            lines.append(f"{profile_key}: {len(items)} paper(s)")
            for item in items[:limit_per_group]:
                paper = current_papers_by_id.get(item.paper_id)
                title = paper.title if paper else item.paper_id
                lines.append(
                    f"  {item.paper_id}: {title} [{item.intent.value}, "
                    f"confidence={item.intent_confidence:.2f}]"
                )
            remaining = len(items) - limit_per_group
            if remaining > 0:
                lines.append(f"  ... and {remaining} more")
        return lines

    @staticmethod
    def _format_extraction_requirement_lines(
        requirements: list[dict[str, str]],
        limit: int = 10,
    ) -> list[str]:
        """Format extraction requirement descriptions for logs."""
        lines: list[str] = []
        for requirement in requirements[:limit]:
            suffix = (
                f" (zotero_key={requirement['zotero_key']})"
                if requirement.get("zotero_key")
                else ""
            )
            lines.append(
                f"{requirement['paper_id']}: {requirement['title']}{suffix} "
                f"[{requirement['reason']}]"
            )
        remaining = len(requirements) - limit
        if remaining > 0:
            lines.append(f"... and {remaining} more")
        return lines

    def _embedding_scope_ids(
        self,
        args: argparse.Namespace,
        plan: SyncPlan,
        final_papers: dict[str, dict],
        final_extractions: dict[str, dict],
        failed_extraction_ids: set[str],
    ) -> set[str]:
        desired_ids = set(final_papers) & set(final_extractions)
        if getattr(args, "paper", []):
            pending_ids = set(plan.pending_work["embeddings"].paper_ids) & set(
                plan.forced_paper_ids
            )
            if plan.pending_work["embeddings"].all:
                pending_ids = set(plan.forced_paper_ids)
            forced_ids = set(plan.forced_paper_ids) & desired_ids
            forced_ids -= failed_extraction_ids
            return forced_ids | pending_ids
        pending_ids = set(plan.pending_work["embeddings"].paper_ids)
        if plan.pending_work["embeddings"].all:
            pending_ids = set(desired_ids)
        if plan.clear_vector_store:
            return desired_ids
        changed_ids = set(plan.change_set.new_items + plan.change_set.modified_items)
        extraction_pending_ids = set(plan.pending_work["extraction"].paper_ids)
        if plan.pending_work["extraction"].all:
            extraction_pending_ids = set(desired_ids)
        changed_ids.update(plan.forced_paper_ids)
        changed_ids.update(extraction_pending_ids)
        changed_ids &= desired_ids
        changed_ids -= failed_extraction_ids
        return changed_ids | pending_ids

    def _apply_retry_resume_limit_skip(
        self,
        args: argparse.Namespace,
        papers: list[PaperMetadata],
        checkpoint_mgr: CheckpointManager,
    ) -> list[PaperMetadata]:
        papers_to_extract = list(papers)
        if getattr(args, "retry_failed", False):
            state = checkpoint_mgr.load()
            if state:
                failed_ids = set(checkpoint_mgr.get_failed_ids())
                if failed_ids:
                    checkpoint_mgr.clear_failed()
                    papers_to_extract = [
                        paper for paper in papers_to_extract if paper.paper_id in failed_ids
                    ]
            else:
                papers_to_extract = []
        elif getattr(args, "resume", False):
            state = checkpoint_mgr.load()
            if state:
                processed_ids = set(state.processed_ids)
                failed_ids = set(checkpoint_mgr.get_failed_ids())
                papers_to_extract = [
                    paper
                    for paper in papers_to_extract
                    if paper.paper_id not in processed_ids and paper.paper_id not in failed_ids
                ]

        skip_ids = set(getattr(args, "skip_paper", []) or [])
        if skip_ids:
            papers_to_extract = [
                paper for paper in papers_to_extract if paper.paper_id not in skip_ids
            ]
        if getattr(args, "limit", None) and not getattr(args, "paper", []):
            papers_to_extract = papers_to_extract[: args.limit]
            self.logger.info("Limited extraction scope to %d papers", len(papers_to_extract))
        return papers_to_extract

    def _show_skipped(self, papers_without_pdf: list[PaperMetadata]) -> None:
        if papers_without_pdf:
            print(f"\nPapers without local extractable sources ({len(papers_without_pdf)}):")
            print("-" * 70)
            for paper in papers_without_pdf:
                print(f"  {paper.paper_id}: {paper.title[:60]}...")
                if paper.authors:
                    print(f"           {paper.author_string}")
            print("-" * 70)
            print(
                f"Total: {len(papers_without_pdf)} papers cannot be extracted "
                "(no local source attachment)"
            )
        else:
            print("\nAll papers have local extractable sources available.")

    def _show_failed(self, checkpoint_mgr: CheckpointManager) -> None:
        state = checkpoint_mgr.load()
        if state and state.failed_items:
            print(f"\nFailed papers ({len(state.failed_items)}):")
            for item in state.failed_items:
                print(f"  - {item.item_id}")
                print(f"    Error: {item.error_type}: {item.error_message}")
                if item.retry_count > 0:
                    print(f"    Retries: {item.retry_count}")
        else:
            print("\nNo failed papers recorded")

    def _show_doi_overlap(
        self,
        current_papers: list[PaperMetadata],
        desired_index_ids: set[str],
    ) -> None:
        scoped_papers = [paper for paper in current_papers if paper.paper_id in desired_index_ids]
        analysis = analyze_doi_overlap(scoped_papers, self.index_dir)
        print(f"\n{'=' * 60}")
        print("DOI Overlap Analysis")
        print(f"{'=' * 60}")
        print("\nExisting index:")
        print(f"  Papers with DOIs: {analysis['existing_index_dois']}")
        print("\nNew source:")
        print(f"  Total papers (with PDFs): {analysis['new_papers_total']}")
        print(f"  With DOIs: {analysis['new_with_doi']}")
        print(f"  Without DOIs: {analysis['new_without_doi']}")
        print("\nOverlap analysis:")
        print(f"  Duplicates (DOI match): {analysis['duplicates_by_doi']}")
        print(f"  Genuinely new (with DOI): {analysis['genuinely_new_with_doi']}")
        print(f"  Total to process: {len(analysis['new_papers_filtered'])}")

    def _clear_or_delete_vectors_for_pending(
        self,
        plan: SyncPlan,
        delete_paper_ids: list[str],
    ) -> None:
        with VectorStore(self.index_dir / "chroma") as vector_store:
            if plan.clear_vector_store:
                vector_store.clear()
                return
            if delete_paper_ids:
                vector_store.delete_papers(delete_paper_ids)

    def _create_staged_chroma_dir(self) -> Path:
        """Create a temporary Chroma directory for a staged full rebuild."""
        return Path(tempfile.mkdtemp(prefix="chroma_staging_", dir=self.index_dir))

    def _commit_staged_chroma_dir(self, staged_dir: Path) -> None:
        """Replace the live Chroma directory with a staged rebuild."""
        target_dir = self.index_dir / "chroma"
        backup_dir = Path(tempfile.mkdtemp(prefix="chroma_backup_", dir=self.index_dir))
        shutil.rmtree(backup_dir)
        try:
            if target_dir.exists():
                shutil.move(str(target_dir), str(backup_dir))
            shutil.move(str(staged_dir), str(target_dir))
        except Exception:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            if backup_dir.exists():
                shutil.move(str(backup_dir), str(target_dir))
            raise
        finally:
            if staged_dir.exists():
                shutil.rmtree(staged_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)

    def _print_plan(
        self,
        plan: SyncPlan,
        current_snapshots: dict[str, PaperSnapshot],
    ) -> None:
        print("\nSync plan:")
        print(f"  Requested mode: {plan.requested_mode}")
        print(f"  Resolved mode: {plan.resolved_mode}")
        print(f"  Current source papers: {len(current_snapshots)}")
        print(f"  Changes: {plan.change_set.summary()}")
        if plan.forced_paper_ids:
            print(f"  Forced targets: {len(plan.forced_paper_ids)}")
        if plan.advisories:
            print("  Advisories:")
            for advisory in plan.advisories:
                print(f"    - {advisory}")
        if plan.reasons:
            print("  Reasons:")
            for reason in plan.reasons:
                print(f"    - {reason}")

    def _requested_sync_mode(self, args: argparse.Namespace) -> str:
        sync_mode = getattr(args, "sync_mode", "auto")
        if getattr(args, "rebuild_embeddings", False):
            if sync_mode not in ("auto", "full"):
                raise ValueError("--rebuild-embeddings conflicts with --sync-mode update")
            if not getattr(args, "paper", []):
                self.logger.warning(
                    "--rebuild-embeddings is deprecated; use --sync-mode full instead",
                )
                return "full"
            self.logger.warning(
                "--rebuild-embeddings with --paper now performs a targeted refresh",
            )
        return sync_mode

    def _source_scope_info(
        self,
        ref_db: BaseReferenceDB,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        payload = {
            "provider": ref_db.provider,
            "source_path": str(ref_db.source_path),
            "collection": getattr(args, "collection", None),
            "index_all": bool(getattr(args, "index_all", False)),
        }
        return {**payload, "fingerprint": fingerprint_payload(payload)}

    def _classification_policy_info(
        self,
        args: argparse.Namespace,
        config: Config,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": CLASSIFICATION_POLICY_VERSION,
            "extraction_intent_schema_version": EXTRACTION_INTENT_SCHEMA_VERSION,
            "min_publication_words": config.processing.min_publication_words,
            "min_publication_pages": config.processing.min_publication_pages,
            "index_all": bool(getattr(args, "index_all", False)),
        }
        return {**payload, "fingerprint": fingerprint_payload(payload)}

    def _extraction_info(self, args: argparse.Namespace, config: Config) -> dict[str, Any]:
        provider = getattr(args, "provider", None) or config.extraction.provider
        provider_settings = config.extraction.get_provider_settings(provider)
        active_profile = get_default_dimension_registry().active_profile
        payload = {
            "schema_version": EXTRACTION_SCHEMA_VERSION,
            "provider": provider,
            "mode": getattr(args, "mode", None) or provider_settings.mode or config.extraction.mode,
            "model": getattr(args, "model", None)
            or provider_settings.model
            or config.extraction.model,
            "summary_model": getattr(args, "summary_model", None),
            "methodology_model": getattr(args, "methodology_model", None),
            "profile_id": active_profile.profile_id,
            "profile_version": active_profile.version,
            "profile_fingerprint": active_profile.fingerprint,
            "opendataloader_mode": config.processing.opendataloader_mode,
            "opendataloader_hybrid_enabled": config.processing.opendataloader_hybrid_enabled,
            "opendataloader_hybrid_backend": config.processing.opendataloader_hybrid_backend,
            "opendataloader_hybrid_fallback": (config.processing.opendataloader_hybrid_fallback),
        }
        return {**payload, "fingerprint": fingerprint_payload(payload)}

    def _embedding_info(self, config: Config) -> dict[str, Any]:
        fingerprint_payload_fields = {
            "model": config.embeddings.model,
            "dimension": config.embeddings.dimension,
            "backend": config.embeddings.backend,
            "query_prefix": config.embeddings.query_prefix,
            "document_prefix": config.embeddings.document_prefix,
        }
        return {
            **fingerprint_payload_fields,
            "batch_size": config.embeddings.batch_size,
            "fingerprint": fingerprint_payload(fingerprint_payload_fields),
        }

    def _structured_store_consistent(
        self,
        existing_papers: dict[str, dict],
        existing_extractions: dict[str, dict],
        manifest: IndexManifest,
    ) -> bool:
        if set(existing_extractions) - set(existing_papers):
            return False
        if manifest.paper_snapshots and not set(existing_papers).issubset(
            set(manifest.paper_snapshots)
        ):
            return False
        return True

    def _validate_vector_store(self) -> str | None:
        chroma_dir = self.index_dir / "chroma"
        try:
            with VectorStore(chroma_dir) as vector_store:
                vector_store.count()
        except Exception as exc:
            return f"Vector store is unreadable: {exc}"
        return None

    def _validate_chunk_types(self) -> str | None:
        chroma_dir = self.index_dir / "chroma"
        try:
            with VectorStore(chroma_dir) as vector_store:
                observed: set[str] = set()
                offset = 0
                page_size = 500
                while True:
                    results = vector_store.collection.get(
                        include=["metadatas"],
                        limit=page_size,
                        offset=offset,
                    )
                    metadatas = results.get("metadatas") or []
                    if not metadatas:
                        break
                    observed.update(
                        meta.get("chunk_type")
                        for meta in metadatas
                        if meta and meta.get("chunk_type")
                    )
                    offset += len(metadatas)
        except Exception as exc:
            return f"Could not inspect chunk types: {exc}"
        invalid = sorted(
            chunk_type
            for chunk_type in observed
            if chunk_type not in set(CHUNK_TYPES) and not chunk_type.startswith("dim_")
        )
        if invalid:
            return f"Vector store contains legacy chunk types: {', '.join(invalid[:8])}"
        return None

    def _resolve_target_ids(
        self,
        target_keys: list[str],
        current_snapshots: dict[str, PaperSnapshot],
        manifest: IndexManifest | None,
        existing_papers: dict[str, dict],
    ) -> tuple[set[str], set[str], set[str]]:
        del manifest
        if not target_keys:
            return set(), set(), set()
        current_by_zotero: dict[str, set[str]] = {}
        for snapshot in current_snapshots.values():
            current_by_zotero.setdefault(snapshot.zotero_key, set()).add(snapshot.paper_id)
        stored_by_zotero: dict[str, set[str]] = {}
        for paper_id, paper_data in existing_papers.items():
            zotero_key = paper_data.get("zotero_key")
            if zotero_key:
                stored_by_zotero.setdefault(zotero_key, set()).add(paper_id)

        current_ids: set[str] = set()
        deleted_ids: set[str] = set()
        unmatched: set[str] = set()
        for target in target_keys:
            if target in current_snapshots:
                current_ids.add(target)
                continue
            if target in existing_papers:
                deleted_ids.add(target)
                continue
            if target in current_by_zotero:
                current_ids.update(current_by_zotero[target])
                continue
            if target in stored_by_zotero:
                deleted_ids.update(stored_by_zotero[target])
                continue
            unmatched.add(target)
        return current_ids, deleted_ids, unmatched

    def _normalize_extractions(
        self,
        extractions: dict[str, dict],
    ) -> dict[str, SemanticAnalysis]:
        normalized = {}
        for paper_id, value in extractions.items():
            inner = value.get("extraction", value) if isinstance(value, dict) else value
            normalized[paper_id] = SemanticAnalysis(**inner)
        return normalized

    def _write_metadata(
        self,
        manifest: IndexManifest,
        final_papers: dict[str, dict],
        final_extractions: dict[str, dict],
        stage_times: dict[str, float],
        failed_paper_ids: list[str],
        summary: dict[str, Any],
    ) -> None:
        metadata = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "index_name": "LITRIS_Index",
            "generated_at": datetime.now().isoformat(),
            "last_full_build": manifest.last_full_build,
            "last_update": manifest.last_successful_sync,
            "sync": {
                "requested_mode": manifest.last_run.get("requested_mode"),
                "resolved_mode": manifest.last_run.get("resolved_mode"),
                "reasons": manifest.last_run.get("reasons", []),
                "advisories": manifest.last_run.get("advisories", []),
            },
            "source_scope": manifest.source_scope,
            "classification_policy": manifest.classification_policy,
            "processing": {
                "extraction_model": manifest.extraction.get("model"),
                "extraction_mode": manifest.extraction.get("mode"),
                "extraction_provider": manifest.extraction.get("provider"),
                "dimension_profile_id": manifest.extraction.get("profile_id"),
                "dimension_profile_version": manifest.extraction.get("profile_version"),
                "dimension_profile_fingerprint": manifest.extraction.get("profile_fingerprint"),
                "embedding_model": manifest.embedding.get("model"),
                "embedding_backend": manifest.embedding.get("backend"),
                "embedding_batch_size_setting": manifest.embedding.get("batch_size"),
                "embedding_batch_size_resolved": manifest.last_run.get(
                    "embedding_batch_size_resolved"
                ),
                "stage_durations_seconds": stage_times,
            },
            "statistics": {
                "total_papers": len(final_papers),
                "total_extractions": len(final_extractions),
                "summary_total_papers": summary.get("total_papers"),
                "summary_total_extractions": summary.get("total_extractions"),
            },
            "failed_papers": sorted(set(failed_paper_ids)),
        }
        safe_write_json(self.index_dir / "metadata.json", metadata)

    def _paper_from_index_dict(self, paper_id: str, paper_data: dict[str, Any]) -> PaperMetadata:
        return PaperMetadata(
            paper_id=paper_id,
            zotero_key=paper_data.get("zotero_key", paper_id.split("_")[0]),
            zotero_item_id=paper_data.get("zotero_item_id", 0),
            item_type=paper_data.get("item_type", "journalArticle"),
            title=paper_data.get("title", "Unknown"),
            authors=paper_data.get("authors", []),
            publication_year=paper_data.get("publication_year"),
            publication_date=paper_data.get("publication_date"),
            journal=paper_data.get("journal"),
            doi=paper_data.get("doi"),
            abstract=paper_data.get("abstract"),
            collections=paper_data.get("collections", []),
            tags=paper_data.get("tags", []),
            pdf_path=paper_data.get("pdf_path"),
            pdf_attachment_key=paper_data.get("pdf_attachment_key"),
            source_path=paper_data.get("source_path"),
            source_attachment_key=paper_data.get("source_attachment_key"),
            source_media_type=paper_data.get("source_media_type"),
            date_added=paper_data.get("date_added") or "2020-01-01T00:00:00",
            date_modified=paper_data.get("date_modified") or "2020-01-01T00:00:00",
        )

    def _print_classification_report(self, index: ClassificationIndex) -> None:
        stats = index.stats
        total = stats.get("total", 0)
        if total == 0:
            print("No papers classified.")
            return
        print("\nClassification summary:")
        for doc_type, count in sorted(stats.get("by_type", {}).items(), key=lambda item: -item[1]):
            pct = count / total * 100
            print(f"  {doc_type:<22} {count:>5} ({pct:.0f}%)")
        ext = stats.get("extractable_count", 0)
        non_ext = stats.get("non_extractable_count", 0)
        print(f"\nExtractable: {ext}  |  Skippable: {non_ext}")
        by_intent, by_profile, confidence_buckets = summarize_intents(index.papers)
        if by_intent:
            print("\nExtraction intent summary:")
            for intent, count in sorted(by_intent.items(), key=lambda item: (-item[1], item[0])):
                pct = count / total * 100
                print(f"  {intent:<28} {count:>5} ({pct:.0f}%)")
        if by_profile:
            print("\nResolved extraction profiles:")
            for profile_key, count in sorted(
                by_profile.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                print(f"  {profile_key:<48} {count:>5}")
        print(
            "\nIntent confidence:"
            f" low={confidence_buckets['low']}"
            f" medium={confidence_buckets['medium']}"
            f" high={confidence_buckets['high']}"
        )

    def _print_cost_estimate(self, estimate: dict[str, Any]) -> None:
        print("\nEstimating extraction cost...")
        print(f"\nPapers with PDFs: {estimate['papers_with_pdf']}")
        if estimate.get("papers_cached", 0) > 0:
            print(f"Papers cached (will skip): {estimate['papers_cached']}")
            print(f"Papers to extract: {estimate['papers_to_extract']}")
        if "average_text_length" in estimate:
            print(f"Average text length: {estimate['average_text_length']:,} chars")
            print(f"Cost per paper: ${estimate['estimated_cost_per_paper']:.4f}")
            print(f"Total estimated cost: ${estimate['estimated_total_cost']:.2f}")
            print(f"Model: {estimate['model']}")
        else:
            print(f"Note: {estimate.get('note', 'Unknown')}")

    def _time_stage(self, stage_name: str, stage_times: dict[str, float]):
        orchestrator = self

        class _StageTimer:
            def __enter__(self_inner):
                self_inner.start = perf_counter()
                orchestrator.logger.info("Starting %s stage", stage_name)
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                stage_times[stage_name] = round(perf_counter() - self_inner.start, 3)
                if exc:
                    orchestrator.logger.error("%s stage failed: %s", stage_name, exc)
                else:
                    orchestrator.logger.info(
                        "Finished %s stage in %.2fs",
                        stage_name,
                        stage_times[stage_name],
                    )
                return False

        return _StageTimer()
