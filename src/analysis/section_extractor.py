"""Section extractor orchestrating PDF extraction and profile-driven LLM analysis."""

import hashlib
import json
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

from src.analysis.extraction_intent_classifier import ExtractionPlanItem
from src.analysis.base_llm import BaseLLMClient, ExtractionMode
from src.analysis.coverage import score_coverage
from src.analysis.dimensions import get_default_dimension_registry, get_dimension_value
from src.analysis.document_classifier import classify as classify_document
from src.analysis.document_types import DocumentType
from src.analysis.llm_factory import create_llm_client
from src.analysis.schemas import ExtractionResult, SemanticAnalysis
from src.analysis.semantic_prompts import (
    SEMANTIC_PROMPT_VERSION,
    build_pass_user_prompt,
    get_pass_definitions,
)
from src.extraction.cascade import ExtractionCascade
from src.extraction.opendataloader_extractor import OpenDataLoaderHybridConfig
from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.text_cleaner import TextCleaner
from src.utils.logging_config import LogContext, get_logger
from src.utils.run_control import PauseRequested, RunControlPoller
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

class ExtractionCache:
    """Cache for extraction results based on content hash."""

    def __init__(self, cache_dir: Path):
        """Initialize extraction cache.

        Args:
            cache_dir: Base cache directory.
        """
        self.cache_dir = cache_dir / "extractions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def compute_content_hash(
        self, pdf_path: Path | None, model: str, prompt_version: str = SEMANTIC_PROMPT_VERSION,
        pass_number: int | None = None,
        profile_fingerprint: str | None = None,
        source_fingerprint: str | None = None,
    ) -> str:
        """Compute content hash for cache key.

        Uses either a supplied source fingerprint or file modification time
        and size to detect changes without reading the entire file. Includes
        model, prompt version, active profile fingerprint, and optional pass
        number to invalidate cache when extraction logic changes.

        Args:
            pdf_path: Path to PDF file when hashing directly from the source file.
            model: Model identifier.
            prompt_version: Version string for extraction prompts.
            pass_number: Pass number (1-6) for per-pass caching. None for full result.
            profile_fingerprint: Fingerprint of the active dimension profile.
            source_fingerprint: Explicit fingerprint for a canonical text snapshot.

        Returns:
            16-character hex hash string.
        """
        hasher = hashlib.sha256()
        if source_fingerprint:
            hasher.update(f"source={source_fingerprint}".encode())
        elif pdf_path is not None:
            stat = pdf_path.stat()
            hasher.update(f"{stat.st_mtime}:{stat.st_size}".encode())
        else:
            hasher.update(b"source=none")
        hasher.update(f":{model}:{prompt_version}".encode())
        if profile_fingerprint:
            hasher.update(f":profile={profile_fingerprint}".encode())
        if pass_number is not None:
            hasher.update(f":pass{pass_number}".encode())
        return hasher.hexdigest()[:16]

    def _get_cache_path(self, paper_id: str, content_hash: str) -> Path:
        """Get cache file path for a paper."""
        return self.cache_dir / f"{paper_id}_{content_hash}.json"

    def get(self, paper_id: str, content_hash: str) -> dict | None:
        """Retrieve cached data.

        Args:
            paper_id: Paper identifier.
            content_hash: Content hash for cache key.

        Returns:
            Cached data dict or None if not found/invalid.
        """
        cache_path = self._get_cache_path(paper_id, content_hash)

        if not cache_path.exists():
            return None

        try:
            with self._lock:
                with open(cache_path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Cache read failed for {paper_id}: {e}")
            return None

    def get_extraction_result(
        self, paper_id: str, content_hash: str
    ) -> ExtractionResult | None:
        """Retrieve cached full extraction result.

        Args:
            paper_id: Paper identifier.
            content_hash: Content hash for cache key.

        Returns:
            Cached ExtractionResult or None if not found/invalid.
        """
        data = self.get(paper_id, content_hash)
        if data is None:
            return None

        try:
            if "extraction" in data and data["extraction"]:
                data["extraction"] = SemanticAnalysis(**data["extraction"])
            return ExtractionResult(**data)
        except Exception as e:
            logger.debug(f"Cache deserialization failed for {paper_id}: {e}")
            return None

    def set(self, paper_id: str, content_hash: str, data: dict) -> None:
        """Store data in cache.

        Args:
            paper_id: Paper identifier.
            content_hash: Content hash for cache key.
            data: Data dict to cache.
        """
        cache_path = self._get_cache_path(paper_id, content_hash)

        try:
            with self._lock:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            logger.debug(f"Cached data for {paper_id}")
        except Exception as e:
            logger.warning(f"Cache write failed for {paper_id}: {e}")

    def set_extraction_result(
        self, paper_id: str, content_hash: str, result: ExtractionResult
    ) -> None:
        """Store full extraction result in cache.

        Args:
            paper_id: Paper identifier.
            content_hash: Content hash for cache key.
            result: Extraction result to cache.
        """
        if not result.success:
            return

        data = {
            "paper_id": result.paper_id,
            "success": result.success,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
            "model_used": result.model_used,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

        if result.extraction:
            data["extraction"] = result.extraction.model_dump()

        self.set(paper_id, content_hash, data)

    def clear(self) -> int:
        """Clear all cached extractions.

        Returns:
            Number of cache entries removed.
        """
        count = 0
        with self._lock:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception:
                    pass
        logger.info(f"Cleared {count} cached extractions")
        return count


@dataclass
class ExtractionStats:
    """Statistics for extraction run."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    cached: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.successful / self.total

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        processed = self.successful + self.cached
        if processed == 0:
            return 0.0
        return self.cached / processed


class SectionExtractor:
    """Orchestrates extraction of structured content from papers using pass-based analysis."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        provider: str = "anthropic",
        mode: ExtractionMode = "api",
        model: str | None = None,
        max_tokens: int = 8192,
        timeout: int | None = None,
        min_text_length: int = 100,
        ocr_on_fail: bool = True,
        skip_non_publications: bool = False,
        min_publication_words: int = 500,
        min_publication_pages: int = 2,
        min_section_hits: int = 0,
        ocr_enabled: bool = False,
        ocr_config: dict | None = None,
        use_cache: bool = True,
        parallel_workers: int = 1,
        reasoning_effort: str | None = None,
        effort: str | None = None,
        run_control_path: Path | None = None,
        cascade_enabled: bool = True,
        companion_dir: Path | None = None,
        arxiv_enabled: bool = True,
        opendataloader_enabled: bool = True,
        opendataloader_mode: str = "fast",
        opendataloader_hybrid_config: OpenDataLoaderHybridConfig | None = None,
        opendataloader_hybrid_fallback: bool = False,
        marker_enabled: bool = True,
    ):
        """Initialize section extractor.

        Args:
            cache_dir: Directory for PDF text cache.
            provider: LLM provider ('anthropic' or 'openai').
            mode: LLM extraction mode.
            model: Model to use (None uses provider default).
            max_tokens: Maximum response tokens.
            timeout: Request timeout in seconds (defaults to provider/client default).
            min_text_length: Minimum text length to attempt extraction.
            ocr_on_fail: Attempt OCR when initial text check fails.
            skip_non_publications: Skip likely non-publication attachments.
            min_publication_words: Minimum words for publication heuristic.
            min_publication_pages: Minimum pages for publication heuristic.
            min_section_hits: Minimum section markers for publication heuristic.
            ocr_enabled: Enable OCR fallback for scanned PDFs.
            ocr_config: Optional OCR handler config (tesseract_cmd, poppler_path, dpi, lang).
            use_cache: Enable extraction caching to skip unchanged papers.
            parallel_workers: Number of parallel workers for CLI mode (1 = sequential).
            reasoning_effort: For OpenAI GPT-5.2: none/low/medium/high/xhigh.
            effort: Claude CLI effort level for extended thinking (low/medium/high).
            run_control_path: Optional cooperative pause-control file path.
        """
        self.pdf_extractor = PDFExtractor(
            cache_dir=cache_dir,
            enable_ocr=ocr_enabled or ocr_on_fail,
            ocr_config=ocr_config,
        )
        self.text_cleaner = TextCleaner()

        # Extraction cascade (wraps pdf_extractor with higher-quality tiers)
        self.cascade: ExtractionCascade | None = None
        if cascade_enabled:
            self.cascade = ExtractionCascade(
                pdf_extractor=self.pdf_extractor,
                enable_arxiv=arxiv_enabled,
                enable_opendataloader=opendataloader_enabled,
                enable_marker=marker_enabled,
                min_words=min_text_length,
                companion_dir=companion_dir,
                opendataloader_mode=opendataloader_mode,
                opendataloader_hybrid=opendataloader_hybrid_config,
                opendataloader_hybrid_fallback=opendataloader_hybrid_fallback,
            )

        self.run_control = RunControlPoller(run_control_path) if run_control_path else None

        # Create LLM client using factory
        self.llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            mode=mode,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout if timeout is not None else 120,
            reasoning_effort=reasoning_effort,
            effort=effort,
            run_control=self.run_control,
        )
        self.max_tokens = max_tokens
        self.timeout = getattr(self.llm_client, "timeout", 120)
        self.reasoning_effort = reasoning_effort
        self.min_text_length = min_text_length
        self.ocr_on_fail = ocr_on_fail
        self.skip_non_publications = skip_non_publications
        self.min_publication_words = min_publication_words
        self.min_publication_pages = min_publication_pages
        self.min_section_hits = min_section_hits
        self.model = self.llm_client.model
        self.mode = mode
        self.provider = provider
        self.parallel_workers = parallel_workers
        self.use_cache = use_cache
        active_profile = get_default_dimension_registry().active_profile
        self.dimension_profile_id = active_profile.profile_id
        self.dimension_profile_version = active_profile.version
        self.dimension_profile_fingerprint = active_profile.fingerprint

        # Initialize extraction cache
        if cache_dir and use_cache:
            self.extraction_cache = ExtractionCache(cache_dir)
        else:
            self.extraction_cache = None

    def _raise_if_pause_requested(self, context: str) -> None:
        """Raise PauseRequested when the active run has been asked to pause."""
        if self.run_control is not None:
            self.run_control.raise_if_pause_requested(context)

    def clear_pause_request(self) -> bool:
        """Clear the cooperative pause request after a graceful stop."""
        if self.run_control is None:
            return False
        return self.run_control.clear()

    def _resolve_pass_plan(
        self,
        section_ids: list[str] | None = None,
    ) -> list[tuple[int, str, list[tuple[str, str]]]]:
        """Return the pass plan for a full or section-scoped extraction."""

        pass_definitions = get_pass_definitions()
        if not section_ids:
            return [
                (pass_num, pass_label, pass_questions)
                for pass_num, (pass_label, pass_questions) in enumerate(
                    pass_definitions,
                    start=1,
                )
            ]

        registry = get_default_dimension_registry()
        profile = registry.active_profile
        requested = list(dict.fromkeys(section_ids))
        section_alias_map = {
            alias: section.id
            for section in profile.ordered_sections
            for alias in [section.id, *section.aliases]
        }
        resolved_section_ids: list[str] = []
        invalid_sections: list[str] = []
        for section_id in requested:
            resolved = section_alias_map.get(section_id)
            if resolved is None:
                invalid_sections.append(section_id)
            elif resolved not in resolved_section_ids:
                resolved_section_ids.append(resolved)

        if invalid_sections:
            raise ValueError(
                "Unknown dimension section(s): " + ", ".join(sorted(invalid_sections))
            )

        selected = set(resolved_section_ids)
        return [
            (pass_num, pass_label, pass_questions)
            for pass_num, ((pass_label, pass_questions), section) in enumerate(
                zip(pass_definitions, profile.ordered_sections, strict=False),
                start=1,
            )
            if section.id in selected
        ]

    def extract_paper(
        self,
        paper: PaperMetadata,
        check_cache: bool = True,
        section_ids: list[str] | None = None,
        text_snapshot: dict[str, object] | None = None,
        extraction_plan: ExtractionPlanItem | None = None,
        skip_llm: bool = False,
    ) -> tuple[ExtractionResult, bool]:
        """Extract structured content from a single paper.

        Args:
            paper: Paper metadata including PDF path.
            check_cache: Whether to check cache before extracting.
            section_ids: Optional section IDs to extract. When omitted, run the
                full active profile.
            text_snapshot: Optional canonical full-text snapshot to reuse
                instead of re-running the extraction cascade.

        Returns:
            Tuple of (ExtractionResult, from_cache).
        """
        import time

        with LogContext(logger, f"Extracting paper: {paper.title[:50]}..."):
            started_at = time.time()
            snapshot_text = None
            snapshot_hash = None
            if text_snapshot:
                raw_snapshot_text = text_snapshot.get("text")
                if isinstance(raw_snapshot_text, str) and raw_snapshot_text.strip():
                    snapshot_text = raw_snapshot_text
                    raw_snapshot_hash = text_snapshot.get("sha256")
                    if isinstance(raw_snapshot_hash, str) and raw_snapshot_hash:
                        snapshot_hash = raw_snapshot_hash
                    else:
                        snapshot_hash = hashlib.sha256(
                            snapshot_text.encode("utf-8")
                        ).hexdigest()

            # Check for source text
            if snapshot_text is None and (not paper.pdf_path or not paper.pdf_path.exists()):
                logger.warning(f"No local extractable source available for paper {paper.paper_id}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error="No local extractable source available",
                ), False

            # Check full-result cache
            full_hash = None
            use_full_result_cache = not section_ids and not skip_llm
            if check_cache and self.extraction_cache and use_full_result_cache:
                full_hash = self.extraction_cache.compute_content_hash(
                    paper.pdf_path,
                    self.model,
                    profile_fingerprint=self.dimension_profile_fingerprint,
                    source_fingerprint=snapshot_hash,
                )
                cached_result = self.extraction_cache.get_extraction_result(
                    paper.paper_id, full_hash
                )
                if cached_result:
                    logger.info(f"Using cached extraction for {paper.paper_id}")
                    return cached_result, True

            snapshot_payload: dict[str, object] | None = None
            planned_intent = extraction_plan.intent.value if extraction_plan else None
            planned_profile_key = (
                extraction_plan.profile.profile_key if extraction_plan else None
            )

            # Extract text from PDF (via cascade or direct) unless a canonical
            # snapshot is already available for this exact source state.
            cascade_result = None
            if snapshot_text is not None:
                text = snapshot_text
                method = str(
                    text_snapshot.get("extraction_method")
                    or text_snapshot.get("source")
                    or "stored_snapshot"
                )
                preserve_md = bool(text_snapshot.get("is_markdown", False))
                is_cleaned = bool(text_snapshot.get("is_cleaned", False))
                snapshot_payload = dict(text_snapshot)
                snapshot_payload["should_persist"] = False
            else:
                try:
                    if self.cascade:
                        cascade_result = self.cascade.extract_text(
                            paper.pdf_path,
                            doi=getattr(paper, "doi", None),
                            url=getattr(paper, "url", None),
                        )
                        text = cascade_result.text
                        method = cascade_result.method
                    else:
                        text, method = self.pdf_extractor.extract_text_with_method(
                            paper.pdf_path
                        )
                    preserve_md = cascade_result.is_markdown if cascade_result else False
                    is_cleaned = False
                except Exception as e:
                    logger.error(f"PDF extraction failed: {e}")
                    return ExtractionResult(
                        paper_id=paper.paper_id,
                        success=False,
                        error=f"PDF extraction failed: {e}",
                    ), False

            def evaluate_text(cleaned_text: str) -> tuple[bool, list[str]]:
                """Evaluate text for extraction eligibility."""
                valid = self.text_cleaner.is_valid_extraction(
                    cleaned_text, min_words=self.min_text_length
                )
                reasons: list[str] = []
                if self.skip_non_publications:
                    stats = self.text_cleaner.get_stats(cleaned_text)
                    if (
                        self.min_publication_words > 0
                        and stats.word_count < self.min_publication_words
                    ):
                        reasons.append(
                            f"word_count<{self.min_publication_words} ({stats.word_count})"
                        )
                    if (
                        self.min_publication_pages > 0
                        and stats.page_count < self.min_publication_pages
                    ):
                        reasons.append(
                            f"page_count<{self.min_publication_pages} ({stats.page_count})"
                        )
                    if self.min_section_hits > 0:
                        section_hits = self.text_cleaner.count_section_markers(
                            cleaned_text
                        )
                        if section_hits < self.min_section_hits:
                            reasons.append(
                                f"section_hits<{self.min_section_hits} ({section_hits})"
                            )
                return valid, reasons

            # Clean text (preserve markdown for companion/marker tiers). Stored
            # snapshots are already canonical cleaned text and should not be
            # re-normalized unless explicitly marked otherwise.
            if not is_cleaned:
                text = self.text_cleaner.clean(text, preserve_markdown=preserve_md)
            valid_text, non_pub_reasons = evaluate_text(text)

            pdf_size = None
            pdf_mtime_ns = None
            if paper.pdf_path and paper.pdf_path.exists():
                try:
                    stat = paper.pdf_path.stat()
                    pdf_size = int(stat.st_size)
                    pdf_mtime_ns = int(stat.st_mtime_ns)
                except OSError:
                    pdf_size = None
                    pdf_mtime_ns = None

            if snapshot_payload is None:
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                snapshot_payload = {
                    "paper_id": paper.paper_id,
                    "text": text,
                    "sha256": text_hash,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "captured_at": datetime.now().isoformat(),
                    "source": "cascade",
                    "extraction_method": (
                        cascade_result.method if cascade_result else method
                    ),
                    "tiers_attempted": (
                        list(cascade_result.tiers_attempted) if cascade_result else [method]
                    ),
                    "is_markdown": preserve_md,
                    "is_cleaned": True,
                    "is_truncated_for_llm": False,
                    "pdf_path": str(paper.pdf_path) if paper.pdf_path else None,
                    "pdf_size": pdf_size,
                    "pdf_mtime_ns": pdf_mtime_ns,
                    "should_persist": True,
                }
            if planned_intent:
                snapshot_payload["planned_extraction_intent"] = planned_intent
            if planned_profile_key:
                snapshot_payload["planned_profile_key"] = planned_profile_key
            if cascade_result and cascade_result.tier_errors:
                snapshot_payload["extraction_diagnostics"] = dict(
                    cascade_result.tier_errors
                )

            extraction_diagnostics: dict[str, str] = {}
            if snapshot_payload is not None:
                raw_diagnostics = snapshot_payload.get("extraction_diagnostics")
                if isinstance(raw_diagnostics, dict):
                    extraction_diagnostics = {
                        str(key): str(value)
                        for key, value in raw_diagnostics.items()
                        if value is not None
                    }

            # OCR fallback for low-quality text
            if (
                self.ocr_on_fail
                and self.pdf_extractor.ocr_handler
                and method == "pymupdf"
                and snapshot_text is None
                and (not valid_text or non_pub_reasons)
            ):
                logger.info(
                    f"Text check failed for {paper.paper_id}; attempting OCR fallback"
                )
                try:
                    ocr_result = self.pdf_extractor.ocr_handler.extract_text(
                        paper.pdf_path
                    )
                    text = self.text_cleaner.clean(ocr_result.text)
                    valid_text, non_pub_reasons = evaluate_text(text)
                    snapshot_payload.update(
                        {
                            "text": text,
                            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "char_count": len(text),
                            "word_count": len(text.split()),
                            "captured_at": datetime.now().isoformat(),
                            "source": "ocr_fallback",
                            "extraction_method": "ocr",
                            "tiers_attempted": [
                                *list(snapshot_payload.get("tiers_attempted", [])),
                                "ocr",
                            ],
                            "is_markdown": False,
                            "is_cleaned": True,
                            "is_truncated_for_llm": False,
                            "should_persist": True,
                        }
                    )
                except Exception as ocr_error:
                    logger.warning(
                        f"OCR fallback failed for {paper.paper_id}: {ocr_error}"
                    )

            if not valid_text:
                logger.warning(f"Insufficient text content for paper {paper.paper_id}")
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error="Insufficient text content",
                    text_snapshot=snapshot_payload,
                    extraction_diagnostics=extraction_diagnostics,
                ), False

            if non_pub_reasons:
                reason_text = "; ".join(non_pub_reasons)
                logger.warning(
                    f"Likely non-publication for {paper.paper_id}: {reason_text}"
                )
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error=f"Likely non-publication: {reason_text}",
                    text_snapshot=snapshot_payload,
                    extraction_diagnostics=extraction_diagnostics,
                ), False

            # Classify document type
            stats = self.text_cleaner.get_stats(text)
            section_hits = self.text_cleaner.count_section_markers(text)
            doc_type, type_confidence = classify_document(
                paper=paper,
                text=text,
                word_count=stats.word_count,
                page_count=stats.page_count,
                section_marker_count=section_hits,
            )

            # Skip LLM extraction for non-academic documents
            if (
                doc_type == DocumentType.NON_ACADEMIC
                and type_confidence >= 0.8
                and self.skip_non_publications
            ):
                logger.info(
                    f"Skipping non-academic document {paper.paper_id} "
                    f"(type_confidence={type_confidence:.2f})"
                )
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=False,
                    error=f"Non-academic document (confidence={type_confidence:.2f})",
                    document_type=doc_type.value,
                    type_confidence=type_confidence,
                    text_snapshot=snapshot_payload,
                    extraction_diagnostics=extraction_diagnostics,
                ), False

            if skip_llm:
                actual_profile_key, override_reason = self._resolve_actual_profile(
                    planned_profile_key=planned_profile_key,
                    extraction_method=(
                        cascade_result.method if cascade_result else method
                    ),
                    extraction_diagnostics=extraction_diagnostics,
                )
                return ExtractionResult(
                    paper_id=paper.paper_id,
                    success=True,
                    duration_seconds=time.time() - started_at,
                    document_type=doc_type.value,
                    type_confidence=type_confidence,
                    model_used=None,
                    extraction_method=(
                        cascade_result.method if cascade_result else method
                    ),
                    text_snapshot=snapshot_payload,
                    planned_extraction_intent=planned_intent,
                    planned_profile_key=planned_profile_key,
                    actual_profile_key=actual_profile_key,
                    extraction_routing_overridden=override_reason is not None,
                    extraction_override_reason=override_reason,
                    extraction_diagnostics=extraction_diagnostics,
                ), False

            full_text = text

            # Truncate only the LLM input view if needed. The canonical stored
            # snapshot remains the full cleaned text.
            text = self._truncate_text_for_provider(full_text)

            # Run 6-pass extraction
            result = self._extract_6_pass(
                paper=paper,
                text=text,
                document_type=doc_type.value,
                section_ids=section_ids,
                source_fingerprint=snapshot_hash,
            )

            # Attach classification and extraction method to result
            result.document_type = doc_type.value
            result.type_confidence = type_confidence
            result.extraction_method = (
                cascade_result.method if cascade_result else method
            )
            actual_profile_key, override_reason = self._resolve_actual_profile(
                planned_profile_key=planned_profile_key,
                extraction_method=result.extraction_method,
                extraction_diagnostics=extraction_diagnostics,
            )
            result.planned_extraction_intent = planned_intent
            result.planned_profile_key = planned_profile_key
            result.actual_profile_key = actual_profile_key
            result.extraction_routing_overridden = override_reason is not None
            result.extraction_override_reason = override_reason
            result.extraction_diagnostics = extraction_diagnostics
            result.text_snapshot = snapshot_payload
            if snapshot_payload is not None:
                snapshot_payload["actual_profile_key"] = actual_profile_key
                if override_reason is not None:
                    snapshot_payload["extraction_override_reason"] = override_reason

            # Set indexed timestamp on success
            if result.success:
                paper.indexed_at = datetime.now()

                # Cache the full result
                if self.extraction_cache and full_hash:
                    self.extraction_cache.set_extraction_result(
                        paper.paper_id, full_hash, result
                    )

            return result, False

    @staticmethod
    def _resolve_actual_profile(
        planned_profile_key: str | None,
        extraction_method: str | None,
        extraction_diagnostics: dict[str, str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Infer the actual runtime profile and whether it overrode the plan."""
        if extraction_method == "opendataloader_hybrid":
            if planned_profile_key and planned_profile_key.startswith("hybrid:"):
                actual_profile_key = planned_profile_key
            else:
                actual_profile_key = "runtime:opendataloader_hybrid"
        elif extraction_method:
            actual_profile_key = extraction_method
        else:
            actual_profile_key = planned_profile_key

        if (
            planned_profile_key
            and actual_profile_key
            and actual_profile_key != planned_profile_key
        ):
            hybrid_error = (extraction_diagnostics or {}).get("opendataloader_hybrid")
            if planned_profile_key.startswith("hybrid:") and hybrid_error:
                return (
                    actual_profile_key,
                    f"planned hybrid failed ({hybrid_error}); runtime used "
                    f"{actual_profile_key}",
                )
            if extraction_method == "opendataloader_hybrid" and planned_profile_key == "fast":
                return actual_profile_key, "runtime escalated from fast to hybrid"
            return actual_profile_key, f"runtime used {actual_profile_key} instead of {planned_profile_key}"
        return actual_profile_key, None

    def _extract_6_pass(
        self,
        paper: PaperMetadata,
        text: str,
        document_type: str,
        section_ids: list[str] | None = None,
        source_fingerprint: str | None = None,
    ) -> ExtractionResult:
        """Run pass-based extraction and merge results.

        Each pass extracts a profile-defined section of semantic dimensions.
        Passes run sequentially per paper for consistent reasoning context.

        Args:
            paper: Paper metadata.
            text: Cleaned, truncated paper text.
            document_type: Document type key for prompt framing.
            section_ids: Optional subset of section IDs to run.
            source_fingerprint: Optional canonical snapshot fingerprint for
                pass-cache invalidation.

        Returns:
            Merged ExtractionResult with all dimensions.
        """
        all_answers: dict[str, str | None] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_duration = 0.0
        errors: list[str] = []

        pass_plan = self._resolve_pass_plan(section_ids=section_ids)
        num_passes = len(pass_plan)
        if num_passes == 0:
            return ExtractionResult(
                paper_id=paper.paper_id,
                success=False,
                error="No extraction sections selected",
                model_used=self.model,
            )

        for pass_num, pass_label, pass_questions in pass_plan:
            self._raise_if_pause_requested(f"before pass {pass_num}")
            pass_fields = [q[0] for q in pass_questions]

            # Check per-pass cache
            cached_answers = None
            pass_hash = None
            if self.extraction_cache:
                pass_hash = self.extraction_cache.compute_content_hash(
                    paper.pdf_path,
                    self.model,
                    pass_number=pass_num,
                    profile_fingerprint=self.dimension_profile_fingerprint,
                    source_fingerprint=source_fingerprint,
                )
                cached_data = self.extraction_cache.get(paper.paper_id, pass_hash)
                if cached_data and "answers" in cached_data:
                    cached_answers = cached_data["answers"]
                    total_input_tokens += cached_data.get("input_tokens", 0)
                    total_output_tokens += cached_data.get("output_tokens", 0)
                    total_duration += cached_data.get("duration_seconds", 0.0)
                    logger.debug(
                        f"Using cached pass {pass_num} for {paper.paper_id}"
                    )

            if cached_answers is not None:
                all_answers.update(cached_answers)
                continue

            # Build pass-specific prompt
            embed_text_in_prompt = not (
                self.provider == "anthropic" and self.mode == "cli"
            )
            prompt = build_pass_user_prompt(
                pass_number=pass_num,
                title=paper.title,
                authors=paper.author_string,
                year=paper.publication_year,
                document_type=document_type,
                text=text,
                embed_text=embed_text_in_prompt,
            )

            # Execute LLM call
            result = self.llm_client.extract(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=paper.author_string,
                year=paper.publication_year,
                item_type=paper.item_type,
                text=text,
                prompt_override=prompt,
            )

            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            total_duration += result.duration_seconds

            if not result.success:
                error_msg = f"pass {pass_num} ({pass_label}): {result.error or 'Unknown error'}"
                errors.append(error_msg)
                logger.warning(
                    f"Pass {pass_num} failed for {paper.paper_id}: {result.error}"
                )
                continue

            # Parse pass answers from extraction result
            pass_answers = self._extract_pass_answers(result, pass_fields)

            # Cache per-pass result
            if self.extraction_cache and pass_hash:
                self.extraction_cache.set(paper.paper_id, pass_hash, {
                    "answers": pass_answers,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "duration_seconds": result.duration_seconds,
                })

            all_answers.update(pass_answers)

            logger.debug(
                f"Pass {pass_num} ({pass_label}) complete for {paper.paper_id} "
                f"({result.duration_seconds:.1f}s)"
            )

        # If all passes failed, return error
        if len(errors) == num_passes:
            return ExtractionResult(
                paper_id=paper.paper_id,
                success=False,
                error=f"All {num_passes} passes failed: " + "; ".join(errors),
                duration_seconds=total_duration,
                model_used=self.model,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                pass_errors=list(errors),
            )

        # Build SemanticAnalysis from merged answers
        analysis = SemanticAnalysis(
            paper_id=paper.paper_id,
            profile_id=self.dimension_profile_id,
            profile_version=self.dimension_profile_version,
            profile_fingerprint=self.dimension_profile_fingerprint,
            prompt_version=SEMANTIC_PROMPT_VERSION,
            extraction_model=self.model,
            extracted_at=datetime.now().isoformat(),
            **all_answers,
        )

        # Compute coverage scoring
        coverage_result = score_coverage(analysis)
        analysis.dimension_coverage = coverage_result.coverage
        analysis.coverage_flags = coverage_result.flags

        # Reject extractions with too many failed passes (likely quota hit)
        min_passes = num_passes // 2  # At least half the passes must succeed
        successful_passes = num_passes - len(errors)
        if successful_passes < min_passes:
            logger.warning(
                f"Paper {paper.paper_id}: only {successful_passes}/{num_passes} "
                f"passes succeeded (coverage: {analysis.dimension_coverage:.0%}). "
                f"Marking as failed for retry."
            )
            return ExtractionResult(
                paper_id=paper.paper_id,
                success=False,
                error=(
                    f"Too few passes succeeded ({successful_passes}/{num_passes}): "
                    + "; ".join(errors)
                ),
                duration_seconds=total_duration,
                model_used=self.model,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                pass_errors=list(errors),
            )

        logger.info(
            f"Extracted paper {paper.paper_id} in {total_duration:.1f}s "
            f"({successful_passes}/{num_passes} passes, "
            f"coverage: {analysis.dimension_coverage:.0%})"
        )

        return ExtractionResult(
            paper_id=paper.paper_id,
            success=True,
            extraction=analysis,
            duration_seconds=total_duration,
            model_used=self.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            pass_errors=list(errors),
        )

    def _extract_pass_answers(
        self, result: ExtractionResult, field_names: list[str]
    ) -> dict[str, str | None]:
        """Extract dimension answers from an LLM result for specific fields.

        The LLM returns a JSON object with field names as keys. This method
        extracts only the fields expected for the current pass.

        Args:
            result: LLM extraction result.
            field_names: Expected field names for this pass (e.g., ["q01_research_question", ...]).

        Returns:
            Dict mapping field names to their string answers (or None).
        """
        answers: dict[str, str | None] = {}

        if result.extraction is None:
            # All fields are None if extraction failed
            for field in field_names:
                answers[field] = None
            return answers

        # Extract fields from the extraction object
        for field in field_names:
            value = get_dimension_value(result.extraction, field)
            if value is None:
                value = getattr(result.extraction, field, None)
            answers[field] = value

        return answers

    def _truncate_text_for_provider(self, text: str) -> str:
        """Apply provider-aware truncation for LLM input.

        Anthropic CLI now routes oversized prompts to API fallback on a
        per-paper basis, so preserve the full cleaned text there. Other
        providers continue to use a bounded LLM view while the canonical
        snapshot on disk stays complete.
        """

        if self.provider == "anthropic" and self.mode == "cli":
            return text
        return self.text_cleaner.truncate_for_llm(text)

    def extract_batch(
        self,
        papers: list[PaperMetadata],
        progress_callback: Callable | None = None,
        section_ids: list[str] | None = None,
        text_snapshots: dict[str, dict[str, object]] | None = None,
        extraction_plans: dict[str, ExtractionPlanItem] | None = None,
        skip_llm: bool = False,
    ) -> Generator[ExtractionResult, None, ExtractionStats]:
        """Extract content from multiple papers.

        Uses parallel workers if configured (parallel_workers > 1).
        Automatically reduces workers on rate limit errors.

        Args:
            papers: List of papers to process.
            progress_callback: Optional callback(current, total, paper_title).
            section_ids: Optional section IDs to extract for every paper.
            text_snapshots: Optional mapping of paper_id to canonical text
                snapshots to reuse instead of re-running the extraction cascade.

        Yields:
            ExtractionResult for each paper.

        Returns:
            ExtractionStats with overall statistics.
        """
        if self.parallel_workers > 1:
            return self._extract_batch_parallel(
                papers,
                progress_callback,
                section_ids=section_ids,
                text_snapshots=text_snapshots,
                extraction_plans=extraction_plans,
                skip_llm=skip_llm,
            )
        return self._extract_batch_sequential(
            papers,
            progress_callback,
            section_ids=section_ids,
            text_snapshots=text_snapshots,
            extraction_plans=extraction_plans,
            skip_llm=skip_llm,
        )

    def _extract_batch_sequential(
        self,
        papers: list[PaperMetadata],
        progress_callback: Callable | None = None,
        section_ids: list[str] | None = None,
        text_snapshots: dict[str, dict[str, object]] | None = None,
        extraction_plans: dict[str, ExtractionPlanItem] | None = None,
        skip_llm: bool = False,
    ) -> Generator[ExtractionResult, None, ExtractionStats]:
        """Sequential batch extraction.

        Args:
            papers: List of papers to process.
            progress_callback: Optional callback(current, total, paper_title).
            section_ids: Optional section IDs to extract for every paper.

        Yields:
            ExtractionResult for each paper.

        Returns:
            ExtractionStats with overall statistics.
        """
        import time

        stats = ExtractionStats(total=len(papers))
        consecutive_rate_limits = 0
        provider_failure_counts: dict[str, int] = {}

        i = 0
        while i < len(papers):
            paper = papers[i]
            self._raise_if_pause_requested("before next sequential paper")

            if progress_callback:
                progress_callback(i + 1, len(papers), paper.title)

            # Skip papers without a local file source or supplied snapshot
            if not paper.pdf_path and (text_snapshots or {}).get(paper.paper_id) is None:
                stats.skipped += 1
                i += 1
                continue

            result, from_cache = self.extract_paper(
                paper,
                section_ids=section_ids,
                text_snapshot=(text_snapshots or {}).get(paper.paper_id),
                extraction_plan=(extraction_plans or {}).get(paper.paper_id),
                skip_llm=skip_llm,
            )

            if from_cache:
                stats.cached += 1
                stats.successful += 1
                consecutive_rate_limits = 0
            elif result.success:
                stats.successful += 1
                stats.total_input_tokens += result.input_tokens
                stats.total_output_tokens += result.output_tokens
                stats.total_duration += result.duration_seconds
                consecutive_rate_limits = 0
            else:
                error_str = result.error or ""
                if error_str.startswith(
                    "Likely non-publication"
                ) or error_str.startswith("Insufficient text content"):
                    stats.skipped += 1
                    consecutive_rate_limits = 0
                elif self._is_rate_limit_error(error_str):
                    consecutive_rate_limits += 1
                    if self._handle_quota_exhaustion(consecutive_rate_limits):
                        # Quota sleep completed, retry this paper
                        consecutive_rate_limits = 0
                        continue  # Don't increment i, retry same paper
                    # Brief backoff for transient rate limit
                    logger.warning(
                        f"Rate limit on {paper.paper_id}, waiting 30s"
                    )
                    time.sleep(30)
                    continue  # Retry same paper
                else:
                    stats.failed += 1
                    consecutive_rate_limits = 0
                    failure_kind = self._get_provider_failure_kind(error_str)
                    if failure_kind:
                        provider_failure_counts[failure_kind] = (
                            provider_failure_counts.get(failure_kind, 0) + 1
                        )
                        if self._should_abort_for_provider_failure(
                            failure_kind,
                            provider_failure_counts[failure_kind],
                        ):
                            logger.error(
                                "Aborting extraction after %d repeated %s failures. "
                                "Latest error for %s: %s",
                                provider_failure_counts[failure_kind],
                                failure_kind,
                                paper.paper_id,
                                error_str,
                            )
                            stats.total_duration += result.duration_seconds
                            yield result
                            break
                stats.total_duration += result.duration_seconds

            yield result
            i += 1

        self._log_completion_stats(stats)
        return stats

    def _extract_batch_parallel(
        self,
        papers: list[PaperMetadata],
        progress_callback: Callable | None = None,
        section_ids: list[str] | None = None,
        text_snapshots: dict[str, dict[str, object]] | None = None,
        extraction_plans: dict[str, ExtractionPlanItem] | None = None,
        skip_llm: bool = False,
    ) -> Generator[ExtractionResult, None, ExtractionStats]:
        """Parallel batch extraction with adaptive rate limit backoff.

        Runs multiple papers concurrently using a thread pool. If rate
        limit errors are detected, reduces worker count and retries.

        Args:
            papers: List of papers to process.
            progress_callback: Optional callback(current, total, paper_title).
            section_ids: Optional section IDs to extract for every paper.

        Yields:
            ExtractionResult for each paper.

        Returns:
            ExtractionStats with overall statistics.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        stats = ExtractionStats(total=len(papers))
        current_workers = self.parallel_workers
        rate_limit_count = 0
        completed = 0
        provider_failure_counts: dict[str, int] = {}
        abort_remaining = False

        logger.info(f"Parallel extraction: {len(papers)} papers, {current_workers} workers")

        # Process in chunks to allow worker adjustment between chunks
        remaining = list(papers)

        while remaining:
            self._raise_if_pause_requested("before scheduling next parallel chunk")
            chunk_size = min(current_workers * 3, len(remaining))
            chunk = remaining[:chunk_size]
            remaining = remaining[chunk_size:]
            pause_requested = False
            pause_error: PauseRequested | None = None

            with ThreadPoolExecutor(max_workers=current_workers) as executor:
                future_to_paper = {
                    executor.submit(
                        self._extract_single_paper,
                        paper,
                        section_ids,
                        (text_snapshots or {}).get(paper.paper_id),
                        (extraction_plans or {}).get(paper.paper_id),
                        skip_llm,
                    ): paper
                    for paper in chunk
                }

                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(papers), paper.title)

                    try:
                        result, from_cache = future.result()
                    except PauseRequested as exc:
                        pause_requested = True
                        pause_error = exc
                        continue
                    except Exception as exc:
                        error_str = str(exc)
                        result = ExtractionResult(
                            paper_id=paper.paper_id,
                            success=False,
                            error=error_str,
                        )
                        from_cache = False

                        # Detect rate limit errors
                        if self._is_rate_limit_error(error_str):
                            rate_limit_count += 1
                            if current_workers > 1:
                                current_workers = max(1, current_workers - 1)
                                logger.warning(
                                    f"Rate limit hit ({rate_limit_count}x). "
                                    f"Reducing workers to {current_workers}. "
                                    f"Waiting 30s before retry."
                                )
                                time.sleep(30)
                                # Re-queue this paper
                                remaining.insert(0, paper)
                                completed -= 1
                                continue

                    if from_cache:
                        stats.cached += 1
                        stats.successful += 1
                    elif result.success:
                        stats.successful += 1
                        stats.total_input_tokens += result.input_tokens
                        stats.total_output_tokens += result.output_tokens
                        stats.total_duration += result.duration_seconds
                        # Reset rate limit count on success
                        rate_limit_count = 0
                    else:
                        error_str = result.error or ""
                        if error_str.startswith(
                            "Likely non-publication"
                        ) or error_str.startswith("Insufficient text content"):
                            stats.skipped += 1
                        elif self._is_rate_limit_error(error_str):
                            rate_limit_count += 1
                            if current_workers > 1:
                                current_workers = max(1, current_workers - 1)
                                logger.warning(
                                    f"Rate limit in result ({rate_limit_count}x). "
                                    f"Reducing workers to {current_workers}. "
                                    f"Waiting 30s."
                                )
                                time.sleep(30)
                                remaining.insert(0, paper)
                                completed -= 1
                                continue
                            elif self._handle_quota_exhaustion(rate_limit_count):
                                rate_limit_count = 0
                                current_workers = self.parallel_workers
                                remaining.insert(0, paper)
                                completed -= 1
                                continue
                            else:
                                stats.failed += 1
                        else:
                            stats.failed += 1
                            failure_kind = self._get_provider_failure_kind(error_str)
                            if failure_kind:
                                provider_failure_counts[failure_kind] = (
                                    provider_failure_counts.get(failure_kind, 0) + 1
                                )
                                if self._should_abort_for_provider_failure(
                                    failure_kind,
                                    provider_failure_counts[failure_kind],
                                ):
                                    abort_remaining = True
                                    logger.error(
                                        "Aborting extraction after %d repeated %s failures. "
                                        "Latest error for %s: %s",
                                        provider_failure_counts[failure_kind],
                                        failure_kind,
                                        paper.paper_id,
                                        error_str,
                                    )
                        stats.total_duration += result.duration_seconds

                    yield result

                if pause_requested:
                    raise pause_error or PauseRequested("Pause requested")

                if abort_remaining:
                    remaining = []
                    break

        if rate_limit_count > 0:
            logger.info(
                f"Rate limits encountered: {rate_limit_count}x. "
                f"Final worker count: {current_workers}"
            )

        self._log_completion_stats(stats)
        return stats

    def _extract_single_paper(
        self,
        paper: PaperMetadata,
        section_ids: list[str] | None = None,
        text_snapshot: dict[str, object] | None = None,
        extraction_plan: ExtractionPlanItem | None = None,
        skip_llm: bool = False,
    ) -> tuple[ExtractionResult, bool]:
        """Extract a single paper (for use in thread pool).

        Args:
            paper: Paper to extract.

        Returns:
            Tuple of (ExtractionResult, from_cache).
        """
        if not paper.pdf_path and text_snapshot is None:
            return ExtractionResult(
                paper_id=paper.paper_id,
                success=False,
                error="No local source path",
            ), False
        return self.extract_paper(
            paper,
            section_ids=section_ids,
            text_snapshot=text_snapshot,
            extraction_plan=extraction_plan,
            skip_llm=skip_llm,
        )

    # Maximum consecutive rate limit errors before triggering quota sleep
    _QUOTA_THRESHOLD = 3
    # Hours to sleep when quota is exhausted
    _QUOTA_SLEEP_HOURS = 4
    # Repeated provider-wide failures before aborting new chunk scheduling
    _FATAL_PROVIDER_FAILURE_THRESHOLD = 2

    @staticmethod
    def _is_rate_limit_error(error: str) -> bool:
        """Check if an error string indicates a rate limit."""
        lower = error.lower()
        return any(phrase in lower for phrase in (
            "rate limit",
            "rate limited",
            "429",
            "too many requests",
            "quota exceeded",
            "throttl",
            "usage_limit_reached",
            "usage limit",
            "usage cap",
            "credit balance",
        ))

    @staticmethod
    def _get_provider_failure_kind(error: str) -> str | None:
        """Classify provider-wide failures that should halt new work quickly."""

        lower = error.lower()
        if SectionExtractor._is_rate_limit_error(lower):
            return "rate_limit"
        if any(
            phrase in lower
            for phrase in (
                "authentication failed during extraction",
                "not authenticated",
                "please re-authenticate",
                "session expired",
                "oauth token",
                "login required",
            )
        ):
            return "authentication"
        return None

    def _should_abort_for_provider_failure(
        self,
        failure_kind: str,
        count: int,
    ) -> bool:
        """Return whether repeated provider-wide failures should halt new work."""

        if failure_kind == "rate_limit":
            return False
        return count >= self._FATAL_PROVIDER_FAILURE_THRESHOLD

    def _handle_quota_exhaustion(self, consecutive_errors: int) -> bool:
        """Sleep and retry if quota appears exhausted.

        Called when rate limit errors persist even at 1 worker. Sleeps
        for ``_QUOTA_SLEEP_HOURS`` with periodic status logs, then
        signals the caller to retry.

        Args:
            consecutive_errors: Number of consecutive rate limit errors.

        Returns:
            True if caller should retry (slept and ready), False if
            threshold not met.
        """
        import time

        if consecutive_errors < self._QUOTA_THRESHOLD:
            return False

        sleep_seconds = self._QUOTA_SLEEP_HOURS * 3600
        logger.warning(
            f"Quota appears exhausted ({consecutive_errors} consecutive rate limit "
            f"errors at 1 worker). Sleeping {self._QUOTA_SLEEP_HOURS} hours. "
            f"Extraction will auto-resume. Safe to leave unattended."
        )

        # Log every 30 min so the process looks alive
        interval = 1800  # 30 minutes
        slept = 0
        while slept < sleep_seconds:
            chunk = min(interval, sleep_seconds - slept)
            time.sleep(chunk)
            slept += chunk
            remaining = (sleep_seconds - slept) / 3600
            if remaining > 0:
                logger.info(
                    f"Quota cooldown: {remaining:.1f} hours remaining"
                )

        logger.info("Quota cooldown complete. Resuming extraction.")
        return True

    def _log_completion_stats(self, stats: ExtractionStats) -> None:
        """Log completion statistics."""
        cache_info = ""
        if stats.cached > 0:
            cache_info = f", {stats.cached} from cache ({stats.cache_hit_rate:.1%} hit rate)"

        logger.info(
            f"Extraction complete: {stats.successful}/{stats.total} successful "
            f"({stats.success_rate:.1%}), {stats.skipped} skipped{cache_info}"
        )

    def clear_cache(self) -> int:
        """Clear the extraction cache.

        Returns:
            Number of cache entries removed.
        """
        if self.extraction_cache:
            return self.extraction_cache.clear()
        return 0

    def estimate_batch_cost(self, papers: list[PaperMetadata]) -> dict:
        """Estimate cost for batch extraction.

        Args:
            papers: List of papers to process.

        Returns:
            Dictionary with cost estimates.
        """
        papers_with_pdf = [p for p in papers if p.pdf_path and p.pdf_path.exists()]

        # Check how many would be cached
        cached_count = 0
        if self.extraction_cache:
            for paper in papers_with_pdf:
                content_hash = self.extraction_cache.compute_content_hash(
                    paper.pdf_path, self.model
                )
                if self.extraction_cache.get_extraction_result(paper.paper_id, content_hash):
                    cached_count += 1

        papers_to_extract = len(papers_with_pdf) - cached_count

        # Sample text lengths
        sample_size = min(10, len(papers_with_pdf))
        sample_lengths = []

        for paper in papers_with_pdf[:sample_size]:
            try:
                text = self.pdf_extractor.extract_text(paper.pdf_path)
                text = self.text_cleaner.clean(text)
                text = self._truncate_text_for_provider(text)
                sample_lengths.append(len(text))
            except Exception:
                continue

        if not sample_lengths:
            return {
                "papers_with_pdf": len(papers_with_pdf),
                "papers_cached": cached_count,
                "papers_to_extract": papers_to_extract,
                "estimated_cost": 0,
                "note": "Could not sample papers for estimate",
            }

        avg_length = sum(sample_lengths) / len(sample_lengths)
        num_passes = len(get_pass_definitions())
        per_paper_cost = self.llm_client.estimate_cost(int(avg_length)) * num_passes
        total_cost = per_paper_cost * papers_to_extract

        return {
            "papers_with_pdf": len(papers_with_pdf),
            "papers_cached": cached_count,
            "papers_to_extract": papers_to_extract,
            "average_text_length": int(avg_length),
            "passes_per_paper": num_passes,
            "estimated_cost_per_paper": round(per_paper_cost, 4),
            "estimated_total_cost": round(total_cost, 2),
            "model": self.model,
        }
