"""Embedding generation for semantic search."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import local
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.analysis.raptor import RaptorSummaries

from src.analysis.dimensions import get_default_dimension_registry
from src.analysis.schemas import DimensionedExtraction, SemanticAnalysis
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

# Lazy: avoid eager imports of heavy optional dependencies at module-load
# time. Importing sentence_transformers transitively loads torch +
# transformers (~14s on first call), which previously forced every MCP
# server startup to wait for libraries it may never use when the ollama
# backend is selected. These attributes start as ``None`` and are populated
# on first use by ``_init_sentence_transformers``/``_init_ollama``. Tests
# patch these attributes directly to inject mocks.
_SentenceTransformer = None
_OllamaClient = None

logger = get_logger(__name__)

ChunkType = str


def get_dimension_chunk_types(profile_id: str | None = None) -> list[str]:
    """Return dimension chunk types for the active registry/profile."""

    registry = get_default_dimension_registry()
    return registry.get_chunk_types(
        profile_id=profile_id,
        include_non_dimension=False,
    )


def get_chunk_types(profile_id: str | None = None) -> list[str]:
    """Return all chunk types for the active registry/profile."""

    registry = get_default_dimension_registry()
    return registry.get_chunk_types(profile_id=profile_id)


def get_dimension_group_map(profile_id: str | None = None) -> dict[str, str]:
    """Return chunk-type to group-name mapping for the active profile."""

    registry = get_default_dimension_registry()
    profile = registry.get_profile(profile_id)
    mapping: dict[str, str] = {}
    for section in profile.ordered_sections:
        for dimension in profile.dimensions_for_section(section.id):
            mapping[dimension.chunk_type] = section.id
            if dimension.legacy_chunk_type:
                mapping[dimension.legacy_chunk_type] = section.id
    return mapping


CHUNK_TYPES: list[str] = get_chunk_types()
DIMENSION_GROUPS: dict[str, str] = get_dimension_group_map()

SentenceTransformer = _SentenceTransformer

DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE = 32
OLLAMA_AUTO_BATCH_START = 8
OLLAMA_AUTO_BATCH_MAX = 512
DEFAULT_OLLAMA_BATCH_SIZE = OLLAMA_AUTO_BATCH_START
OLLAMA_AUTO_TARGET_SECONDS = 75.0
OLLAMA_AUTO_BATCH_CANDIDATES = (8, 16, 32, 64, 128, 256, 384, 512)
EMBEDDING_PROGRESS_LOG_INTERVAL_SECONDS = 15.0
EMBEDDING_PROGRESS_WINDOW_SECONDS = 60.0
TRUST_REMOTE_CODE_MODEL_PREFIXES = ("Alibaba-NLP/", "jinaai/")


@dataclass
class EmbeddingChunk:
    """A chunk of text with its embedding and metadata."""

    paper_id: str
    chunk_id: str
    chunk_type: ChunkType
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OllamaProbeResult:
    """Observed result for one Ollama embedding batch probe."""

    batch_size: int
    duration_seconds: float
    success: bool

    @property
    def throughput(self) -> float:
        """Return probe throughput as texts per second."""
        return self.batch_size / max(self.duration_seconds, 1e-9)


@dataclass(frozen=True)
class EmbeddingProgressSnapshot:
    """Progress snapshot for one embedding run."""

    backend: str
    model: str
    batch_size: int
    completed_chunks: int
    total_chunks: int
    elapsed_seconds: float
    rolling_chunks_per_second: float
    overall_chunks_per_second: float
    eta_seconds: float

    @property
    def progress_fraction(self) -> float:
        """Return normalized completion between 0 and 1."""
        if self.total_chunks <= 0:
            return 1.0
        return min(1.0, self.completed_chunks / self.total_chunks)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the progress snapshot."""
        return {
            "backend": self.backend,
            "model": self.model,
            "batch_size": self.batch_size,
            "completed_chunks": self.completed_chunks,
            "total_chunks": self.total_chunks,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "rolling_chunks_per_second": round(self.rolling_chunks_per_second, 6),
            "overall_chunks_per_second": round(self.overall_chunks_per_second, 6),
            "eta_seconds": round(self.eta_seconds, 3),
            "progress_fraction": round(self.progress_fraction, 6),
        }


class EmbeddingProgressTracker:
    """Track rolling embedding throughput and ETA."""

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        batch_size: int,
        total_chunks: int,
    ) -> None:
        self.backend = backend
        self.model = model
        self.batch_size = batch_size
        self.total_chunks = total_chunks
        self.started_at = perf_counter()
        self._history: deque[tuple[float, int]] = deque()
        self._last_log_at = self.started_at

    def snapshot(self, completed_chunks: int) -> EmbeddingProgressSnapshot:
        """Return a progress snapshot after updating the rolling window."""
        now = perf_counter()
        self._history.append((now, completed_chunks))
        while self._history and now - self._history[0][0] > EMBEDDING_PROGRESS_WINDOW_SECONDS:
            self._history.popleft()

        elapsed = max(now - self.started_at, 1e-9)
        overall_rate = completed_chunks / elapsed
        rolling_rate = overall_rate
        if len(self._history) >= 2:
            start_time, start_completed = self._history[0]
            delta_time = now - start_time
            delta_completed = completed_chunks - start_completed
            if delta_time > 0 and delta_completed >= 0:
                rolling_rate = delta_completed / delta_time
        rate_for_eta = rolling_rate if rolling_rate > 0 else overall_rate
        remaining_chunks = max(self.total_chunks - completed_chunks, 0)
        eta_seconds = remaining_chunks / max(rate_for_eta, 1e-9) if remaining_chunks else 0.0
        return EmbeddingProgressSnapshot(
            backend=self.backend,
            model=self.model,
            batch_size=self.batch_size,
            completed_chunks=completed_chunks,
            total_chunks=self.total_chunks,
            elapsed_seconds=elapsed,
            rolling_chunks_per_second=rolling_rate,
            overall_chunks_per_second=overall_rate,
            eta_seconds=eta_seconds,
        )

    def maybe_log(self, snapshot: EmbeddingProgressSnapshot) -> None:
        """Log progress snapshots at a bounded cadence."""
        now = perf_counter()
        if (
            snapshot.completed_chunks < snapshot.total_chunks
            and now - self._last_log_at < EMBEDDING_PROGRESS_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_log_at = now
        percent = snapshot.progress_fraction * 100
        logger.info(
            "Embedding progress: %d/%d chunks (%.1f%%) backend=%s model=%s "
            "batch_size=%d rolling=%.2f chunks/s eta=%s",
            snapshot.completed_chunks,
            snapshot.total_chunks,
            percent,
            snapshot.backend,
            snapshot.model,
            snapshot.batch_size,
            snapshot.rolling_chunks_per_second,
            self._format_duration(snapshot.eta_seconds),
        )

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        """Render a compact ETA string."""
        seconds = max(int(round(total_seconds)), 0)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"


class EmbeddingGenerator:
    """Generate embeddings for paper extractions.

    Supports two backends:
    - sentence-transformers (default): local models via the sentence-transformers library
    - ollama: models served by a local Ollama instance
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_tokens: int = 512,
        device: str | None = None,
        backend: str = "sentence-transformers",
        ollama_base_url: str = "http://localhost:11434",
        ollama_concurrency: int = 1,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ):
        """Initialize embedding generator.

        Args:
            model_name: Name of the embedding model.
            max_chunk_tokens: Maximum tokens per chunk.
            device: Device to use for sentence-transformers ('cpu', 'cuda', or None for auto).
            backend: Embedding backend ('sentence-transformers' or 'ollama').
            ollama_base_url: Base URL for the Ollama server (only used with ollama backend).
            ollama_concurrency: Maximum number of in-flight Ollama embedding requests.
            query_prefix: Prefix to prepend to query texts (e.g., instruction for Qwen3).
            document_prefix: Prefix to prepend to document texts during indexing.
        """
        self.model_name = model_name
        self.max_chunk_tokens = max_chunk_tokens
        self.device = device
        self.backend = backend
        self.ollama_concurrency = max(1, int(ollama_concurrency))
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.ollama_max_retries = 3
        self.ollama_retry_backoff_seconds = 2.0

        logger.info(f"Loading embedding model: {model_name} (backend={backend})")

        if backend == "ollama":
            self._init_ollama(ollama_base_url)
        else:
            self._init_sentence_transformers()

    def _init_sentence_transformers(self) -> None:
        """Initialize sentence-transformers backend."""
        transformer_cls = _SentenceTransformer
        if transformer_cls is None:
            try:
                from sentence_transformers import SentenceTransformer as transformer_cls
            except Exception as exc:  # noqa: BLE001 - surface actionable error
                raise RuntimeError(
                    "sentence-transformers failed to import. "
                    "Install dependencies from requirements.txt and ensure the .venv is active."
                ) from exc
        init_kwargs = self._sentence_transformer_init_kwargs()
        self.model = transformer_cls(self.model_name, device=self.device, **init_kwargs)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        resolved_device = getattr(self.model, "device", self.device)
        logger.info(
            "Model loaded. Embedding dimension: %s (device=%s)",
            self.embedding_dim,
            resolved_device if resolved_device is not None else "auto",
        )

    def _sentence_transformer_init_kwargs(self) -> dict[str, Any]:
        """Return model-init kwargs for sentence-transformers backends."""
        if self.model_name.startswith(TRUST_REMOTE_CODE_MODEL_PREFIXES):
            logger.info(
                "Enabling trust_remote_code for sentence-transformers model %s",
                self.model_name,
            )
            return {"trust_remote_code": True}
        return {}

    def _init_ollama(self, base_url: str) -> None:
        """Initialize Ollama backend.

        Args:
            base_url: Base URL for the Ollama server.
        """
        client_cls = _OllamaClient
        if client_cls is None:
            try:
                from ollama import Client as client_cls
            except Exception as exc:  # noqa: BLE001 - surface actionable error
                raise RuntimeError(
                    "ollama package failed to import. Install it with: pip install ollama"
                ) from exc
        self._ollama_client_cls = client_cls
        self._ollama_base_url = base_url
        self._ollama_client_local = local()
        self._ollama_client = client_cls(host=base_url)
        self.model = None  # No sentence-transformers model
        # Probe the model to get the embedding dimension
        try:
            probe = self._get_ollama_client().embed(model=self.model_name, input=["test"])
            self.embedding_dim = len(probe["embeddings"][0])
            logger.info(
                "Ollama model loaded. Embedding dimension: %d (concurrency=%d)",
                self.embedding_dim,
                self.ollama_concurrency,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama at {base_url} or load model '{self.model_name}'. "
                f"Ensure Ollama is running and the model is pulled: ollama pull {self.model_name}"
            ) from exc

    def _get_ollama_client(self):
        """Return a thread-local Ollama client when concurrency is enabled."""
        if max(1, int(getattr(self, "ollama_concurrency", 1))) <= 1:
            return self._ollama_client
        client = getattr(self._ollama_client_local, "client", None)
        if client is None:
            client = self._ollama_client_cls(host=self._ollama_base_url)
            self._ollama_client_local.client = client
        return client

    def _apply_prefix(self, text: str, prefix: str | None) -> str:
        """Apply a prefix to text if set.

        Args:
            text: The text to prefix.
            prefix: The prefix string, or None to skip.

        Returns:
            Prefixed text.
        """
        if prefix:
            return prefix + text
        return text

    def create_chunks(
        self,
        paper: PaperMetadata,
        extraction: SemanticAnalysis | DimensionedExtraction,
        raptor_summaries: RaptorSummaries | None = None,
    ) -> list[EmbeddingChunk]:
        """Create text chunks from paper metadata and semantic analysis.

        Generates up to 43 chunks per paper:
        - 1 abstract (from Zotero metadata)
        - Up to 40 dimension chunks (one per non-None q field)
        - 2 RAPTOR chunks (paper_overview, core_contribution)

        Args:
            paper: Paper metadata.
            extraction: SemanticAnalysis with q01-q40 dimension answers.
            raptor_summaries: Optional RAPTOR hierarchical summaries for this paper.

        Returns:
            List of EmbeddingChunk objects without embeddings.
        """
        chunks = []

        quality_source = None
        if hasattr(extraction, "get_role"):
            quality_source = extraction.get_role("quality")
        if quality_source is None:
            quality_source = getattr(extraction, "q21_quality", None)
        quality_rating = self._derive_quality_rating(quality_source)

        base_metadata = {
            "title": paper.title,
            "authors": paper.author_string,
            "year": paper.publication_year,
            "collections": ",".join(paper.collections) if paper.collections else "",
            "item_type": paper.item_type,
            "quality_rating": quality_rating,
            "dimension_coverage": extraction.dimension_coverage,
        }

        # Abstract chunk (from Zotero metadata, not extraction)
        if paper.abstract:
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_abstract",
                    chunk_type="abstract",
                    text=self._truncate_text(paper.abstract),
                    metadata={**base_metadata, "dimension_group": "metadata"},
                )
            )

        # Dimension chunks: one per non-None active dimension
        registry = get_default_dimension_registry()
        profile = registry.profiles.get(
            getattr(extraction, "profile_id", None),
            registry.active_profile,
        )
        dimension_values = (
            extraction.dimension_map
            if hasattr(extraction, "dimension_map")
            else extraction.to_dimensioned_extraction().dimension_map
        )
        group_map = get_dimension_group_map(profile.profile_id)
        for dimension in profile.enabled_dimensions:
            value = dimension_values.get(dimension.id)
            if value is None:
                continue
            dim_key = dimension.chunk_type
            group = DIMENSION_GROUPS.get(dim_key, "unknown")
            group = group_map.get(dim_key, group)
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_{dim_key}",
                    chunk_type=dim_key,
                    text=self._truncate_text(value),
                    metadata={**base_metadata, "dimension_group": group},
                )
            )

        # RAPTOR hierarchical summary chunks
        if raptor_summaries is not None:
            if raptor_summaries.paper_overview:
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_raptor_overview",
                        chunk_type="raptor_overview",
                        text=self._truncate_text(raptor_summaries.paper_overview),
                        metadata={**base_metadata, "dimension_group": "raptor"},
                    )
                )
            if raptor_summaries.core_contribution:
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_raptor_core",
                        chunk_type="raptor_core",
                        text=self._truncate_text(raptor_summaries.core_contribution),
                        metadata={**base_metadata, "dimension_group": "raptor"},
                    )
                )

        return chunks

    @staticmethod
    def _derive_quality_rating(q21_prose: str | None) -> int:
        """Derive integer quality rating (1-5) from q21_quality prose.

        Looks for explicit rating patterns in the prose. Defaults to 3.
        """
        if not q21_prose:
            return 0
        import re

        # Match patterns like "4/5", "rated 4", "score: 4", "rating of 4"
        m = re.search(r"(\d)\s*/\s*5|(?:rat(?:ed?|ing)|score)[:\s]+(\d)", q21_prose.lower())
        if m:
            return int(m.group(1) or m.group(2))
        # Keyword heuristic
        low = q21_prose.lower()
        if any(w in low for w in ("excellent", "outstanding", "exceptional")):
            return 5
        if any(w in low for w in ("strong", "high quality", "rigorous", "well-designed")):
            return 4
        if any(w in low for w in ("weak", "poor", "limited", "significant flaws")):
            return 2
        if any(w in low for w in ("very weak", "fundamentally flawed")):
            return 1
        return 3

    def _truncate_text(self, text: str) -> str:
        """Truncate text to approximate max token limit.

        Args:
            text: Text to truncate.

        Returns:
            Truncated text.
        """
        # Rough approximation: 1 token ~= 4 characters
        max_chars = self.max_chunk_tokens * 4
        if len(text) <= max_chars:
            return text

        # Truncate at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]

        return truncated + "..."

    def generate_embeddings(
        self,
        chunks: list[EmbeddingChunk],
        batch_size: int = 32,
        show_progress: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[EmbeddingChunk]:
        """Generate embeddings for chunks (documents).

        Applies document_prefix if configured.

        Args:
            chunks: List of chunks without embeddings.
            batch_size: Batch size for embedding generation.
            show_progress: Whether to show progress bar.
            progress_callback: Optional callback receiving progress snapshots.

        Returns:
            List of chunks with embeddings populated.
        """
        if not chunks:
            return []
        embedded_chunks: list[EmbeddingChunk] = []
        for batch in self.iter_embedded_batches(
            chunks,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_callback=progress_callback,
        ):
            embedded_chunks.extend(batch)
        logger.info(f"Generated {len(chunks)} embeddings")
        return embedded_chunks

    def iter_embedded_batches(
        self,
        chunks: list[EmbeddingChunk],
        batch_size: int = 32,
        show_progress: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[list[EmbeddingChunk]]:
        """Yield embedded chunk batches in generation order."""
        if not chunks:
            return

        texts = [self._apply_prefix(chunk.text, self.document_prefix) for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        tracker = EmbeddingProgressTracker(
            backend=self.backend,
            model=self.model_name,
            batch_size=batch_size,
            total_chunks=len(chunks),
        )
        progress_bar = None
        if show_progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(total=len(chunks), desc="Embedding", unit="chunk")
            except Exception:  # noqa: BLE001 - progress is optional
                progress_bar = None

        completed = 0
        try:
            if self.backend == "ollama":
                for index, embeddings in enumerate(
                    self._iter_ollama_batch_embeddings(texts, batch_size=batch_size)
                ):
                    chunk_batch = chunks[index * batch_size : (index + 1) * batch_size]
                    completed = self._finalize_embedded_batch(
                        chunk_batch=chunk_batch,
                        embeddings=embeddings,
                        completed=completed,
                        tracker=tracker,
                        progress_bar=progress_bar,
                        progress_callback=progress_callback,
                    )
                    yield chunk_batch
            else:
                for index in range(0, len(chunks), batch_size):
                    chunk_batch = chunks[index : index + batch_size]
                    text_batch = texts[index : index + batch_size]
                    embeddings = self._embed_prefixed_batch(text_batch, batch_size=batch_size)
                    completed = self._finalize_embedded_batch(
                        chunk_batch=chunk_batch,
                        embeddings=embeddings,
                        completed=completed,
                        tracker=tracker,
                        progress_bar=progress_bar,
                        progress_callback=progress_callback,
                    )
                    yield chunk_batch
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _finalize_embedded_batch(
        self,
        *,
        chunk_batch: list[EmbeddingChunk],
        embeddings: list[list[float]] | Any,
        completed: int,
        tracker: EmbeddingProgressTracker,
        progress_bar,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> int:
        """Attach embeddings to one chunk batch and emit progress."""
        for chunk, embedding in zip(chunk_batch, embeddings, strict=True):
            if hasattr(embedding, "tolist"):
                chunk.embedding = embedding.tolist()
            else:
                chunk.embedding = list(embedding)
        completed += len(chunk_batch)
        if progress_bar is not None:
            progress_bar.update(len(chunk_batch))
        snapshot = tracker.snapshot(completed)
        if self.backend == "ollama":
            tracker.maybe_log(snapshot)
        if progress_callback is not None:
            progress_callback(snapshot.to_dict())
        return completed

    def _embed_prefixed_batch(
        self,
        prefixed_texts: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]] | Any:
        """Embed one already-prefixed batch for the active backend."""
        if self.backend == "ollama":
            return self._ollama_embed_with_fallback(prefixed_texts)
        return self.model.encode(
            prefixed_texts,
            batch_size=min(batch_size, len(prefixed_texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _iter_ollama_batch_embeddings(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> Iterator[list[list[float]]]:
        """Yield Ollama embedding batches in order with bounded concurrency."""
        if not texts:
            return
        batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]
        ollama_concurrency = max(1, int(getattr(self, "ollama_concurrency", 1)))
        if ollama_concurrency <= 1 or len(batches) <= 1:
            for batch in batches:
                yield self._ollama_embed_with_fallback(batch)
            return

        max_workers = min(ollama_concurrency, len(batches))
        logger.info(
            "Embedding via Ollama with concurrency=%d over %d request batches",
            max_workers,
            len(batches),
        )
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ollama-embed") as executor:
            in_flight: dict[int, Future[list[list[float]]]] = {}
            next_submit = 0
            next_yield = 0
            while next_yield < len(batches):
                while next_submit < len(batches) and len(in_flight) < max_workers:
                    in_flight[next_submit] = executor.submit(
                        self._ollama_embed_with_fallback,
                        batches[next_submit],
                    )
                    next_submit += 1
                future = in_flight.pop(next_yield)
                yield future.result()
                next_yield += 1

    def resolve_batch_size(
        self,
        batch_size: int | str | None,
        texts: list[str] | None = None,
    ) -> int:
        """Resolve the embedding batch size for this backend.

        Args:
            batch_size: Requested batch size setting. Accepts a positive integer,
                ``"auto"``, or ``None``.
            texts: Raw document texts for auto-probing. Only used for Ollama.

        Returns:
            Concrete positive batch size to use for embedding generation.
        """
        requested = batch_size
        if requested is None:
            requested = (
                "auto" if self.backend == "ollama" else DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE
            )

        if isinstance(requested, int):
            return requested
        if isinstance(requested, str) and requested.isdigit():
            return int(requested)

        if requested != "auto":
            raise ValueError(f"Unsupported embedding batch size setting: {requested!r}")

        if self.backend != "ollama":
            logger.info(
                "Embedding batch size auto uses default %d for backend=%s",
                DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE,
                self.backend,
            )
            return DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE

        return self._auto_select_ollama_batch_size(texts or [])

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single query text.

        Applies query_prefix if configured.

        Args:
            text: Text to embed (treated as a query).

        Returns:
            Embedding vector as list of floats.
        """
        prefixed = self._apply_prefix(text, self.query_prefix)

        if self.backend == "ollama":
            response = self._get_ollama_client().embed(model=self.model_name, input=[prefixed])
            return list(response["embeddings"][0])

        embedding = self.model.encode(prefixed, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts (documents).

        Applies document_prefix if configured.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.

        Returns:
            List of embedding vectors.
        """
        prefixed = [self._apply_prefix(t, self.document_prefix) for t in texts]

        if self.backend == "ollama":
            return self._ollama_embed_batch(prefixed, batch_size=batch_size)

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def _ollama_embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of texts via Ollama.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call.

        Returns:
            List of embedding vectors.
        """
        all_embeddings: list[list[float]] = []
        for batch_embeddings in self._iter_ollama_batch_embeddings(texts, batch_size=batch_size):
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def _auto_select_ollama_batch_size(self, texts: list[str]) -> int:
        """Probe Ollama upward from a safe batch size on this machine.

        The probe uses the longest available texts up to the configured ceiling so
        the selected size reflects both request capability and response time.
        """
        if not texts:
            logger.info(
                "Embedding batch size auto has no texts to probe; using safe Ollama batch size %d",
                DEFAULT_OLLAMA_BATCH_SIZE,
            )
            return DEFAULT_OLLAMA_BATCH_SIZE

        max_candidate = max(1, min(len(texts), OLLAMA_AUTO_BATCH_MAX))
        sample_texts = self._select_probe_texts(texts, limit=max_candidate)
        candidate_sizes = self._candidate_ollama_batch_sizes(max_candidate)
        logger.info(
            "Auto-probing Ollama embedding batch size from %d up to %d using %d representative chunks",
            candidate_sizes[0],
            max_candidate,
            len(sample_texts),
        )

        probe_results: list[OllamaProbeResult] = []
        for batch_size in candidate_sizes:
            result = self._probe_ollama_batch_timed(sample_texts[:batch_size])
            probe_results.append(result)
            if not result.success:
                if batch_size == candidate_sizes[0]:
                    logger.warning(
                        "Ollama auto-probe failed at safe starting batch size %d; using batch size 1",
                        batch_size,
                    )
                    return 1
                break
            if result.duration_seconds > OLLAMA_AUTO_TARGET_SECONDS:
                logger.info(
                    "Stopping Ollama auto-probe at batch size %d because %.2fs exceeded the %.2fs latency target",
                    batch_size,
                    result.duration_seconds,
                    OLLAMA_AUTO_TARGET_SECONDS,
                )
                break

        selected = self._select_auto_probe_result(probe_results)
        self._log_auto_probe_result(selected, max_candidate, len(texts), probe_results)
        return selected.batch_size

    def _candidate_ollama_batch_sizes(self, max_candidate: int) -> list[int]:
        """Return the ordered candidate batch sizes to probe."""
        candidates = [size for size in OLLAMA_AUTO_BATCH_CANDIDATES if size <= max_candidate]
        if not candidates:
            return [max_candidate]
        if candidates[-1] != max_candidate:
            candidates.append(max_candidate)
        return candidates

    def _select_auto_probe_result(
        self,
        probe_results: list[OllamaProbeResult],
    ) -> OllamaProbeResult:
        """Choose the best successful probe by measured sustained throughput."""
        successes = [result for result in probe_results if result.success]
        if not successes:
            return OllamaProbeResult(
                batch_size=1,
                duration_seconds=0.0,
                success=True,
            )

        within_budget = [
            result for result in successes if result.duration_seconds <= OLLAMA_AUTO_TARGET_SECONDS
        ]
        if within_budget:
            return max(
                within_budget,
                key=lambda result: (result.throughput, -result.duration_seconds, result.batch_size),
            )
        return max(
            successes,
            key=lambda result: (result.throughput, -result.duration_seconds, result.batch_size),
        )

    def _select_probe_texts(self, texts: list[str], limit: int) -> list[str]:
        """Select the longest prefixed texts for conservative batch probing."""
        heap: list[tuple[int, str]] = []
        for text in texts:
            prefixed = self._apply_prefix(text, self.document_prefix)
            item = (len(prefixed), prefixed)
            if len(heap) < limit:
                heapq.heappush(heap, item)
                continue
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        return [text for _, text in sorted(heap, reverse=True)]

    def _probe_ollama_batch(self, texts: list[str]) -> bool:
        """Test whether Ollama can handle a batch in a single request.

        Retries transient failures, but does not split the batch. That keeps the
        probe focused on finding the largest viable request size for the current
        backend state.
        """
        if not texts:
            return True

        delay = self.ollama_retry_backoff_seconds
        last_exc: Exception | None = None
        for attempt in range(1, self.ollama_max_retries + 1):
            try:
                response = self._get_ollama_client().embed(model=self.model_name, input=texts)
                if len(response["embeddings"]) != len(texts):
                    raise RuntimeError(
                        f"Expected {len(texts)} embeddings, got {len(response['embeddings'])}"
                    )
                return True
            except Exception as exc:  # noqa: BLE001 - client errors are backend-specific
                last_exc = exc
                if attempt < self.ollama_max_retries:
                    sleep(delay)
                    delay = min(delay * 2, 10.0)

        logger.info(
            "Ollama auto-probe rejected batch size %d: %s",
            len(texts),
            last_exc,
        )
        return False

    def _probe_ollama_batch_timed(self, texts: list[str]) -> OllamaProbeResult:
        """Probe one batch size and record both success and elapsed time."""
        start = perf_counter()
        success = self._probe_ollama_batch(texts)
        duration = perf_counter() - start
        logger.info(
            "Ollama auto-probe batch size %d %s in %.2fs",
            len(texts),
            "succeeded" if success else "failed",
            duration,
        )
        return OllamaProbeResult(
            batch_size=len(texts),
            duration_seconds=duration,
            success=success,
        )

    def _log_auto_probe_result(
        self,
        selected_result: OllamaProbeResult,
        max_candidate: int,
        total_text_count: int,
        probe_results: list[OllamaProbeResult],
    ) -> None:
        """Log the resolved auto batch size and whether the probe hit its ceiling."""
        summary = ", ".join(
            (
                f"{result.batch_size}:{'ok' if result.success else 'fail'}@"
                f"{result.duration_seconds:.1f}s"
            )
            for result in probe_results
        )
        if (
            selected_result.batch_size == max_candidate
            and max_candidate == OLLAMA_AUTO_BATCH_MAX
            and total_text_count > OLLAMA_AUTO_BATCH_MAX
        ):
            logger.info(
                "Auto-selected Ollama embedding batch size %d (hit auto-probe ceiling %d; probes: %s)",
                selected_result.batch_size,
                OLLAMA_AUTO_BATCH_MAX,
                summary,
            )
            return
        logger.info(
            "Auto-selected Ollama embedding batch size %d within %.2fs latency budget (observed throughput %.2f texts/s; probes: %s)",
            selected_result.batch_size,
            OLLAMA_AUTO_TARGET_SECONDS,
            selected_result.throughput,
            summary,
        )

    def _ollama_embed_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama with retries and recursive batch splitting."""
        if not texts:
            return []

        last_exc: Exception | None = None
        delay = self.ollama_retry_backoff_seconds
        for attempt in range(1, self.ollama_max_retries + 1):
            try:
                response = self._get_ollama_client().embed(model=self.model_name, input=texts)
                return [list(emb) for emb in response["embeddings"]]
            except Exception as exc:  # noqa: BLE001 - client errors are backend-specific
                last_exc = exc
                if attempt < self.ollama_max_retries:
                    logger.warning(
                        "Ollama embedding batch failed on attempt %d/%d for %d texts: %s",
                        attempt,
                        self.ollama_max_retries,
                        len(texts),
                        exc,
                    )
                    sleep(delay)
                    delay = min(delay * 2, 10.0)

        if len(texts) == 1:
            assert last_exc is not None
            raise last_exc

        midpoint = max(1, len(texts) // 2)
        logger.warning(
            "Ollama embedding batch for %d texts still failed after retries; "
            "retrying as batches of %d and %d",
            len(texts),
            midpoint,
            len(texts) - midpoint,
        )
        return self._ollama_embed_with_fallback(
            texts[:midpoint]
        ) + self._ollama_embed_with_fallback(texts[midpoint:])

    def process_papers(
        self,
        papers: list[PaperMetadata],
        extractions: dict[str, SemanticAnalysis],
        batch_size: int = 32,
    ) -> list[EmbeddingChunk]:
        """Process multiple papers and generate embeddings.

        Args:
            papers: List of paper metadata.
            extractions: Dictionary mapping paper_id to SemanticAnalysis.
            batch_size: Batch size for embedding generation.

        Returns:
            List of all chunks with embeddings.
        """
        all_chunks = []

        for paper in papers:
            if paper.paper_id not in extractions:
                logger.warning(f"No extraction found for paper {paper.paper_id}")
                continue

            extraction = extractions[paper.paper_id]
            chunks = self.create_chunks(paper, extraction)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(papers)} papers")

        if all_chunks:
            all_chunks = self.generate_embeddings(all_chunks, batch_size=batch_size)

        return all_chunks
