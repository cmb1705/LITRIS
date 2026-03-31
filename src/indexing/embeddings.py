"""Embedding generation for semantic search."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.analysis.raptor import RaptorSummaries

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:  # noqa: BLE001 - handle missing or broken optional dependency
    _SentenceTransformer = None

try:
    from ollama import Client as _OllamaClient
except Exception:  # noqa: BLE001 - handle missing or broken optional dependency
    _OllamaClient = None

from src.analysis.schemas import SemanticAnalysis
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

# Dimension chunk types: one per non-None SemanticAnalysis question field
_DIM_CHUNK_TYPES = [f"dim_q{i:02d}" for i in range(1, 41)]

ChunkType = str  # "dim_q01" through "dim_q40", "raptor_overview", "raptor_core", "abstract"

CHUNK_TYPES: list[str] = [
    "abstract",
    *_DIM_CHUNK_TYPES,
    "raptor_overview",
    "raptor_core",
]

# Mapping from dimension chunk type to thematic group
DIMENSION_GROUPS: dict[str, str] = {}
_GROUP_RANGES = [
    ("research_core", range(1, 6)),
    ("methodology", range(6, 11)),
    ("context", range(11, 17)),
    ("meta", range(17, 25)),
    ("scholarly", range(25, 32)),
    ("impact", range(32, 41)),
]
for _group, _rng in _GROUP_RANGES:
    for _i in _rng:
        DIMENSION_GROUPS[f"dim_q{_i:02d}"] = _group

SentenceTransformer = _SentenceTransformer

DEFAULT_SENTENCE_TRANSFORMER_BATCH_SIZE = 32
OLLAMA_AUTO_BATCH_START = 8
OLLAMA_AUTO_BATCH_MAX = 128
DEFAULT_OLLAMA_BATCH_SIZE = OLLAMA_AUTO_BATCH_START


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
            query_prefix: Prefix to prepend to query texts (e.g., instruction for Qwen3).
            document_prefix: Prefix to prepend to document texts during indexing.
        """
        self.model_name = model_name
        self.max_chunk_tokens = max_chunk_tokens
        self.device = device
        self.backend = backend
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
        self.model = transformer_cls(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

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
        self._ollama_client = client_cls(host=base_url)
        self.model = None  # No sentence-transformers model
        # Probe the model to get the embedding dimension
        try:
            probe = self._ollama_client.embed(model=self.model_name, input=["test"])
            self.embedding_dim = len(probe["embeddings"][0])
            logger.info(f"Ollama model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama at {base_url} or load model '{self.model_name}'. "
                f"Ensure Ollama is running and the model is pulled: ollama pull {self.model_name}"
            ) from exc

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
        extraction: SemanticAnalysis,
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

        # Derive quality_rating from q21_quality prose (integer 1-5)
        quality_rating = self._derive_quality_rating(extraction.q21_quality)

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

        # Dimension chunks: one per non-None q field
        for field_name in extraction.DIMENSION_FIELDS:
            value = getattr(extraction, field_name, None)
            if value is None:
                continue
            # e.g., "q01_research_question" -> "dim_q01"
            dim_key = f"dim_{field_name[:3]}"
            group = DIMENSION_GROUPS.get(dim_key, "unknown")
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
        m = re.search(r'(\d)\s*/\s*5|(?:rat(?:ed?|ing)|score)[:\s]+(\d)', q21_prose.lower())
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
    ) -> list[EmbeddingChunk]:
        """Generate embeddings for chunks (documents).

        Applies document_prefix if configured.

        Args:
            chunks: List of chunks without embeddings.
            batch_size: Batch size for embedding generation.
            show_progress: Whether to show progress bar.

        Returns:
            List of chunks with embeddings populated.
        """
        if not chunks:
            return []

        texts = [self._apply_prefix(chunk.text, self.document_prefix) for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")

        if self.backend == "ollama":
            embeddings = self._ollama_embed_batch(texts, batch_size=batch_size)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if hasattr(embedding, "tolist"):
                chunk.embedding = embedding.tolist()
            else:
                chunk.embedding = list(embedding)

        logger.info(f"Generated {len(chunks)} embeddings")
        return chunks

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
            response = self._ollama_client.embed(model=self.model_name, input=[prefixed])
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
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._ollama_embed_with_fallback(batch))
        return all_embeddings

    def _auto_select_ollama_batch_size(self, texts: list[str]) -> int:
        """Probe Ollama upward from a safe batch size on this machine.

        The probe uses the longest available texts up to the configured ceiling so
        the selected size is conservative for the current corpus and hardware.
        """
        if not texts:
            logger.info(
                "Embedding batch size auto has no texts to probe; using safe Ollama batch size %d",
                DEFAULT_OLLAMA_BATCH_SIZE,
            )
            return DEFAULT_OLLAMA_BATCH_SIZE

        max_candidate = max(1, min(len(texts), OLLAMA_AUTO_BATCH_MAX))
        sample_texts = self._select_probe_texts(texts, limit=max_candidate)
        start = max(1, min(OLLAMA_AUTO_BATCH_START, max_candidate))
        logger.info(
            "Auto-probing Ollama embedding batch size from %d up to %d using %d representative chunks",
            start,
            max_candidate,
            len(sample_texts),
        )

        if not self._probe_ollama_batch(sample_texts[:start]):
            logger.warning(
                "Ollama auto-probe failed at safe starting batch size %d; using batch size 1",
                start,
            )
            return 1

        best = start
        lower_failure = max_candidate + 1
        candidate = start

        while candidate < max_candidate:
            next_candidate = min(candidate * 2, max_candidate)
            if next_candidate == candidate:
                break
            if self._probe_ollama_batch(sample_texts[:next_candidate]):
                best = next_candidate
                candidate = next_candidate
                continue
            lower_failure = next_candidate
            break

        if lower_failure == max_candidate + 1:
            self._log_auto_probe_result(best, max_candidate, len(texts))
            return best

        low = best + 1
        high = lower_failure - 1
        while low <= high:
            mid = (low + high) // 2
            if self._probe_ollama_batch(sample_texts[:mid]):
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        self._log_auto_probe_result(best, max_candidate, len(texts))
        return best

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
                response = self._ollama_client.embed(model=self.model_name, input=texts)
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

    def _log_auto_probe_result(
        self,
        selected_batch_size: int,
        max_candidate: int,
        total_text_count: int,
    ) -> None:
        """Log the resolved auto batch size and whether the probe hit its ceiling."""
        if (
            selected_batch_size == max_candidate
            and max_candidate == OLLAMA_AUTO_BATCH_MAX
            and total_text_count > OLLAMA_AUTO_BATCH_MAX
        ):
            logger.info(
                "Auto-selected Ollama embedding batch size %d (hit auto-probe ceiling %d)",
                selected_batch_size,
                OLLAMA_AUTO_BATCH_MAX,
            )
            return
        logger.info("Auto-selected Ollama embedding batch size %d", selected_batch_size)

    def _ollama_embed_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama with retries and recursive batch splitting."""
        if not texts:
            return []

        last_exc: Exception | None = None
        delay = self.ollama_retry_backoff_seconds
        for attempt in range(1, self.ollama_max_retries + 1):
            try:
                response = self._ollama_client.embed(model=self.model_name, input=texts)
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
        return (
            self._ollama_embed_with_fallback(texts[:midpoint])
            + self._ollama_embed_with_fallback(texts[midpoint:])
        )

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
