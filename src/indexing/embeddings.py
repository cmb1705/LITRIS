"""Embedding generation for semantic search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

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

from src.analysis.schemas import PaperExtraction
from src.utils.logging_config import get_logger
from src.zotero.models import PaperMetadata

logger = get_logger(__name__)

ChunkType = Literal[
    "abstract",
    "thesis",
    "contribution",
    "methodology",
    "findings",
    "claims",
    "limitations",
    "future_work",
    "full_summary",
    "section_summary",
    "paper_overview",
    "core_contribution",
]

CHUNK_TYPES: list[ChunkType] = [
    "abstract",
    "thesis",
    "contribution",
    "methodology",
    "findings",
    "claims",
    "limitations",
    "future_work",
    "full_summary",
    "section_summary",
    "paper_overview",
    "core_contribution",
]

SentenceTransformer = _SentenceTransformer


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
        extraction: PaperExtraction,
        raptor_summaries: RaptorSummaries | None = None,
    ) -> list[EmbeddingChunk]:
        """Create text chunks from paper metadata and extraction.

        Args:
            paper: Paper metadata.
            extraction: LLM extraction result.
            raptor_summaries: Optional RAPTOR hierarchical summaries for this paper.

        Returns:
            List of EmbeddingChunk objects without embeddings.
        """
        chunks = []
        base_metadata = {
            "title": paper.title,
            "authors": paper.author_string,
            "year": paper.publication_year,
            "collections": ",".join(paper.collections) if paper.collections else "",
            "item_type": paper.item_type,
            "quality_rating": extraction.quality_rating
            if extraction.quality_rating is not None
            else 0,
        }

        # Abstract chunk
        if paper.abstract:
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_abstract",
                    chunk_type="abstract",
                    text=self._truncate_text(paper.abstract),
                    metadata=base_metadata,
                )
            )

        # Thesis statement chunk
        if extraction.thesis_statement:
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_thesis",
                    chunk_type="thesis",
                    text=self._truncate_text(extraction.thesis_statement),
                    metadata=base_metadata,
                )
            )

        # Contribution summary chunk
        if extraction.contribution_summary:
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_contribution",
                    chunk_type="contribution",
                    text=self._truncate_text(extraction.contribution_summary),
                    metadata=base_metadata,
                )
            )

        # Methodology chunk
        if extraction.methodology:
            method = extraction.methodology
            method_text_parts = []
            if method.approach:
                method_text_parts.append(f"Approach: {method.approach}")
            if method.design:
                method_text_parts.append(f"Design: {method.design}")
            if method.data_sources:
                method_text_parts.append(f"Data sources: {', '.join(method.data_sources)}")
            if method.analysis_methods:
                method_text_parts.append(f"Analysis: {', '.join(method.analysis_methods)}")
            if method.sample_size:
                method_text_parts.append(f"Sample: {method.sample_size}")

            if method_text_parts:
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_methodology",
                        chunk_type="methodology",
                        text=self._truncate_text(". ".join(method_text_parts)),
                        metadata=base_metadata,
                    )
                )

        # Key findings chunks
        if extraction.key_findings:
            findings_text = ". ".join(f.finding for f in extraction.key_findings)
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_findings",
                    chunk_type="findings",
                    text=self._truncate_text(findings_text),
                    metadata=base_metadata,
                )
            )

        # Key claims chunks (individual)
        if extraction.key_claims:
            for i, claim in enumerate(extraction.key_claims):
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_claim_{i}",
                        chunk_type="claims",
                        text=self._truncate_text(claim.claim),
                        metadata={**base_metadata, "claim_index": i},
                    )
                )

        # Limitations chunk
        if extraction.limitations:
            limitations_text = ". ".join(extraction.limitations)
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_limitations",
                    chunk_type="limitations",
                    text=self._truncate_text(limitations_text),
                    metadata=base_metadata,
                )
            )

        # Future directions chunk
        if extraction.future_directions:
            future_text = ". ".join(extraction.future_directions)
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_future_work",
                    chunk_type="future_work",
                    text=self._truncate_text(future_text),
                    metadata=base_metadata,
                )
            )

        # Full summary chunk (combines multiple fields)
        summary_parts = []
        if extraction.thesis_statement:
            summary_parts.append(extraction.thesis_statement)
        if extraction.contribution_summary:
            summary_parts.append(extraction.contribution_summary)
        if extraction.conclusions:
            summary_parts.append(extraction.conclusions)

        if summary_parts:
            chunks.append(
                EmbeddingChunk(
                    paper_id=paper.paper_id,
                    chunk_id=f"{paper.paper_id}_full_summary",
                    chunk_type="full_summary",
                    text=self._truncate_text(" ".join(summary_parts)),
                    metadata=base_metadata,
                )
            )

        # RAPTOR hierarchical summary chunks (if generated)
        if raptor_summaries is not None:
            if raptor_summaries.section_summary:
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_section_summary",
                        chunk_type="section_summary",
                        text=self._truncate_text(raptor_summaries.section_summary),
                        metadata=base_metadata,
                    )
                )
            if raptor_summaries.paper_overview:
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_paper_overview",
                        chunk_type="paper_overview",
                        text=self._truncate_text(raptor_summaries.paper_overview),
                        metadata=base_metadata,
                    )
                )
            if raptor_summaries.core_contribution:
                chunks.append(
                    EmbeddingChunk(
                        paper_id=paper.paper_id,
                        chunk_id=f"{paper.paper_id}_core_contribution",
                        chunk_type="core_contribution",
                        text=self._truncate_text(raptor_summaries.core_contribution),
                        metadata=base_metadata,
                    )
                )

        return chunks

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
            response = self._ollama_client.embed(model=self.model_name, input=batch)
            all_embeddings.extend(list(emb) for emb in response["embeddings"])
        return all_embeddings

    def process_papers(
        self,
        papers: list[PaperMetadata],
        extractions: dict[str, PaperExtraction],
        batch_size: int = 32,
    ) -> list[EmbeddingChunk]:
        """Process multiple papers and generate embeddings.

        Args:
            papers: List of paper metadata.
            extractions: Dictionary mapping paper_id to extraction.
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
