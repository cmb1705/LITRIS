"""Configuration loader with validation."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from src.config_migration import (
    CURRENT_VERSION,
    migrate_config,
    needs_migration,
)
from src.utils.logging_config import get_logger
from src.utils.secrets import get_anthropic_api_key

logger = get_logger(__name__)


class ZoteroConfig(BaseModel):
    """Zotero-related configuration."""

    database_path: Path
    storage_path: Path

    @field_validator("database_path", "storage_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class ModelOverrides(BaseModel):
    """Model overrides for specific item types or extraction sections."""

    # Item type overrides - use specific model for certain document types
    journal_article: str | None = None
    book: str | None = None
    book_section: str | None = None
    thesis: str | None = None
    conference_paper: str | None = None
    report: str | None = None
    preprint: str | None = None

    # Section overrides for multi-pass extraction (future feature)
    # methodology: str | None = None
    # findings: str | None = None
    # summary: str | None = None


class ExtractionConfig(BaseModel):
    """LLM extraction configuration."""

    provider: str = "anthropic"
    mode: str = "cli"
    model: str | None = None  # None means use provider's default
    max_tokens: int = 100000
    timeout: int = 120
    use_cache: bool = True
    parallel_workers: int = 1
    reasoning_effort: str | None = None  # For OpenAI GPT-5.2: none/low/medium/high/xhigh
    model_overrides: ModelOverrides | None = None  # Per-item-type model selection

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = {"anthropic", "openai", "google", "ollama", "llamacpp"}
        if v not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got '{v}'")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate extraction mode."""
        valid_modes = {"api", "cli", "batch_api"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{v}'")
        return v

    @field_validator("reasoning_effort")
    @classmethod
    def validate_reasoning_effort(cls, v: str | None) -> str | None:
        """Validate reasoning effort for OpenAI models."""
        if v is None:
            return v
        valid_efforts = {"none", "low", "medium", "high", "xhigh"}
        if v not in valid_efforts:
            raise ValueError(f"reasoning_effort must be one of {valid_efforts}, got '{v}'")
        return v

    @field_validator("parallel_workers")
    @classmethod
    def validate_parallel_workers(cls, v: int) -> int:
        """Validate parallel workers count."""
        if v < 1:
            raise ValueError("parallel_workers must be at least 1")
        if v > 10:
            raise ValueError("parallel_workers should not exceed 10")
        return v

    def get_model_or_default(self, item_type: str | None = None) -> str:
        """Get model name, using provider default if not specified.

        Args:
            item_type: Optional item type for model override lookup.
                       Supported types: journalArticle, book, bookSection,
                       thesis, conferencePaper, report, preprint.

        Returns:
            Model name to use for extraction.
        """
        # Check for item type override
        if item_type and self.model_overrides:
            override = self._get_item_type_override(item_type)
            if override:
                return override

        # Use explicit model if set
        if self.model:
            return self.model

        # Return provider defaults
        defaults = {
            "anthropic": "claude-opus-4-5-20251101",
            "openai": "gpt-5.2",
            "google": "gemini-3-pro",
            "ollama": "llama3",
            "llamacpp": "llama-3",
        }
        return defaults.get(self.provider, "claude-opus-4-5-20251101")

    def _get_item_type_override(self, item_type: str) -> str | None:
        """Get model override for a specific item type.

        Args:
            item_type: Zotero item type (journalArticle, book, etc.)

        Returns:
            Override model name or None.
        """
        if not self.model_overrides:
            return None

        # Map Zotero item types to override fields
        type_mapping = {
            "journalArticle": self.model_overrides.journal_article,
            "book": self.model_overrides.book,
            "bookSection": self.model_overrides.book_section,
            "thesis": self.model_overrides.thesis,
            "conferencePaper": self.model_overrides.conference_paper,
            "report": self.model_overrides.report,
            "preprint": self.model_overrides.preprint,
        }

        return type_mapping.get(item_type)


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384


class StorageConfig(BaseModel):
    """Storage paths configuration."""

    chroma_path: Path = Path("data/chroma")
    cache_path: Path = Path("data/cache")
    collection_name: str = "literature_review"

    @field_validator("chroma_path", "cache_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class ProcessingConfig(BaseModel):
    """Processing options configuration."""

    batch_size: int = 10
    ocr_enabled: bool = False
    min_text_length: int = 100
    ocr_config: dict[str, Any] | None = None


class FederatedIndexConfig(BaseModel):
    """Configuration for a single federated index source.

    Each federated index points to a separate LITRIS index directory that can be
    searched alongside the primary index. All indexes must use compatible embedding
    models and schema versions.
    """

    path: Path = Field(description="Path to index directory (must contain papers.json, extractions.json, chroma/)")
    label: str = Field(description="Display name for this index in search results")
    enabled: bool = Field(default=True, description="Whether to include this index in federated searches")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="Relevance weight for results from this index (0.0-2.0)")


class FederatedSearchConfig(BaseModel):
    """Configuration for multi-index federated search.

    Federated search enables querying multiple LITRIS indexes simultaneously,
    merging and ranking results across sources. Use cases include:
    - Searching across multiple Zotero libraries
    - Combining results from different research groups
    - Keeping separate indexes for different domains while searching together

    Compatibility Requirements:
    - All indexes must use the same embedding model (e.g., all-MiniLM-L6-v2)
    - Schema versions must be compatible (same major version)
    - Primary index is always searched; federated indexes are additive

    Example YAML:
        federated:
          enabled: true
          merge_strategy: "interleave"
          dedup_threshold: 0.95
          indexes:
            - path: "/data/colleague_index"
              label: "Colleague Library"
              weight: 1.0
            - path: "/data/historical_index"
              label: "Historical Archive"
              weight: 0.8
              enabled: false
    """

    enabled: bool = Field(default=False, description="Enable federated search across multiple indexes")
    indexes: list[FederatedIndexConfig] = Field(default_factory=list, description="List of additional indexes to search")
    merge_strategy: str = Field(
        default="interleave",
        description="How to merge results: 'interleave' (round-robin by score), 'concat' (primary first), 'rerank' (combined scoring)"
    )
    dedup_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for deduplication (0.95 = near-identical papers)"
    )
    max_results_per_index: int = Field(
        default=50,
        ge=1,
        description="Maximum results to retrieve from each index before merging"
    )


class Config(BaseModel):
    """Main configuration class for the Literature Review system."""

    version: str = CURRENT_VERSION
    zotero: ZoteroConfig
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    federated: FederatedSearchConfig = Field(default_factory=FederatedSearchConfig)

    _project_root: Path | None = None

    @classmethod
    def load(
        cls,
        config_path: Path | str | None = None,
        auto_migrate: bool = True,
    ) -> "Config":
        """Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config.yaml. If None, searches for config.yaml
                        in current directory and parent directories.
            auto_migrate: If True, automatically migrate old config versions.

        Returns:
            Loaded and validated Config instance.

        Raises:
            FileNotFoundError: If config.yaml cannot be found.
            ValueError: If required configuration is missing or invalid.
        """
        # Load environment variables from .env
        load_dotenv()

        # Find config file
        if config_path is None:
            config_path = cls._find_config_file()
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                "Please create config.yaml or specify a valid path."
            )

        # Load YAML configuration
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        # Handle config migration if needed
        if auto_migrate and needs_migration(yaml_config):
            old_version = yaml_config.get("version", "1.0.0")
            logger.info(f"Migrating config from version {old_version} to {CURRENT_VERSION}")
            yaml_config = migrate_config(yaml_config)

        # Apply environment variable overrides
        yaml_config = cls._apply_env_overrides(yaml_config)

        # Remove version field if present (Pydantic will use the default)
        # but keep it for the model
        if "version" not in yaml_config:
            yaml_config["version"] = CURRENT_VERSION

        # Create config instance
        config = cls(**yaml_config)
        config._project_root = config_path.parent

        # Ensure output directories exist
        config._ensure_directories()

        return config

    @classmethod
    def _find_config_file(cls) -> Path:
        """Find config.yaml in current or parent directories."""
        current = Path.cwd()

        # Check current directory and up to 3 parent levels
        for _ in range(4):
            config_path = current / "config.yaml"
            if config_path.exists():
                return config_path
            if current.parent == current:
                break
            current = current.parent

        # Default to current directory
        return Path.cwd() / "config.yaml"

    @classmethod
    def _apply_env_overrides(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Zotero path overrides
        if zotero_db := os.getenv("ZOTERO_DATABASE_PATH"):
            config.setdefault("zotero", {})["database_path"] = zotero_db
        if zotero_storage := os.getenv("ZOTERO_STORAGE_PATH"):
            config.setdefault("zotero", {})["storage_path"] = zotero_storage

        # Extraction overrides
        if extraction_provider := os.getenv("EXTRACTION_PROVIDER"):
            config.setdefault("extraction", {})["provider"] = extraction_provider
        if extraction_mode := os.getenv("EXTRACTION_MODE"):
            config.setdefault("extraction", {})["mode"] = extraction_mode
        if extraction_model := os.getenv("EXTRACTION_MODEL"):
            config.setdefault("extraction", {})["model"] = extraction_model

        return config

    def _ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        dirs_to_create = [
            self.get_cache_path(),
            self.get_chroma_path(),
            self.get_cache_path() / "pdf_text",
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    def get_zotero_db_path(self) -> Path:
        """Get validated path to Zotero database.

        Returns:
            Path to zotero.sqlite file.

        Raises:
            FileNotFoundError: If database file doesn't exist.
        """
        path = self.zotero.database_path
        if not path.is_absolute() and self._project_root:
            path = self._project_root / path

        if not path.exists():
            raise FileNotFoundError(
                f"Zotero database not found: {path}\n"
                "Please check zotero.database_path in config.yaml"
            )
        return path

    def get_storage_path(self) -> Path:
        """Get validated path to Zotero storage directory.

        Returns:
            Path to Zotero storage directory containing PDFs.

        Raises:
            FileNotFoundError: If storage directory doesn't exist.
        """
        path = self.zotero.storage_path
        if not path.is_absolute() and self._project_root:
            path = self._project_root / path

        if not path.exists():
            raise FileNotFoundError(
                f"Zotero storage directory not found: {path}\n"
                "Please check zotero.storage_path in config.yaml"
            )
        return path

    def get_anthropic_key(self) -> str | None:
        """Get Anthropic API key from environment or keyring.

        Returns:
            API key string, or None if using CLI mode.

        Raises:
            ValueError: If batch_api mode but no API key set.
        """
        api_key = get_anthropic_api_key()

        if self.extraction.mode == "batch_api" and not api_key:
            raise ValueError(
                "Anthropic API key is required for batch_api mode.\n"
                "Either set ANTHROPIC_API_KEY or store it in the OS keyring "
                "(service: 'litris', key: 'ANTHROPIC_API_KEY'), "
                "or change extraction.mode to 'cli' in config.yaml"
            )

        return api_key

    def get_chroma_path(self) -> Path:
        """Get path to ChromaDB storage directory."""
        path = self.storage.chroma_path
        if not path.is_absolute() and self._project_root:
            path = self._project_root / path
        return path

    def get_cache_path(self) -> Path:
        """Get path to cache directory."""
        path = self.storage.cache_path
        if not path.is_absolute() and self._project_root:
            path = self._project_root / path
        return path
