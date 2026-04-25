"""Configuration loader with validation."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from src.analysis.constants import DEFAULT_MODELS
from src.analysis.dimensions import DEFAULT_DIMENSION_PROFILE, configure_dimension_registry
from src.config_migration import (
    CURRENT_VERSION,
    migrate_config,
    needs_migration,
)
from src.utils.logging_config import get_logger
from src.utils.secrets import get_anthropic_api_key

logger = get_logger(__name__)


def parse_embedding_batch_size_setting(value: Any) -> int | str:
    """Parse an embedding batch size setting.

    Accepts positive integers or the string ``"auto"``.

    Args:
        value: Raw config or CLI value.

    Returns:
        ``"auto"`` or a positive integer batch size.

    Raises:
        ValueError: If the value is not ``"auto"`` or a positive integer.
    """
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError("embedding batch size must be a positive integer or 'auto'") from exc
    if isinstance(value, int):
        if value < 1:
            raise ValueError("embedding batch size must be at least 1")
        return value
    raise ValueError("embedding batch size must be a positive integer or 'auto'")


DEFAULT_OPENDATALOADER_HYBRID_PYTHON = (
    r"C:\Users\cmb17\AppData\Local\Programs\Python\Python310\python.exe"
)


class ManagedHybridServerConfig(BaseModel):
    """A fixed OpenDataLoader hybrid backend endpoint."""

    url: str
    force_ocr: bool = False
    enrich_formula: bool = False
    enrich_picture_description: bool = False

    @field_validator("url")
    @classmethod
    def normalize_url(cls, v: str) -> str:
        """Normalize configured URLs so callers can compare reliably."""
        url = v.strip()
        if not url:
            raise ValueError("managed hybrid server url cannot be empty")
        if "://" not in url:
            url = f"http://{url}"
        return url.rstrip("/")


def default_opendataloader_hybrid_servers() -> dict[str, ManagedHybridServerConfig]:
    """Return the managed localhost hybrid pool defaults."""
    return {
        "base": ManagedHybridServerConfig(url="http://127.0.0.1:5002"),
        "ocr": ManagedHybridServerConfig(
            url="http://127.0.0.1:5003",
            force_ocr=True,
        ),
        "formula": ManagedHybridServerConfig(
            url="http://127.0.0.1:5004",
            enrich_formula=True,
        ),
        "picture": ManagedHybridServerConfig(
            url="http://127.0.0.1:5005",
            enrich_picture_description=True,
        ),
        "ocr_formula": ManagedHybridServerConfig(
            url="http://127.0.0.1:5006",
            force_ocr=True,
            enrich_formula=True,
        ),
        "ocr_picture": ManagedHybridServerConfig(
            url="http://127.0.0.1:5007",
            force_ocr=True,
            enrich_picture_description=True,
        ),
        "formula_picture": ManagedHybridServerConfig(
            url="http://127.0.0.1:5008",
            enrich_formula=True,
            enrich_picture_description=True,
        ),
        "ocr_formula_picture": ManagedHybridServerConfig(
            url="http://127.0.0.1:5009",
            force_ocr=True,
            enrich_formula=True,
            enrich_picture_description=True,
        ),
    }


class ZoteroConfig(BaseModel):
    """Zotero-related configuration."""

    database_path: Path
    storage_path: Path
    write_method: str = "auto"  # auto, api, sqlite
    api_key: str | None = None
    user_id: str | None = None

    @field_validator("database_path", "storage_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    @field_validator("write_method")
    @classmethod
    def validate_write_method(cls, v: str) -> str:
        """Validate write method."""
        valid = {"auto", "api", "sqlite"}
        if v not in valid:
            raise ValueError(f"write_method must be one of {valid}, got '{v}'")
        return v


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


class ProviderSettings(BaseModel):
    """Per-provider extraction settings."""

    model: str | None = None  # None means use provider's default
    mode: str | None = None  # Override default mode for this provider
    effort: str | None = None  # Claude CLI effort level
    reasoning_effort: str | None = None  # OpenAI reasoning effort
    timeout: int | None = None  # Override default timeout


class ExtractionConfig(BaseModel):
    """LLM extraction configuration."""

    provider: str = "anthropic"
    mode: str = "cli"
    model: str | None = None  # None means use provider's default
    max_tokens: int = 100000
    timeout: int = 120
    use_cache: bool = True
    parallel_workers: int = 1
    reasoning_effort: str | None = None  # For OpenAI GPT-5.x: none/low/medium/high/xhigh
    effort: str | None = (
        None  # Claude CLI effort level: low/medium/high (controls extended thinking)
    )
    model_overrides: ModelOverrides | None = None  # Per-item-type model selection
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)

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

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, v: str | None) -> str | None:
        """Validate Claude CLI effort level for extended thinking."""
        if v is None:
            return v
        valid_efforts = {"low", "medium", "high"}
        if v not in valid_efforts:
            raise ValueError(f"effort must be one of {valid_efforts}, got '{v}'")
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

    def get_provider_settings(self, provider: str | None = None) -> ProviderSettings:
        """Get resolved settings for a provider.

        Merges per-provider settings over top-level defaults. If a
        per-provider section exists, its non-None values override the
        top-level extraction config values.

        Args:
            provider: Provider name. If None, uses self.provider.

        Returns:
            ProviderSettings with resolved values.
        """
        provider = provider or self.provider
        per_provider = self.providers.get(provider, ProviderSettings())

        return ProviderSettings(
            model=per_provider.model or self.model,
            mode=per_provider.mode or self.mode,
            effort=per_provider.effort or self.effort,
            reasoning_effort=per_provider.reasoning_effort or self.reasoning_effort,
            timeout=per_provider.timeout or self.timeout,
        )

    def apply_provider(self, provider: str | None = None) -> None:
        """Apply per-provider settings to the top-level config fields.

        Mutates self so that self.model, self.mode, self.effort,
        self.reasoning_effort, and self.timeout reflect the active
        provider's settings. Call this after resolving --provider CLI arg.

        Args:
            provider: Provider name. If None, uses self.provider.
        """
        provider = provider or self.provider
        self.provider = provider
        settings = self.get_provider_settings(provider)
        self.model = settings.model
        self.mode = settings.mode or self.mode
        self.effort = settings.effort
        self.reasoning_effort = settings.reasoning_effort
        self.timeout = settings.timeout or self.timeout

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

        # Return provider defaults from centralized constants
        return DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS["anthropic"])

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
    backend: str = "sentence-transformers"
    device: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_concurrency: int = 1
    query_prefix: str | None = None
    document_prefix: str | None = None
    batch_size: int | str = "auto"

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate embedding backend."""
        valid = {"sentence-transformers", "ollama"}
        if v not in valid:
            raise ValueError(f"backend must be one of {valid}, got '{v}'")
        return v

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: Any) -> int | str:
        """Validate embedding batch size."""
        return parse_embedding_batch_size_setting(v)

    @field_validator("device", mode="before")
    @classmethod
    def normalize_device(cls, v: Any) -> str | None:
        """Normalize blank embedding devices to auto-detect."""
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @field_validator("ollama_concurrency", mode="before")
    @classmethod
    def validate_ollama_concurrency(cls, v: Any) -> int:
        """Validate the bounded number of in-flight Ollama embedding requests."""
        if v is None or v == "":
            return 1
        try:
            value = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("ollama_concurrency must be a positive integer") from exc
        if value < 1:
            raise ValueError("ollama_concurrency must be at least 1")
        return value


class DimensionsConfig(BaseModel):
    """Dimension profile and suggestion configuration."""

    active_profile: str = DEFAULT_DIMENSION_PROFILE
    profile_paths: list[Path] = Field(default_factory=list)
    approval_required: bool = True
    suggestion_sample_size: int = 25
    suggestion_max_proposals: int = 5
    suggestion_neighbor_count: int = 3
    suggestion_use_llm: bool = True

    @field_validator("profile_paths", mode="before")
    @classmethod
    def convert_profile_paths(cls, value: Any) -> list[Path]:
        """Convert profile path inputs into ``Path`` objects."""
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(item) if isinstance(item, str) else item for item in value]

    @field_validator("suggestion_sample_size")
    @classmethod
    def validate_suggestion_sample_size(cls, value: int) -> int:
        """Require a positive suggestion sample size."""
        if value < 1:
            raise ValueError("suggestion_sample_size must be at least 1")
        return value

    @field_validator("suggestion_max_proposals", "suggestion_neighbor_count")
    @classmethod
    def validate_positive_suggestion_limits(cls, value: int) -> int:
        """Require positive limits for suggestion generation."""
        if value < 1:
            raise ValueError("Suggestion limits must be at least 1")
        return value


class StorageConfig(BaseModel):
    """Storage paths configuration."""

    chroma_path: Path = Path("data/chroma")
    cache_path: Path = Path("data/cache")
    collection_name: str = "literature_review"
    index_path: Path | None = Field(
        default=None,
        description=(
            "Optional override for the index output directory (semantic_analyses.json, "
            "dimension_profile.json, similarity pairs, proposals, extraction manifests). "
            "If unset, falls back to <project_root>/data/index."
        ),
    )

    @field_validator("chroma_path", "cache_path", "index_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class ClassificationConfig(BaseModel):
    """Document type classification configuration."""

    enabled: bool = True
    skip_non_academic: bool = True
    min_type_confidence: float = 0.6
    non_academic_item_types: list[str] = Field(
        default_factory=lambda: [
            "presentation",
            "artwork",
            "film",
            "audioRecording",
            "videoRecording",
            "map",
        ]
    )


class ProcessingConfig(BaseModel):
    """Processing options configuration."""

    batch_size: int = 10
    ocr_enabled: bool = False
    ocr_on_fail: bool = True
    min_text_length: int = 100
    skip_non_publications: bool = False
    min_publication_words: int = 500
    min_publication_pages: int = 2
    min_section_hits: int = 0
    ocr_config: dict[str, Any] | None = None
    cascade_enabled: bool = True
    companion_dir: Path | None = None
    arxiv_enabled: bool = True
    opendataloader_enabled: bool = True
    opendataloader_mode: str = "fast"
    opendataloader_hybrid_enabled: bool = False
    opendataloader_hybrid_fallback: bool = False
    opendataloader_hybrid_backend: str = "docling-fast"
    opendataloader_hybrid_client_mode: str = "auto"
    opendataloader_hybrid_url: str | None = None
    opendataloader_hybrid_timeout_ms: int = 60000
    opendataloader_hybrid_autostart: bool = False
    opendataloader_hybrid_host: str = "127.0.0.1"
    opendataloader_hybrid_port: int = 5002
    opendataloader_hybrid_startup_timeout_seconds: float = 30.0
    opendataloader_hybrid_force_ocr: bool = False
    opendataloader_hybrid_ocr_lang: str | None = None
    opendataloader_hybrid_enrich_formula: bool = False
    opendataloader_hybrid_auto_picture_intents: bool = False
    opendataloader_hybrid_enrich_picture_description: bool = False
    opendataloader_hybrid_picture_description_prompt: str | None = None
    opendataloader_hybrid_device: str = "cuda"
    opendataloader_hybrid_python_executable: str = DEFAULT_OPENDATALOADER_HYBRID_PYTHON
    opendataloader_hybrid_servers: dict[str, ManagedHybridServerConfig] = Field(
        default_factory=default_opendataloader_hybrid_servers
    )
    marker_enabled: bool = True
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class FederatedIndexConfig(BaseModel):
    """Configuration for a single federated index source.

    Each federated index points to a separate LITRIS index directory that can be
    searched alongside the primary index. All indexes must use compatible embedding
    models and schema versions.
    """

    path: Path = Field(
        description="Path to index directory (must contain papers.json, semantic_analyses.json, chroma/)"
    )
    label: str = Field(description="Display name for this index in search results")
    enabled: bool = Field(
        default=True, description="Whether to include this index in federated searches"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Relevance weight for results from this index (0.0-2.0)",
    )


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

    enabled: bool = Field(
        default=False, description="Enable federated search across multiple indexes"
    )
    indexes: list[FederatedIndexConfig] = Field(
        default_factory=list, description="List of additional indexes to search"
    )
    merge_strategy: str = Field(
        default="interleave",
        description="How to merge results: 'interleave' (round-robin by score), 'concat' (primary first), 'rerank' (combined scoring)",
    )
    dedup_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for deduplication (0.95 = near-identical papers)",
    )
    max_results_per_index: int = Field(
        default=50, ge=1, description="Maximum results to retrieve from each index before merging"
    )


class Config(BaseModel):
    """Main configuration class for the Literature Review system."""

    version: str = CURRENT_VERSION
    zotero: ZoteroConfig
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    dimensions: DimensionsConfig = Field(default_factory=DimensionsConfig)
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
        # Load environment variables from .env (override system env vars)
        load_dotenv(override=True)

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
        config.configure_dimension_registry()

        # Ensure output directories exist
        config._ensure_directories()

        return config

    @classmethod
    def _find_config_file(cls) -> Path:
        """Find config.yaml.

        Search order:
        1. LITRIS_CONFIG_PATH environment variable (explicit override, useful
           when LITRIS is spawned as an MCP server from another project's cwd).
        2. Current directory and up to 3 parent levels (legacy behavior for
           running tools from inside the LITRIS repo).
        3. The LITRIS install root (the directory two levels up from this
           file). This makes the MCP server self-locating when spawned with
           an arbitrary cwd that the caller cannot control (e.g. Claude Code
           on Windows, which does not reliably honor the .mcp.json cwd field).
        4. Cwd as the last-resort path so the raised FileNotFoundError still
           points somewhere meaningful.
        """
        # 1. Explicit override via env var
        env_path = os.getenv("LITRIS_CONFIG_PATH")
        if env_path:
            return Path(env_path)

        # 2. Current directory and up to 3 parent levels
        current = Path.cwd()
        for _ in range(4):
            config_path = current / "config.yaml"
            if config_path.exists():
                return config_path
            if current.parent == current:
                break
            current = current.parent

        # 3. LITRIS install root (self-healing fallback)
        install_root = Path(__file__).resolve().parent.parent
        install_config = install_root / "config.yaml"
        if install_config.exists():
            return install_config

        # 4. Error path pointing back at cwd
        return Path.cwd() / "config.yaml"

    @classmethod
    def _apply_env_overrides(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Zotero path overrides
        if zotero_db := os.getenv("ZOTERO_DATABASE_PATH"):
            config.setdefault("zotero", {})["database_path"] = zotero_db
        if zotero_storage := os.getenv("ZOTERO_STORAGE_PATH"):
            config.setdefault("zotero", {})["storage_path"] = zotero_storage
        if zotero_api_key := os.getenv("ZOTERO_API_KEY"):
            config.setdefault("zotero", {})["api_key"] = zotero_api_key
        if zotero_user_id := os.getenv("ZOTERO_USER_ID"):
            config.setdefault("zotero", {})["user_id"] = zotero_user_id

        # Extraction overrides
        if extraction_provider := os.getenv("EXTRACTION_PROVIDER"):
            config.setdefault("extraction", {})["provider"] = extraction_provider
        if extraction_mode := os.getenv("EXTRACTION_MODE"):
            config.setdefault("extraction", {})["mode"] = extraction_mode
        if extraction_model := os.getenv("EXTRACTION_MODEL"):
            config.setdefault("extraction", {})["model"] = extraction_model

        if active_profile := os.getenv("LITRIS_DIMENSION_PROFILE"):
            config.setdefault("dimensions", {})["active_profile"] = active_profile

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

    def resolve_dimension_profile_paths(self) -> list[Path]:
        """Return absolute paths for configured external dimension profiles."""

        resolved: list[Path] = []
        for path in self.dimensions.profile_paths:
            candidate = path
            if not candidate.is_absolute() and self._project_root:
                candidate = self._project_root / candidate
            resolved.append(candidate)
        return resolved

    def configure_dimension_registry(self) -> None:
        """Configure the process-wide dimension registry from this config."""

        configure_dimension_registry(
            active_profile_id=self.dimensions.active_profile,
            profile_paths=self.resolve_dimension_profile_paths(),
        )

    def get_cache_path(self) -> Path:
        """Get path to cache directory."""
        path = self.storage.cache_path
        if not path.is_absolute() and self._project_root:
            path = self._project_root / path
        return path

    def get_index_path(self, project_root: Path) -> Path:
        """Resolve the index output directory.

        Priority:
        1. ``storage.index_path`` in config.yaml (relative paths resolve to
           the config file's directory).
        2. ``<project_root>/data/index`` as the hardcoded default.

        Args:
            project_root: LITRIS project root used for the fallback path.
                Typically ``Path(__file__).parent.parent`` of ``build_index.py``.
        """
        if self.storage.index_path is not None:
            path = self.storage.index_path
            if not path.is_absolute() and self._project_root:
                path = self._project_root / path
            return path
        return project_root / "data" / "index"
