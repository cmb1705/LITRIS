"""Configuration loader with validation."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class ZoteroConfig(BaseModel):
    """Zotero-related configuration."""

    database_path: Path
    storage_path: Path

    @field_validator("database_path", "storage_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class ExtractionConfig(BaseModel):
    """LLM extraction configuration."""

    mode: str = "cli"
    model: str = "claude-opus-4-5-20251101"
    max_tokens: int = 100000
    timeout: int = 120

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate extraction mode."""
        valid_modes = {"cli", "batch_api"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{v}'")
        return v


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


class Config(BaseModel):
    """Main configuration class for the Literature Review system."""

    zotero: ZoteroConfig
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    _project_root: Path | None = None

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> "Config":
        """Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config.yaml. If None, searches for config.yaml
                        in current directory and parent directories.

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

        # Apply environment variable overrides
        yaml_config = cls._apply_env_overrides(yaml_config)

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

        # Extraction mode override
        if extraction_mode := os.getenv("EXTRACTION_MODE"):
            config.setdefault("extraction", {})["mode"] = extraction_mode

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
        """Get Anthropic API key from environment.

        Returns:
            API key string, or None if using CLI mode.

        Raises:
            ValueError: If batch_api mode but no API key set.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if self.extraction.mode == "batch_api" and not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for batch_api mode.\n"
                "Either set the API key or change extraction.mode to 'cli' in config.yaml"
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
