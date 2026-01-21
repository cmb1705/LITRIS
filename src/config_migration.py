"""Configuration schema versioning and migration utilities.

This module handles migration of config.yaml files between schema versions.
When the configuration schema changes in backwards-incompatible ways,
migration functions can automatically upgrade old configs.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Current config schema version
CURRENT_VERSION = "1.2.0"

# Minimum supported version for migration
MIN_SUPPORTED_VERSION = "1.0.0"


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse semantic version string to tuple.

    Args:
        version: Version string like "1.2.3".

    Returns:
        Tuple of (major, minor, patch).
    """
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings.

    Args:
        v1: First version string.
        v2: Second version string.

    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2.
    """
    p1 = parse_version(v1)
    p2 = parse_version(v2)
    if p1 < p2:
        return -1
    elif p1 > p2:
        return 1
    return 0


def get_config_version(config: dict[str, Any]) -> str:
    """Extract version from config dict.

    Args:
        config: Configuration dictionary.

    Returns:
        Version string, defaults to "1.0.0" for unversioned configs.
    """
    return config.get("version", "1.0.0")


# Migration functions registry
# Keys are target versions, values are migration functions
_migrations: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_migration(target_version: str):
    """Decorator to register a migration function.

    Migration functions transform config from the previous version
    to the target version.

    Args:
        target_version: Version this migration upgrades to.

    Example:
        @register_migration("1.1.0")
        def migrate_to_1_1_0(config):
            # Add new 'provider' field with default
            config.setdefault("extraction", {}).setdefault("provider", "anthropic")
            return config
    """
    def decorator(func: Callable[[dict[str, Any]], dict[str, Any]]):
        _migrations[target_version] = func
        return func
    return decorator


def get_migration_path(from_version: str, to_version: str) -> list[str]:
    """Get ordered list of versions to migrate through.

    Args:
        from_version: Starting version.
        to_version: Target version.

    Returns:
        List of version strings to migrate through (excluding from_version).
    """
    if compare_versions(from_version, to_version) >= 0:
        return []

    # Get all registered migration versions
    migration_versions = sorted(_migrations.keys(), key=parse_version)

    # Filter to versions between from and to (inclusive of to)
    path = []
    for v in migration_versions:
        if compare_versions(v, from_version) > 0 and compare_versions(v, to_version) <= 0:
            path.append(v)

    return path


def migrate_config(
    config: dict[str, Any],
    target_version: str | None = None,
) -> dict[str, Any]:
    """Migrate configuration to target version.

    Args:
        config: Configuration dictionary to migrate.
        target_version: Target version (defaults to CURRENT_VERSION).

    Returns:
        Migrated configuration dictionary with updated version.

    Raises:
        ValueError: If config version is too old to migrate.
    """
    if target_version is None:
        target_version = CURRENT_VERSION

    current = get_config_version(config)

    # Check if already at target
    if compare_versions(current, target_version) >= 0:
        return config

    # Check if too old
    if compare_versions(current, MIN_SUPPORTED_VERSION) < 0:
        raise ValueError(
            f"Config version {current} is too old to migrate. "
            f"Minimum supported version is {MIN_SUPPORTED_VERSION}. "
            "Please recreate your config.yaml from the example."
        )

    # Get migration path
    path = get_migration_path(current, target_version)

    if not path:
        # No migrations needed, just update version
        config["version"] = target_version
        return config

    # Apply migrations in order
    result = config.copy()
    for version in path:
        migration_func = _migrations.get(version)
        if migration_func:
            logger.info(f"Applying config migration to version {version}")
            result = migration_func(result)
            result["version"] = version

    # Ensure final version is set
    result["version"] = target_version
    return result


def needs_migration(config: dict[str, Any]) -> bool:
    """Check if configuration needs migration.

    Args:
        config: Configuration dictionary.

    Returns:
        True if config version is older than current.
    """
    current = get_config_version(config)
    return compare_versions(current, CURRENT_VERSION) < 0


def backup_config(config_path: Path) -> Path:
    """Create backup of config file before migration.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Path to backup file.
    """
    backup_path = config_path.with_suffix(f".yaml.bak.{get_config_version_from_file(config_path)}")

    # Find unique backup name
    counter = 1
    while backup_path.exists():
        backup_path = config_path.with_suffix(f".yaml.bak.{counter}")
        counter += 1

    import shutil
    shutil.copy2(config_path, backup_path)
    logger.info(f"Config backup created: {backup_path}")
    return backup_path


def get_config_version_from_file(config_path: Path) -> str:
    """Read version from config file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Version string.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return get_config_version(config)


def migrate_config_file(
    config_path: Path,
    target_version: str | None = None,
    backup: bool = True,
) -> dict[str, Any]:
    """Migrate config file in place.

    Args:
        config_path: Path to config.yaml.
        target_version: Target version (defaults to CURRENT_VERSION).
        backup: Whether to create backup before migration.

    Returns:
        Migrated configuration dictionary.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not needs_migration(config):
        return config

    if backup:
        backup_config(config_path)

    migrated = migrate_config(config, target_version)

    # Write migrated config
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(migrated, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Config migrated to version {get_config_version(migrated)}")
    return migrated


# ============================================================================
# Migration Definitions
# ============================================================================
# Add new migrations here as the schema evolves.
# Each migration should transform config from the previous version.


@register_migration("1.1.0")
def migrate_to_1_1_0(config: dict[str, Any]) -> dict[str, Any]:
    """Migration from 1.0.0 to 1.1.0.

    Changes:
    - Add extraction.provider field (default: anthropic)
    - Add extraction.reasoning_effort field (default: None)
    - Add processing.ocr_config field (default: None)
    """
    extraction = config.setdefault("extraction", {})

    # Add provider if not present
    if "provider" not in extraction:
        extraction["provider"] = "anthropic"

    # Add reasoning_effort if not present
    if "reasoning_effort" not in extraction:
        extraction["reasoning_effort"] = None

    # Add ocr_config if processing exists
    processing = config.get("processing", {})
    if processing and "ocr_config" not in processing:
        processing["ocr_config"] = None

    return config


@register_migration("1.2.0")
def migrate_to_1_2_0(config: dict[str, Any]) -> dict[str, Any]:
    """Migration from 1.1.0 to 1.2.0.

    Changes:
    - Add extraction.model_overrides field (default: None)
      This enables per-item-type model selection.
    """
    extraction = config.setdefault("extraction", {})

    # Add model_overrides if not present (optional, defaults to None)
    if "model_overrides" not in extraction:
        extraction["model_overrides"] = None

    return config
