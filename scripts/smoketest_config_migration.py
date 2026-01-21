#!/usr/bin/env python
"""Smoketest for configuration schema versioning and migration."""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_version_parsing():
    """Test semantic version parsing."""
    print("=" * 60)
    print("Testing Version Parsing")
    print("=" * 60)

    from src.config_migration import compare_versions, parse_version

    all_passed = True

    # Test parse_version
    test_cases = [
        ("1.0.0", (1, 0, 0)),
        ("1.2.3", (1, 2, 3)),
        ("2.10.5", (2, 10, 5)),
    ]

    for version_str, expected in test_cases:
        result = parse_version(version_str)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        print(f"  parse_version('{version_str}'): {status}")
        if not passed:
            print(f"    Expected: {expected}, Got: {result}")
            all_passed = False

    # Test compare_versions
    compare_cases = [
        ("1.0.0", "1.0.0", 0),
        ("1.0.0", "1.0.1", -1),
        ("1.1.0", "1.0.0", 1),
        ("2.0.0", "1.9.9", 1),
    ]

    for v1, v2, expected in compare_cases:
        result = compare_versions(v1, v2)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        print(f"  compare_versions('{v1}', '{v2}'): {status}")
        if not passed:
            print(f"    Expected: {expected}, Got: {result}")
            all_passed = False

    return all_passed


def test_migration_detection():
    """Test migration detection logic."""
    print("\n" + "=" * 60)
    print("Testing Migration Detection")
    print("=" * 60)

    from src.config_migration import CURRENT_VERSION, get_config_version, needs_migration

    all_passed = True

    # Config without version should default to 1.0.0
    config_no_version = {"zotero": {"database_path": "/tmp/test.sqlite", "storage_path": "/tmp/storage"}}
    version = get_config_version(config_no_version)
    passed = version == "1.0.0"
    status = "PASS" if passed else "FAIL"
    print(f"  Config without version defaults to 1.0.0: {status}")
    if not passed:
        print(f"    Got: {version}")
        all_passed = False

    # Config with old version needs migration
    config_old = {"version": "1.0.0"}
    needs = needs_migration(config_old)
    passed = needs is True
    status = "PASS" if passed else "FAIL"
    print(f"  Config v1.0.0 needs migration: {status}")
    if not passed:
        all_passed = False

    # Config with current version doesn't need migration
    config_current = {"version": CURRENT_VERSION}
    needs = needs_migration(config_current)
    passed = needs is False
    status = "PASS" if passed else "FAIL"
    print(f"  Config v{CURRENT_VERSION} doesn't need migration: {status}")
    if not passed:
        all_passed = False

    return all_passed


def test_migration_1_0_to_1_1():
    """Test migration from 1.0.0 to 1.1.0."""
    print("\n" + "=" * 60)
    print("Testing Migration 1.0.0 -> 1.1.0")
    print("=" * 60)

    from src.config_migration import migrate_config

    all_passed = True

    # Create a v1.0.0 config (no provider, no reasoning_effort)
    config_v1 = {
        "version": "1.0.0",
        "zotero": {
            "database_path": "/tmp/test.sqlite",
            "storage_path": "/tmp/storage",
        },
        "extraction": {
            "mode": "cli",
            "model": "claude-opus-4-5-20251101",
        },
        "processing": {
            "batch_size": 10,
        },
    }

    migrated = migrate_config(config_v1, target_version="1.1.0")

    # Check provider was added
    provider = migrated.get("extraction", {}).get("provider")
    passed = provider == "anthropic"
    status = "PASS" if passed else "FAIL"
    print(f"  extraction.provider added: {status}")
    if not passed:
        print(f"    Expected: 'anthropic', Got: {provider}")
        all_passed = False

    # Check reasoning_effort was added
    effort = migrated.get("extraction", {}).get("reasoning_effort")
    passed = effort is None
    status = "PASS" if passed else "FAIL"
    print(f"  extraction.reasoning_effort added: {status}")
    if not passed:
        print(f"    Expected: None, Got: {effort}")
        all_passed = False

    # Check version was updated
    version = migrated.get("version")
    passed = version == "1.1.0"
    status = "PASS" if passed else "FAIL"
    print(f"  version updated to 1.1.0: {status}")
    if not passed:
        print(f"    Expected: '1.1.0', Got: {version}")
        all_passed = False

    return all_passed


def test_migration_path():
    """Test migration path calculation."""
    print("\n" + "=" * 60)
    print("Testing Migration Path Calculation")
    print("=" * 60)

    from src.config_migration import get_migration_path

    all_passed = True

    # Path from 1.0.0 to 1.1.0 should include 1.1.0
    path = get_migration_path("1.0.0", "1.1.0")
    passed = "1.1.0" in path
    status = "PASS" if passed else "FAIL"
    print(f"  Path from 1.0.0 to 1.1.0 includes 1.1.0: {status}")
    if not passed:
        print(f"    Got path: {path}")
        all_passed = False

    # Path from current to current should be empty
    from src.config_migration import CURRENT_VERSION
    path = get_migration_path(CURRENT_VERSION, CURRENT_VERSION)
    passed = len(path) == 0
    status = "PASS" if passed else "FAIL"
    print(f"  Path from current to current is empty: {status}")
    if not passed:
        print(f"    Got path: {path}")
        all_passed = False

    return all_passed


def test_config_load_with_migration():
    """Test Config.load() with auto-migration."""
    print("\n" + "=" * 60)
    print("Testing Config.load() with Migration")
    print("=" * 60)

    import yaml

    from src.config import Config

    all_passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Create a v1.0.0 config file (no version field)
        old_config = {
            "zotero": {
                "database_path": str(Path(tmpdir) / "test.sqlite"),
                "storage_path": str(Path(tmpdir) / "storage"),
            },
            "extraction": {
                "mode": "cli",
            },
        }

        # Create fake paths
        (Path(tmpdir) / "test.sqlite").touch()
        (Path(tmpdir) / "storage").mkdir()

        with open(config_path, "w") as f:
            yaml.safe_dump(old_config, f)

        # Load config - should auto-migrate
        config = Config.load(config_path)

        # Check that config loaded with current version
        from src.config_migration import CURRENT_VERSION
        passed = config.version == CURRENT_VERSION
        status = "PASS" if passed else "FAIL"
        print(f"  Config loaded with current version: {status}")
        if not passed:
            print(f"    Expected: {CURRENT_VERSION}, Got: {config.version}")
            all_passed = False

        # Check that provider default was set
        passed = config.extraction.provider == "anthropic"
        status = "PASS" if passed else "FAIL"
        print(f"  Provider default applied: {status}")
        if not passed:
            print(f"    Expected: 'anthropic', Got: {config.extraction.provider}")
            all_passed = False

    return all_passed


def test_config_file_backup():
    """Test config file backup before migration."""
    print("\n" + "=" * 60)
    print("Testing Config File Backup")
    print("=" * 60)

    import yaml

    from src.config_migration import backup_config

    all_passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"

        # Create a config file
        config = {"version": "1.0.0", "test": "data"}
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        # Create backup
        backup_path = backup_config(config_path)

        # Check backup exists
        passed = backup_path.exists()
        status = "PASS" if passed else "FAIL"
        print(f"  Backup file created: {status}")
        if not passed:
            all_passed = False

        # Check backup contents match
        if passed:
            with open(backup_path) as f:
                backup_content = yaml.safe_load(f)
            passed = backup_content == config
            status = "PASS" if passed else "FAIL"
            print(f"  Backup contents match original: {status}")
            if not passed:
                all_passed = False

    return all_passed


def main():
    print("LITRIS Config Migration Smoketest")
    print("=" * 60)

    results = {}

    results["version_parsing"] = test_version_parsing()
    results["migration_detection"] = test_migration_detection()
    results["migration_1_0_to_1_1"] = test_migration_1_0_to_1_1()
    results["migration_path"] = test_migration_path()
    results["config_load_migration"] = test_config_load_with_migration()
    results["config_backup"] = test_config_file_backup()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\nConfig migration system is ready!")
    else:
        print("\nSome tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
