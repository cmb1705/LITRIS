"""Tests for configuration schema versioning and migration."""

import tempfile
from pathlib import Path

import pytest
import yaml


class TestVersionParsing:
    """Tests for version string parsing."""

    def test_parse_valid_version(self):
        """Should parse valid semantic version."""
        from src.config_migration import parse_version

        assert parse_version("1.0.0") == (1, 0, 0)
        assert parse_version("2.10.5") == (2, 10, 5)

    def test_parse_invalid_version(self):
        """Should reject invalid version formats."""
        from src.config_migration import parse_version

        with pytest.raises(ValueError):
            parse_version("1.0")
        with pytest.raises(ValueError):
            parse_version("1.0.0.0")
        with pytest.raises(ValueError):
            parse_version("v1.0.0")


class TestVersionComparison:
    """Tests for version comparison."""

    def test_compare_equal_versions(self):
        """Equal versions should return 0."""
        from src.config_migration import compare_versions

        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("2.5.3", "2.5.3") == 0

    def test_compare_less_than(self):
        """Older version should return -1."""
        from src.config_migration import compare_versions

        assert compare_versions("1.0.0", "1.0.1") == -1
        assert compare_versions("1.0.0", "1.1.0") == -1
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_compare_greater_than(self):
        """Newer version should return 1."""
        from src.config_migration import compare_versions

        assert compare_versions("1.0.1", "1.0.0") == 1
        assert compare_versions("1.1.0", "1.0.0") == 1
        assert compare_versions("2.0.0", "1.9.9") == 1


class TestConfigVersionDetection:
    """Tests for config version detection."""

    def test_config_with_version(self):
        """Should return explicit version."""
        from src.config_migration import get_config_version

        config = {"version": "1.2.3"}
        assert get_config_version(config) == "1.2.3"

    def test_config_without_version(self):
        """Should default to 1.0.0 for unversioned configs."""
        from src.config_migration import get_config_version

        config = {"zotero": {}}
        assert get_config_version(config) == "1.0.0"

    def test_needs_migration_old_config(self):
        """Old config should need migration."""
        from src.config_migration import needs_migration

        config = {"version": "1.0.0"}
        assert needs_migration(config) is True

    def test_needs_migration_current_config(self):
        """Current config should not need migration."""
        from src.config_migration import CURRENT_VERSION, needs_migration

        config = {"version": CURRENT_VERSION}
        assert needs_migration(config) is False


class TestMigrationPath:
    """Tests for migration path calculation."""

    def test_path_same_version(self):
        """Same version should have empty path."""
        from src.config_migration import get_migration_path

        assert get_migration_path("1.0.0", "1.0.0") == []
        assert get_migration_path("1.1.0", "1.1.0") == []

    def test_path_older_to_newer(self):
        """Should include intermediate versions."""
        from src.config_migration import get_migration_path

        path = get_migration_path("1.0.0", "1.1.0")
        assert "1.1.0" in path

    def test_path_newer_to_older(self):
        """Downgrade should have empty path."""
        from src.config_migration import get_migration_path

        assert get_migration_path("1.1.0", "1.0.0") == []


class TestMigration_1_0_to_1_1:
    """Tests for 1.0.0 to 1.1.0 migration."""

    def test_adds_provider_field(self):
        """Migration should add provider field."""
        from src.config_migration import migrate_config

        config = {
            "version": "1.0.0",
            "extraction": {"mode": "cli"},
        }
        result = migrate_config(config, "1.1.0")
        assert result["extraction"]["provider"] == "anthropic"

    def test_adds_reasoning_effort_field(self):
        """Migration should add reasoning_effort field."""
        from src.config_migration import migrate_config

        config = {
            "version": "1.0.0",
            "extraction": {"mode": "cli"},
        }
        result = migrate_config(config, "1.1.0")
        assert result["extraction"]["reasoning_effort"] is None

    def test_preserves_existing_fields(self):
        """Migration should preserve existing fields."""
        from src.config_migration import migrate_config

        config = {
            "version": "1.0.0",
            "extraction": {"mode": "api", "model": "claude-3"},
            "zotero": {"database_path": "/test"},
        }
        result = migrate_config(config, "1.1.0")
        assert result["extraction"]["mode"] == "api"
        assert result["extraction"]["model"] == "claude-3"
        assert result["zotero"]["database_path"] == "/test"

    def test_updates_version(self):
        """Migration should update version field."""
        from src.config_migration import migrate_config

        config = {"version": "1.0.0"}
        result = migrate_config(config, "1.1.0")
        assert result["version"] == "1.1.0"


class TestMigrateConfig:
    """Tests for config migration function."""

    def test_migrate_unversioned_config(self):
        """Should migrate config without version field."""
        from src.config_migration import CURRENT_VERSION, migrate_config

        config = {"extraction": {"mode": "cli"}}
        result = migrate_config(config)
        assert result["version"] == CURRENT_VERSION
        assert result["extraction"]["provider"] == "anthropic"

    def test_skip_migration_if_current(self):
        """Should not modify current version config."""
        from src.config_migration import CURRENT_VERSION, migrate_config

        config = {"version": CURRENT_VERSION, "custom": "value"}
        result = migrate_config(config)
        assert result == config

    def test_reject_too_old_version(self):
        """Should reject versions older than minimum supported."""
        from src.config_migration import migrate_config

        config = {"version": "0.1.0"}
        with pytest.raises(ValueError, match="too old"):
            migrate_config(config)


class TestConfigBackup:
    """Tests for config backup functionality."""

    def test_creates_backup_file(self):
        """Should create backup of config file."""
        from src.config_migration import backup_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("test: data")

            backup_path = backup_config(config_path)
            assert backup_path.exists()
            assert backup_path.read_text() == "test: data"

    def test_backup_unique_names(self):
        """Should create unique backup names."""
        from src.config_migration import backup_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("test: data")

            backup1 = backup_config(config_path)
            backup2 = backup_config(config_path)
            assert backup1 != backup2
            assert backup1.exists()
            assert backup2.exists()


class TestConfigMigrateFile:
    """Tests for in-place file migration."""

    def test_migrate_file_updates_content(self):
        """Should update config file content."""
        from src.config_migration import CURRENT_VERSION, migrate_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            old_config = {"version": "1.0.0", "extraction": {"mode": "cli"}}
            with open(config_path, "w") as f:
                yaml.safe_dump(old_config, f)

            migrate_config_file(config_path, backup=False)

            with open(config_path) as f:
                result = yaml.safe_load(f)

            assert result["version"] == CURRENT_VERSION
            assert result["extraction"]["provider"] == "anthropic"

    def test_migrate_file_creates_backup(self):
        """Should create backup when requested."""
        from src.config_migration import migrate_config_file

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            old_config = {"version": "1.0.0"}
            with open(config_path, "w") as f:
                yaml.safe_dump(old_config, f)

            migrate_config_file(config_path, backup=True)

            backup_files = list(Path(tmpdir).glob("*.bak*"))
            assert len(backup_files) >= 1


class TestConfigLoadWithMigration:
    """Tests for Config.load() with migration support."""

    def test_load_migrates_old_config(self):
        """Should auto-migrate old config on load."""
        from src.config import Config
        from src.config_migration import CURRENT_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create required files
            db_path = Path(tmpdir) / "test.sqlite"
            storage_path = Path(tmpdir) / "storage"
            db_path.touch()
            storage_path.mkdir()

            old_config = {
                "zotero": {
                    "database_path": str(db_path),
                    "storage_path": str(storage_path),
                },
                "extraction": {"mode": "cli"},
            }
            with open(config_path, "w") as f:
                yaml.safe_dump(old_config, f)

            config = Config.load(config_path)
            assert config.version == CURRENT_VERSION
            assert config.extraction.provider == "anthropic"

    def test_load_preserves_current_config(self):
        """Should not modify current version configs."""
        from src.config import Config
        from src.config_migration import CURRENT_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create required files
            db_path = Path(tmpdir) / "test.sqlite"
            storage_path = Path(tmpdir) / "storage"
            db_path.touch()
            storage_path.mkdir()

            current_config = {
                "version": CURRENT_VERSION,
                "zotero": {
                    "database_path": str(db_path),
                    "storage_path": str(storage_path),
                },
                "extraction": {"provider": "openai", "mode": "api"},
            }
            with open(config_path, "w") as f:
                yaml.safe_dump(current_config, f)

            config = Config.load(config_path)
            assert config.extraction.provider == "openai"

    def test_load_disable_auto_migrate(self):
        """Should skip migration when auto_migrate=False."""
        from src.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            # Create required files
            db_path = Path(tmpdir) / "test.sqlite"
            storage_path = Path(tmpdir) / "storage"
            db_path.touch()
            storage_path.mkdir()

            old_config = {
                "version": "1.0.0",
                "zotero": {
                    "database_path": str(db_path),
                    "storage_path": str(storage_path),
                },
                "extraction": {"mode": "cli", "provider": "anthropic"},
            }
            with open(config_path, "w") as f:
                yaml.safe_dump(old_config, f)

            config = Config.load(config_path, auto_migrate=False)
            # When auto_migrate=False, version should be preserved from file
            assert config.version == "1.0.0"


class TestConfigVersion:
    """Tests for Config version field."""

    def test_config_has_version_field(self):
        """Config model should have version field."""
        from src.config import Config
        from src.config_migration import CURRENT_VERSION

        # Check field exists in model
        assert "version" in Config.model_fields

        # Check default value
        assert Config.model_fields["version"].default == CURRENT_VERSION
