"""
Upgrade top-level requirements to their latest compatible versions.

The script:
- Loads packages from the provided requirement files (including nested -r files).
- Skips entries pinned with ``==`` to avoid breaking explicit compatibility pins.
- Upgrades only packages declared in the requirement files that are currently
  outdated.
- Runs ``pip check`` after upgrades to verify dependency consistency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Set


def _requirement_class():
    """
    Lazy import of packaging.Requirement, installing packaging if needed.
    """
    try:
        from packaging.requirements import Requirement  # type: ignore
    except ModuleNotFoundError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "packaging"],
            check=True,
        )
        from packaging.requirements import Requirement  # type: ignore
    return Requirement


def parse_requirements(paths: Iterable[Path]) -> list[Any]:
    """
    Parse requirement files and return requirement objects.

    Recurses into ``-r`` includes and ignores blank lines/comments.
    """
    Requirement = _requirement_class()
    result: list[Any] = []
    visited: Set[Path] = set()

    def _parse(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)

        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r"):
                include_target = line[2:].strip()
                if include_target:
                    _parse((resolved.parent / include_target).resolve())
                continue

            requirement_text = line.split("#", 1)[0].strip()
            if requirement_text:
                result.append(Requirement(requirement_text))

    for req_path in paths:
        _parse(req_path)

    return result


def get_outdated() -> dict[str, dict[str, str]]:
    """Return a mapping of lowercase package name to pip outdated entry."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "list",
        "--outdated",
        "--format=json",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    entries = json.loads(completed.stdout)
    return {entry["name"].lower(): entry for entry in entries}


def is_pinned(req: Any) -> bool:
    """Return True if the requirement has an exact pin (== or ===)."""
    return any(spec.operator in {"==", "==="} for spec in req.specifier)


def upgrade_packages(packages: list[str]) -> None:
    """Upgrade the given packages using pip with a conservative strategy."""
    if not packages:
        print("All requirement packages are already up to date.")
        return

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        *packages,
    ]
    print("Upgrading:")
    for name in packages:
        print(f"- {name}")
    subprocess.run(install_cmd, check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade packages declared in requirement files.",
    )
    parser.add_argument(
        "-r",
        "--requirements",
        action="append",
        default=["requirements.txt", "requirements-dev.txt"],
        help="Requirement file to include (can be passed multiple times).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned upgrades without installing them.",
    )
    args = parser.parse_args()

    requirement_paths = [Path(p) for p in args.requirements]
    requirements = parse_requirements(requirement_paths)
    outdated = get_outdated()

    targets: Set[str] = set()
    for req in requirements:
        if is_pinned(req):
            continue
        if req.name.lower() in outdated:
            targets.add(outdated[req.name.lower()]["name"])

    if not targets:
        print("All requirement packages are already up to date.")
        return 0

    print("Planned upgrades:")
    for name in sorted(targets):
        entry = outdated[name.lower()]
        print(f"- {entry['name']}: {entry['version']} -> {entry['latest']}")

    if args.dry_run:
        print("Dry run: no packages were upgraded.")
        return 0

    upgrade_packages(sorted(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
