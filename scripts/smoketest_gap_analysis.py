#!/usr/bin/env python
"""Smoketest for gap analysis pipeline."""

import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.file_utils import safe_write_json  # noqa: E402


def _seed_index(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    papers = [
        {
            "paper_id": "p1",
            "title": "Topic A study",
            "abstract": "Qualitative investigation of topic A.",
            "publication_year": 2018,
            "collections": ["core"],
        },
        {
            "paper_id": "p2",
            "title": "Topic B overview",
            "abstract": "Survey of topic B.",
            "publication_year": 2020,
            "collections": ["core"],
        },
    ]
    extractions = {
        "p1": {
            "paper_id": "p1",
            "extraction": {
                "discipline_tags": ["Topic A"],
                "keywords": ["alpha"],
                "methodology": {"approach": "Qualitative"},
                "future_directions": ["Explore beta networks"],
            },
        },
        "p2": {
            "paper_id": "p2",
            "extraction": {
                "discipline_tags": ["Topic B"],
                "keywords": ["beta"],
                "methodology": {"approach": "Quantitative"},
                "future_directions": ["Investigate gamma datasets"],
            },
        },
    }
    safe_write_json(index_dir / "papers.json", {"papers": papers})
    safe_write_json(index_dir / "extractions.json", {"extractions": extractions})


def main() -> int:
    print("LITRIS Gap Analysis Smoketest")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        index_dir = tmpdir / "index"
        output_dir = tmpdir / "out"
        _seed_index(index_dir)

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "gap_analysis.py"),
            "--index-dir",
            str(index_dir),
            "--output-dir",
            str(output_dir),
            "--output-format",
            "markdown",
            "--max-items",
            "5",
            "--min-count",
            "1",
            "--quantile",
            "0.0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            print("Smoketest FAILED")
            return 1

        outputs = list(output_dir.glob("*.md"))
        if not outputs:
            print("No markdown outputs found.")
            return 1

        print(f"Smoketest OK: {outputs[0].name}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
