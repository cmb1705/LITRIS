#!/usr/bin/env python
"""Reset old PaperExtraction entries so they get re-extracted with 6-pass pipeline.

Removes old-schema (PaperExtraction) entries from the checkpoint and
extractions.json while preserving new-schema (SemanticAnalysis) entries
in semantic_analyses.json and per-pass cache.

Usage:
    python scripts/reset_old_schema.py --dry-run   # Preview what would be reset
    python scripts/reset_old_schema.py              # Execute reset

After running, resume the build to re-extract old-schema papers:
    python scripts/build_index.py --provider openai --mode cli --model gpt-5.5 \
        --use-subscription --resume --gap-fill --gap-fill-threshold 0.90
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset old PaperExtraction entries for re-extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reset without making changes",
    )
    args = parser.parse_args()

    index_dir = Path("data/index")
    ext_path = index_dir / "extractions.json"
    cp_path = index_dir / "checkpoint.json"
    sa_path = index_dir / "semantic_analyses.json"

    if not ext_path.exists():
        print("No extractions.json found. Nothing to reset.")
        return 0

    # Load extractions
    ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
    if isinstance(ext_data, dict) and "extractions" in ext_data:
        extractions = ext_data["extractions"]
        wrapper = ext_data
    elif isinstance(ext_data, dict):
        extractions = ext_data
        wrapper = None
    else:
        print(f"Unexpected extractions.json format: {type(ext_data)}")
        return 1

    # Classify old vs new schema
    old_ids = set()
    new_ids = set()
    for pid, entry in extractions.items():
        ext = entry.get("extraction", {})
        has_q = any(k.startswith("q") and len(k) > 3 and k[1:3].isdigit() for k in ext)
        if has_q:
            new_ids.add(pid)
        else:
            old_ids.add(pid)

    # Load semantic_analyses.json to count preserved work
    sa_count = 0
    if sa_path.exists():
        sa_data = json.loads(sa_path.read_text(encoding="utf-8"))
        sa_count = len(sa_data.get("extractions", {}))

    # Load checkpoint
    cp = json.loads(cp_path.read_text(encoding="utf-8")) if cp_path.exists() else {}
    processed_ids = set(cp.get("processed_ids", []))

    # IDs to remove from checkpoint (old-schema papers)
    # Checkpoint uses paper_id format like "ZOTKEY_ITEMID"
    old_checkpoint_ids = processed_ids & old_ids

    print("=== Schema Reset Analysis ===\n")
    print("extractions.json:")
    print(f"  Old schema (PaperExtraction): {len(old_ids)}")
    print(f"  New schema (SemanticAnalysis): {len(new_ids)}")
    print("")
    print(f"semantic_analyses.json: {sa_count} (preserved)")
    print("")
    print("Checkpoint:")
    print(f"  Total processed IDs: {len(processed_ids)}")
    print(f"  Old-schema IDs to remove: {len(old_checkpoint_ids)}")
    print(f"  Will remain: {len(processed_ids) - len(old_checkpoint_ids)}")
    print("")
    print("After reset:")
    print(f"  Papers needing re-extraction: ~{len(old_ids)}")
    print(f"  Papers already done (new schema): {sa_count + len(new_ids)}")

    if args.dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to execute.")
        return 0

    # Confirm
    try:
        response = input(
            f"\nRemove {len(old_ids)} old-schema entries and reset checkpoint? [y/N]: "
        )
        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = index_dir / "backups"
    backup_dir.mkdir(exist_ok=True)

    shutil.copy2(ext_path, backup_dir / f"extractions_{timestamp}.json")
    if cp_path.exists():
        shutil.copy2(cp_path, backup_dir / f"checkpoint_{timestamp}.json")
    print(f"\nBackups saved to {backup_dir}/")

    # Remove old-schema entries from extractions.json
    for pid in old_ids:
        extractions.pop(pid, None)

    if wrapper is not None:
        wrapper["extractions"] = extractions
        ext_path.write_text(
            json.dumps(wrapper, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    else:
        ext_path.write_text(
            json.dumps(extractions, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print(f"Removed {len(old_ids)} old-schema entries from extractions.json")

    # Reset checkpoint: remove old-schema IDs from processed list
    if cp:
        remaining_ids = [pid for pid in cp.get("processed_ids", []) if pid not in old_ids]
        cp["processed_ids"] = remaining_ids
        cp["processed_count"] = len(remaining_ids)
        cp["success_count"] = len([pid for pid in remaining_ids if pid in new_ids])
        cp["failed_count"] = 0  # Reset failed count; they'll be retried
        cp["last_updated"] = datetime.now().isoformat()
        cp_path.write_text(
            json.dumps(cp, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(
            f"Reset checkpoint: {len(remaining_ids)} IDs remain, {len(old_checkpoint_ids)} removed"
        )

    print(f"\nDone. Resume the build to re-extract {len(old_ids)} papers:")
    print(
        "  python scripts/build_index.py --provider openai --mode cli "
        "--model gpt-5.5 --use-subscription --resume --gap-fill"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
