#!/usr/bin/env python
"""Extract key learnings from Claude Code session transcripts.

Parses JSONL transcript files and extracts patterns, troubleshooting info,
and project knowledge for loading into memory MCP.
"""

import json
import re
from pathlib import Path


def extract_messages(jsonl_path: Path) -> list[dict]:
    """Extract user and assistant messages from a transcript."""
    messages = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("type") in ("user", "assistant"):
                    msg = entry.get("message", {})
                    content = msg.get("content", [])
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    if text_parts:
                        messages.append({
                            "role": entry["type"],
                            "text": "\n".join(text_parts),
                            "timestamp": entry.get("timestamp", ""),
                        })
            except json.JSONDecodeError:
                continue
    return messages


def find_patterns(messages: list[dict]) -> dict:
    """Find common patterns and learnings in messages."""
    patterns = {
        "errors_fixed": [],
        "commands_used": [],
        "files_modified": [],
        "troubleshooting": [],
    }

    for msg in messages:
        text = msg["text"]

        # Find error patterns
        if "Error" in text or "error" in text:
            # Look for fix patterns
            if "fix" in text.lower() or "solved" in text.lower():
                patterns["errors_fixed"].append(text[:500])

        # Find commands
        cmd_matches = re.findall(r'`([^`]+)`', text)
        for cmd in cmd_matches:
            if cmd.startswith("python ") or cmd.startswith("pip ") or cmd.startswith("git "):
                if cmd not in patterns["commands_used"]:
                    patterns["commands_used"].append(cmd)

        # Find file modifications
        file_matches = re.findall(r'(src/[^\s]+\.py|scripts/[^\s]+\.py)', text)
        for f in file_matches:
            if f not in patterns["files_modified"]:
                patterns["files_modified"].append(f)

    return patterns


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract knowledge from Claude transcripts")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.home() / ".claude/projects/d--Git-Repos-LITRIS",
        help="Path to Claude project transcripts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/extracted_knowledge.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max transcripts to process",
    )
    args = parser.parse_args()

    if not args.project_dir.exists():
        print(f"Project directory not found: {args.project_dir}")
        return 1

    transcripts = sorted(args.project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"Found {len(transcripts)} transcripts, processing {min(len(transcripts), args.limit)}")

    all_patterns = {
        "errors_fixed": [],
        "commands_used": [],
        "files_modified": [],
        "troubleshooting": [],
        "transcripts_processed": 0,
    }

    for transcript in transcripts[:args.limit]:
        print(f"Processing: {transcript.name}")
        messages = extract_messages(transcript)
        patterns = find_patterns(messages)

        for key in ["errors_fixed", "commands_used", "files_modified", "troubleshooting"]:
            for item in patterns[key]:
                if item not in all_patterns[key]:
                    all_patterns[key].append(item)

        all_patterns["transcripts_processed"] += 1

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_patterns, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted knowledge saved to: {args.output}")
    print(f"  Commands found: {len(all_patterns['commands_used'])}")
    print(f"  Files modified: {len(all_patterns['files_modified'])}")
    print(f"  Errors documented: {len(all_patterns['errors_fixed'])}")

    return 0


if __name__ == "__main__":
    exit(main())
