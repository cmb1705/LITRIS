#!/usr/bin/env python
"""Extract key learnings from Claude Code session transcripts.

Parses JSONL transcript files and extracts patterns, troubleshooting info,
and project knowledge for loading into memory MCP.

Enhanced with pattern matching for:
- Technical patterns (encoding, architecture, compatibility)
- Troubleshooting solutions (error -> fix pairs)
- Architecture insights (component relationships)
- Configuration learnings (hooks, settings, MCP)
"""

import json
import re
from collections import defaultdict
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


# Pattern definitions for knowledge extraction
TECHNICAL_PATTERNS = [
    # Windows compatibility
    (r"encoding\s*=\s*['\"]utf-8['\"]", "windows_encoding", "Use encoding='utf-8' for Windows compatibility"),
    (r"errors\s*=\s*['\"]replace['\"]", "error_handling", "Use errors='replace' to handle unencodable chars"),
    (r"2>NUL", "windows_redirect", "Use 2>NUL instead of 2>/dev/null on Windows"),
    # Architecture patterns
    (r"cli_executor\.extract\s*\(", "cli_extraction", "Use cli_executor.extract(prompt, text) for paper extraction"),
    (r"call_with_prompt\s*\(", "cli_combined", "call_with_prompt() combines prompt and text differently"),
    # MCP patterns
    (r"mcp__\w+__\w+", "mcp_tool", "MCP tool naming convention: mcp__server__tool"),
    # Hook patterns
    (r"PostToolUse|PreToolUse|SessionStart", "hook_event", "Claude Code hook event types"),
]

TROUBLESHOOTING_PATTERNS = [
    # Error types
    (r"UnicodeEncodeError|UnicodeDecodeError", "unicode_error"),
    (r"AttributeError:\s*'(\w+)'\s*object", "attribute_error"),
    (r"KeyError:\s*['\"](\w+)['\"]", "key_error"),
    (r"FileNotFoundError", "file_not_found"),
    (r"rate.?limit|429|too.?many.?requests", "rate_limit"),
    (r"auth.*fail|token.*expir", "auth_error"),
]

SOLUTION_INDICATORS = [
    r"\bfix(?:ed|es)?\b",
    r"\bsolv(?:ed|es)?\b",
    r"\bresol(?:ved|ution)\b",
    r"\bworkaround\b",
    r"\bsolution\b",
    r"the\s+(?:issue|problem|error)\s+(?:was|is)",
]


def find_patterns(messages: list[dict]) -> dict:
    """Find common patterns and learnings in messages."""
    patterns = {
        "errors_fixed": [],
        "commands_used": [],
        "files_modified": [],
        "troubleshooting": [],
        "technical_patterns": defaultdict(list),
        "error_solution_pairs": [],
        "architecture_insights": [],
        "configuration_learnings": [],
    }

    # Combine solution indicators into one regex
    solution_regex = re.compile("|".join(SOLUTION_INDICATORS), re.IGNORECASE)

    for i, msg in enumerate(messages):
        text = msg["text"]

        # Find technical patterns
        for pattern, category, description in TECHNICAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if description not in patterns["technical_patterns"][category]:
                    patterns["technical_patterns"][category].append(description)

        # Find error-solution pairs (error in one message, solution in next)
        for error_pattern, error_type in TROUBLESHOOTING_PATTERNS:
            if re.search(error_pattern, text, re.IGNORECASE):
                # Look for solution in nearby messages
                context_start = max(0, i - 1)
                context_end = min(len(messages), i + 3)
                for j in range(context_start, context_end):
                    if solution_regex.search(messages[j]["text"]):
                        pair = {
                            "error_type": error_type,
                            "error_context": text[:300],
                            "solution_context": messages[j]["text"][:500],
                        }
                        # Avoid duplicates
                        if not any(
                            p["error_type"] == error_type and p["error_context"][:100] == pair["error_context"][:100]
                            for p in patterns["error_solution_pairs"]
                        ):
                            patterns["error_solution_pairs"].append(pair)
                        break

        # Find error patterns with fixes (legacy)
        if "Error" in text or "error" in text:
            if solution_regex.search(text):
                patterns["errors_fixed"].append(text[:500])

        # Find commands
        cmd_matches = re.findall(r"`([^`]+)`", text)
        for cmd in cmd_matches:
            if any(cmd.startswith(prefix) for prefix in ["python ", "pip ", "git ", "bd ", "claude "]):
                if cmd not in patterns["commands_used"]:
                    patterns["commands_used"].append(cmd)

        # Find file modifications
        file_matches = re.findall(r"(src/[^\s\]]+\.py|scripts/[^\s\]]+\.py)", text)
        for f in file_matches:
            # Clean up any trailing characters
            f = re.sub(r"[)\]\.,;:]+$", "", f)
            if f not in patterns["files_modified"]:
                patterns["files_modified"].append(f)

        # Find architecture insights (component relationships)
        arch_patterns = [
            r"(\w+)\s+(?:uses|calls|depends on|imports)\s+(\w+)",
            r"(\w+)\s+(?:->|-->|=>)\s+(\w+)",
        ]
        for arch_pattern in arch_patterns:
            arch_matches = re.findall(arch_pattern, text)
            for match in arch_matches:
                if len(match) == 2 and match[0] != match[1]:
                    insight = f"{match[0]} -> {match[1]}"
                    if insight not in patterns["architecture_insights"]:
                        patterns["architecture_insights"].append(insight)

        # Find configuration learnings
        config_patterns = [
            (r"settings\.json", "Claude Code settings"),
            (r"\.mcp\.json", "MCP server config"),
            (r"config\.yaml", "Project config"),
            (r"\.claude/", "Claude directory structure"),
        ]
        for config_pattern, config_type in config_patterns:
            if re.search(config_pattern, text):
                # Extract relevant context
                lines = text.split("\n")
                for line in lines:
                    if re.search(config_pattern, line) and len(line) < 200:
                        learning = f"{config_type}: {line.strip()}"
                        if learning not in patterns["configuration_learnings"]:
                            patterns["configuration_learnings"].append(learning)

    # Convert defaultdict to regular dict
    patterns["technical_patterns"] = dict(patterns["technical_patterns"])

    return patterns


def format_for_memory_mcp(all_patterns: dict) -> dict:
    """Format extracted patterns for memory MCP loading."""
    entities = []
    relations = []

    # Create entities from technical patterns
    for category, descriptions in all_patterns.get("technical_patterns", {}).items():
        for desc in descriptions:
            entities.append({
                "name": f"pattern_{category}",
                "entityType": "technical_pattern",
                "observations": [desc],
            })

    # Create entities from error-solution pairs
    for pair in all_patterns.get("error_solution_pairs", []):
        entity_name = f"troubleshooting_{pair['error_type']}"
        # Check if entity already exists
        existing = next((e for e in entities if e["name"] == entity_name), None)
        if existing:
            existing["observations"].append(pair["solution_context"][:300])
        else:
            entities.append({
                "name": entity_name,
                "entityType": "troubleshooting",
                "observations": [
                    f"Error: {pair['error_context'][:200]}",
                    f"Solution: {pair['solution_context'][:300]}",
                ],
            })

    # Create entities from files modified
    file_categories = defaultdict(list)
    for f in all_patterns.get("files_modified", []):
        if f.startswith("src/analysis/"):
            file_categories["analysis_module"].append(f)
        elif f.startswith("src/indexing/"):
            file_categories["indexing_module"].append(f)
        elif f.startswith("src/query/"):
            file_categories["query_module"].append(f)
        elif f.startswith("src/zotero/"):
            file_categories["zotero_module"].append(f)
        elif f.startswith("scripts/"):
            file_categories["scripts"].append(f)

    for category, files in file_categories.items():
        entities.append({
            "name": f"litris_{category}",
            "entityType": "code_module",
            "observations": files[:10],  # Limit to 10 files per category
        })

    # Create relations between modules
    module_relations = [
        ("litris_analysis_module", "processes", "litris_indexing_module"),
        ("litris_query_module", "searches", "litris_indexing_module"),
        ("litris_zotero_module", "provides_data_to", "litris_analysis_module"),
    ]
    for from_entity, relation_type, to_entity in module_relations:
        if any(e["name"] == from_entity for e in entities) and any(e["name"] == to_entity for e in entities):
            relations.append({
                "from": from_entity,
                "relationType": relation_type,
                "to": to_entity,
            })

    return {
        "entities": entities,
        "relations": relations,
    }


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
        "--memory-output",
        type=Path,
        default=Path("data/memory_mcp_entities.json"),
        help="Output JSON file formatted for memory MCP",
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
        "technical_patterns": defaultdict(list),
        "error_solution_pairs": [],
        "architecture_insights": [],
        "configuration_learnings": [],
        "transcripts_processed": 0,
    }

    for transcript in transcripts[: args.limit]:
        print(f"Processing: {transcript.name}")
        messages = extract_messages(transcript)
        patterns = find_patterns(messages)

        # Merge patterns
        for key in ["errors_fixed", "commands_used", "files_modified", "troubleshooting",
                    "error_solution_pairs", "architecture_insights", "configuration_learnings"]:
            for item in patterns.get(key, []):
                if item not in all_patterns[key]:
                    all_patterns[key].append(item)

        # Merge technical patterns
        for category, descriptions in patterns.get("technical_patterns", {}).items():
            for desc in descriptions:
                if desc not in all_patterns["technical_patterns"][category]:
                    all_patterns["technical_patterns"][category].append(desc)

        all_patterns["transcripts_processed"] += 1

    # Convert defaultdict to regular dict for JSON serialization
    all_patterns["technical_patterns"] = dict(all_patterns["technical_patterns"])

    # Save raw results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_patterns, f, indent=2, ensure_ascii=False)

    # Save memory MCP formatted results
    memory_data = format_for_memory_mcp(all_patterns)
    with open(args.memory_output, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted knowledge saved to: {args.output}")
    print(f"Memory MCP entities saved to: {args.memory_output}")
    print(f"  Commands found: {len(all_patterns['commands_used'])}")
    print(f"  Files modified: {len(all_patterns['files_modified'])}")
    print(f"  Errors documented: {len(all_patterns['errors_fixed'])}")
    print(f"  Technical patterns: {sum(len(v) for v in all_patterns['technical_patterns'].values())}")
    print(f"  Error-solution pairs: {len(all_patterns['error_solution_pairs'])}")
    print(f"  Memory entities: {len(memory_data['entities'])}")
    print(f"  Memory relations: {len(memory_data['relations'])}")

    return 0


if __name__ == "__main__":
    exit(main())
