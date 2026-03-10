"""Heuristic gap analysis over indexed literature metadata and extractions."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.file_utils import safe_read_json, safe_write_json

_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")

_STOPWORDS = {
    "about",
    "across",
    "after",
    "among",
    "analysis",
    "approach",
    "based",
    "between",
    "case",
    "data",
    "design",
    "effects",
    "example",
    "from",
    "future",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "research",
    "results",
    "study",
    "studies",
    "using",
    "with",
}


@dataclass(frozen=True)
class GapDetectionConfig:
    """Configuration for heuristic gap detection.

    Memory: The analysis holds token maps and counters for all papers and
    extractions in memory (O(n) with corpus size).
    """

    max_items: int = 10
    min_count: int = 2
    quantile: float = 0.2
    evidence_limit: int = 3
    include_abstracts: bool = False
    sparse_year_max_count: int = 1
    future_direction_min_mentions: int = 1
    future_direction_max_coverage: int = 1
    token_min_len: int = 3


def analyze_gap_report(
    papers: Iterable[dict],
    extractions: dict[str, dict],
    config: GapDetectionConfig,
    collections: list[str] | None = None,
) -> dict:
    """Analyze the corpus for potential research gaps.

    Args:
        papers: Iterable of paper metadata dictionaries.
        extractions: Mapping of paper_id to extraction dictionaries.
        config: Gap detection configuration.
        collections: Optional collection filter.

    Returns:
        Gap analysis report dictionary.

    Memory:
        O(n) for papers/extractions plus token maps for coverage checks.
    """
    papers_list = _filter_papers(papers, collections)
    paper_lookup = {p.get("paper_id"): p for p in papers_list if p.get("paper_id")}

    topic_counts: Counter[str] = Counter()
    topic_evidence: dict[str, list[dict]] = defaultdict(list)
    method_counts: Counter[str] = Counter()
    method_evidence: dict[str, list[dict]] = defaultdict(list)
    year_counts: Counter[int] = Counter()

    token_index = _build_token_index(
        papers_list,
        extractions,
        include_abstracts=config.include_abstracts,
        token_min_len=config.token_min_len,
    )
    future_direction_records = _collect_future_directions(
        papers_list,
        extractions,
        config=config,
    )

    for paper in papers_list:
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue
        extraction = extractions.get(paper_id, {})
        ext_data = extraction.get("extraction", extraction)
        _count_topics(topic_counts, topic_evidence, ext_data, paper, config)
        _count_methods(method_counts, method_evidence, ext_data, paper, config)
        _count_year(year_counts, paper)

    topic_gaps = _select_underrepresented(
        topic_counts,
        topic_evidence,
        config,
    )
    method_gaps = _select_underrepresented(
        method_counts,
        method_evidence,
        config,
    )
    year_gaps = _summarize_year_gaps(
        year_counts,
        sparse_year_max_count=config.sparse_year_max_count,
        max_items=config.max_items,
    )
    future_gaps = _evaluate_future_directions(
        future_direction_records,
        token_index,
        config,
        paper_lookup,
    )

    extractions_in_scope = sum(
        1 for paper_id in paper_lookup if paper_id in extractions
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "corpus": {
            "papers": len(papers_list),
            "extractions": extractions_in_scope,
            "collections": collections or [],
        },
        "parameters": {
            "max_items": config.max_items,
            "min_count": config.min_count,
            "quantile": config.quantile,
            "include_abstracts": config.include_abstracts,
            "sparse_year_max_count": config.sparse_year_max_count,
            "future_direction_min_mentions": config.future_direction_min_mentions,
            "future_direction_max_coverage": config.future_direction_max_coverage,
        },
        "topics_underrepresented": topic_gaps,
        "methodologies_underrepresented": method_gaps,
        "year_gaps": year_gaps,
        "future_directions": future_gaps,
        "notes": _build_notes(papers_list, extractions_in_scope, topic_gaps, method_gaps),
    }
    return report


def load_gap_report(
    index_dir: Path,
    config: GapDetectionConfig,
    collections: list[str] | None = None,
) -> dict:
    """Load index artifacts and run gap analysis."""
    papers = _load_papers(index_dir)
    extractions = _load_extractions(index_dir)
    return analyze_gap_report(papers, extractions, config, collections=collections)


def format_gap_report_markdown(report: dict) -> str:
    """Format gap analysis report as Markdown."""
    corpus = report.get("corpus", {})
    params = report.get("parameters", {})
    lines = [
        "# Gap Analysis Report",
        "",
        f"**Generated:** {report.get('generated_at', 'Unknown')}",
        f"**Papers analyzed:** {corpus.get('papers', 0)}",
        f"**Extractions available:** {corpus.get('extractions', 0)}",
        "",
        "## Parameters",
        "",
        f"- Max items: {params.get('max_items')}",
        f"- Min count: {params.get('min_count')}",
        f"- Quantile: {params.get('quantile')}",
        f"- Include abstracts: {params.get('include_abstracts')}",
        "",
        "## Underrepresented Topics",
        "",
    ]

    lines.extend(_format_gap_items(report.get("topics_underrepresented", [])))

    lines.extend(
        [
            "",
            "## Underrepresented Methodologies",
            "",
        ]
    )
    lines.extend(_format_gap_items(report.get("methodologies_underrepresented", [])))

    lines.extend(
        [
            "",
            "## Time Period Gaps",
            "",
        ]
    )
    lines.extend(_format_year_gaps(report.get("year_gaps", {})))

    lines.extend(
        [
            "",
            "## Future Directions With Low Coverage",
            "",
        ]
    )
    lines.extend(_format_future_gaps(report.get("future_directions", [])))

    notes = report.get("notes", [])
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def save_gap_report(report: dict, output_dir: Path, output_format: str) -> Path:
    """Save the gap analysis report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    ext = "json" if output_format == "json" else "md"
    filename = f"{date_str}_gap_analysis.{ext}"
    output_path = output_dir / filename

    if output_format == "json":
        safe_write_json(output_path, report)
    else:
        output_path.write_text(format_gap_report_markdown(report), encoding="utf-8")

    latest_path = output_dir / f"latest.{ext}"
    if output_format == "json":
        safe_write_json(latest_path, report)
    else:
        latest_path.write_text(format_gap_report_markdown(report), encoding="utf-8")

    return output_path


def _filter_papers(
    papers: Iterable[dict],
    collections: list[str] | None,
) -> list[dict]:
    if not collections:
        return list(papers)

    selected = []
    for paper in papers:
        paper_colls = set(paper.get("collections", []) or [])
        if any(c in paper_colls for c in collections):
            selected.append(paper)
    return selected


def _load_papers(index_dir: Path) -> list[dict]:
    papers_path = index_dir / "papers.json"
    data = safe_read_json(papers_path, default={"papers": []})
    if isinstance(data, dict) and "papers" in data:
        return data.get("papers", [])
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    return []


def _load_extractions(index_dir: Path) -> dict[str, dict]:
    extractions_path = index_dir / "extractions.json"
    data = safe_read_json(extractions_path, default={})
    if isinstance(data, dict) and "extractions" in data:
        data = data.get("extractions", {})
    if isinstance(data, list):
        return {e["paper_id"]: e for e in data if "paper_id" in e}
    if isinstance(data, dict):
        return data
    return {}


def _normalize_label(value: str) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _tokenize(text: str, min_len: int) -> list[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [
        t for t in tokens if len(t) >= min_len and t not in _STOPWORDS
    ]


def _count_topics(
    counter: Counter[str],
    evidence: dict[str, list[dict]],
    extraction: dict,
    paper: dict,
    config: GapDetectionConfig,
) -> None:
    tags = extraction.get("discipline_tags") or []
    keywords = extraction.get("keywords") or []
    for value in tags + keywords:
        label = _normalize_label(str(value))
        if not label:
            continue
        counter[label] += 1
        _add_evidence(evidence[label], paper, config.evidence_limit)


def _count_methods(
    counter: Counter[str],
    evidence: dict[str, list[dict]],
    extraction: dict,
    paper: dict,
    config: GapDetectionConfig,
) -> None:
    method = extraction.get("methodology") or {}
    if not isinstance(method, dict):
        method = {"approach": str(method)}

    fields = {
        "approach": method.get("approach"),
        "design": method.get("design"),
        "time_period": method.get("time_period"),
    }
    for field, value in fields.items():
        if value:
            label = _normalize_label(f"{field}: {value}")
            counter[label] += 1
            _add_evidence(evidence[label], paper, config.evidence_limit)

    for field in ("analysis_methods", "data_sources"):
        values = method.get(field) or []
        for value in values:
            label = _normalize_label(f"{field}: {value}")
            if not label:
                continue
            counter[label] += 1
            _add_evidence(evidence[label], paper, config.evidence_limit)


def _count_year(counter: Counter[int], paper: dict) -> None:
    year_value = paper.get("publication_year")
    if year_value and str(year_value).isdigit():
        counter[int(year_value)] += 1


def _add_evidence(
    evidence_list: list[dict],
    paper: dict,
    limit: int,
) -> None:
    if len(evidence_list) >= limit:
        return
    evidence_list.append(
        {
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title") or "Unknown",
            "year": paper.get("publication_year"),
        }
    )


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    q = max(0.0, min(1.0, q))
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def _select_underrepresented(
    counter: Counter[str],
    evidence: dict[str, list[dict]],
    config: GapDetectionConfig,
) -> list[dict]:
    if not counter:
        return []
    threshold = max(config.min_count, _quantile(list(counter.values()), config.quantile))
    candidates = [
        (label, count)
        for label, count in counter.items()
        if count <= threshold
    ]
    candidates.sort(key=lambda x: (x[1], x[0]))
    output = []
    for label, count in candidates[: config.max_items]:
        output.append(
            {
                "label": label,
                "count": count,
                "evidence": evidence.get(label, [])[: config.evidence_limit],
            }
        )
    return output


def _summarize_year_gaps(
    counts: Counter[int],
    sparse_year_max_count: int,
    max_items: int,
) -> dict:
    if not counts:
        return {
            "min_year": None,
            "max_year": None,
            "missing_ranges": [],
            "sparse_years": [],
        }

    min_year = min(counts)
    max_year = max(counts)
    missing = [year for year in range(min_year, max_year + 1) if counts.get(year, 0) == 0]
    ranges = _collapse_year_ranges(missing)
    ranges.sort(key=lambda r: (-r["length"], r["start"]))

    sparse = sorted(
        [
            {"year": year, "count": count}
            for year, count in counts.items()
            if count <= sparse_year_max_count
        ],
        key=lambda x: (x["year"]),
    )
    return {
        "min_year": min_year,
        "max_year": max_year,
        "missing_ranges": ranges[:max_items],
        "sparse_years": sparse[:max_items],
    }


def _collapse_year_ranges(years: list[int]) -> list[dict]:
    if not years:
        return []
    years = sorted(years)
    ranges = []
    start = prev = years[0]
    for year in years[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append({"start": start, "end": prev, "length": prev - start + 1})
        start = prev = year
    ranges.append({"start": start, "end": prev, "length": prev - start + 1})
    return ranges


def _build_token_index(
    papers: list[dict],
    extractions: dict[str, dict],
    include_abstracts: bool,
    token_min_len: int,
) -> dict[str, set[str]]:
    token_index: dict[str, set[str]] = defaultdict(set)
    for paper in papers:
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue
        text_parts = [paper.get("title", "")]
        if include_abstracts:
            text_parts.append(paper.get("abstract", ""))
        extraction = extractions.get(paper_id, {})
        ext_data = extraction.get("extraction", extraction)
        text_parts.extend(ext_data.get("keywords", []) or [])
        text_parts.extend(ext_data.get("discipline_tags", []) or [])
        tokens = set(_tokenize(" ".join(map(str, text_parts)), token_min_len))
        for token in tokens:
            token_index[token].add(paper_id)
    return token_index


def _collect_future_directions(
    papers: list[dict],
    extractions: dict[str, dict],
    config: GapDetectionConfig,
) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for paper in papers:
        paper_id = paper.get("paper_id")
        if not paper_id:
            continue
        extraction = extractions.get(paper_id, {})
        ext_data = extraction.get("extraction", extraction)
        directions = ext_data.get("future_directions") or []
        for direction in directions:
            direction_text = str(direction).strip()
            if not direction_text:
                continue
            normalized = _normalize_label(direction_text)
            if normalized not in records:
                records[normalized] = {
                    "direction": direction_text,
                    "mentions": set(),
                    "tokens": _tokenize(direction_text, config.token_min_len)[:4],
                }
            records[normalized]["mentions"].add(paper_id)
    return records


def _evaluate_future_directions(
    records: dict[str, dict],
    token_index: dict[str, set[str]],
    config: GapDetectionConfig,
    paper_lookup: dict[str, dict],
) -> list[dict]:
    gaps = []
    for record in records.values():
        mention_count = len(record["mentions"])
        if mention_count < config.future_direction_min_mentions:
            continue
        tokens = record.get("tokens") or []
        if not tokens:
            continue
        coverage = _coverage_for_tokens(tokens, token_index)
        if coverage <= config.future_direction_max_coverage:
            gaps.append(
                {
                    "direction": record["direction"],
                    "mention_count": mention_count,
                    "coverage_count": coverage,
                    "evidence": _evidence_from_ids(
                        record["mentions"],
                        paper_lookup,
                        config.evidence_limit,
                    ),
                }
            )
    gaps.sort(key=lambda x: (-x["mention_count"], x["direction"]))
    return gaps[: config.max_items]


def _coverage_for_tokens(tokens: list[str], token_index: dict[str, set[str]]) -> int:
    coverage_set: set[str] | None = None
    for token in tokens:
        candidates = token_index.get(token, set())
        if coverage_set is None:
            coverage_set = set(candidates)
        else:
            coverage_set &= candidates
        if not coverage_set:
            return 0
    return len(coverage_set or [])


def _evidence_from_ids(
    paper_ids: set[str],
    paper_lookup: dict[str, dict],
    limit: int,
) -> list[dict]:
    evidence = []
    for paper_id in sorted(paper_ids)[:limit]:
        paper = paper_lookup.get(paper_id, {})
        evidence.append(
            {
                "paper_id": paper_id,
                "title": paper.get("title", "Unknown"),
                "year": paper.get("publication_year"),
            }
        )
    return evidence


def _format_gap_items(items: list[dict]) -> list[str]:
    if not items:
        return ["- No gaps identified.", ""]
    lines = []
    for item in items:
        lines.append(f"- {item['label']} (count: {item['count']})")
        evidence = item.get("evidence", [])
        if evidence:
            lines.append("  Evidence:")
            for paper in evidence:
                title = paper.get("title", "Unknown")
                year = paper.get("year")
                year_text = f" ({year})" if year else ""
                lines.append(f"  - {title}{year_text}")
    lines.append("")
    return lines


def _format_year_gaps(summary: dict) -> list[str]:
    if not summary:
        return ["- No year data available.", ""]
    lines = []
    missing = summary.get("missing_ranges", [])
    sparse = summary.get("sparse_years", [])
    if missing:
        lines.append("- Missing year ranges:")
        for gap in missing:
            start = gap["start"]
            end = gap["end"]
            label = f"{start}-{end}" if start != end else f"{start}"
            lines.append(f"  - {label} ({gap['length']} years)")
    else:
        lines.append("- Missing year ranges: None")
    if sparse:
        lines.append("- Sparse years:")
        for year in sparse:
            lines.append(f"  - {year['year']}: {year['count']} paper(s)")
    else:
        lines.append("- Sparse years: None")
    lines.append("")
    return lines


def _format_future_gaps(items: list[dict]) -> list[str]:
    if not items:
        return ["- No gaps identified.", ""]
    lines = []
    for item in items:
        lines.append(
            f"- {item['direction']} (mentions: {item['mention_count']}, coverage: {item['coverage_count']})"
        )
        evidence = item.get("evidence", [])
        if evidence:
            lines.append("  Evidence:")
            for paper in evidence:
                title = paper.get("title", "Unknown")
                year = paper.get("year")
                year_text = f" ({year})" if year else ""
                lines.append(f"  - {title}{year_text}")
    lines.append("")
    return lines


def _build_notes(
    papers: list[dict],
    extraction_count: int,
    topic_gaps: list[dict],
    method_gaps: list[dict],
) -> list[str]:
    notes = []
    if not papers:
        notes.append("No papers available after applying filters.")
    if extraction_count == 0:
        notes.append("No extractions found; topic/methodology gaps may be incomplete.")
    if not topic_gaps:
        notes.append("No underrepresented topics detected under current thresholds.")
    if not method_gaps:
        notes.append("No underrepresented methodologies detected under current thresholds.")
    notes.append(
        "Gap signals are heuristic and should be validated against domain expertise."
    )
    return notes
