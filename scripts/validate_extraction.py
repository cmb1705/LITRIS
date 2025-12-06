#!/usr/bin/env python
"""Validate extraction results and generate quality report."""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.schemas import PaperExtraction
from src.utils.file_utils import safe_read_json, safe_write_json
from src.utils.logging_config import setup_logging


@dataclass
class ValidationResult:
    """Result of validating a single extraction."""

    paper_id: str
    title: str
    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    field_coverage: dict[str, bool] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of validation results."""

    total_extractions: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    warning_count: int = 0
    avg_confidence: float = 0.0
    field_coverage_rates: dict[str, float] = field(default_factory=dict)
    common_issues: list[tuple[str, int]] = field(default_factory=list)
    low_confidence_papers: list[str] = field(default_factory=list)


class ExtractionValidator:
    """Validate extraction results for quality and completeness."""

    REQUIRED_FIELDS = ["research_questions", "methodology", "key_findings"]
    RECOMMENDED_FIELDS = ["key_claims", "conclusions", "limitations"]

    MIN_CONFIDENCE = 0.5
    LOW_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        min_confidence: float = 0.5,
        require_claims: bool = True,
    ):
        """Initialize validator.

        Args:
            min_confidence: Minimum confidence score for valid extraction.
            require_claims: Whether key_claims is required.
        """
        self.min_confidence = min_confidence
        self.require_claims = require_claims

    def validate_extraction(
        self,
        paper_id: str,
        title: str,
        extraction_data: dict,
    ) -> ValidationResult:
        """Validate a single extraction.

        Args:
            paper_id: Paper identifier.
            title: Paper title.
            extraction_data: Extraction data dictionary.

        Returns:
            ValidationResult with issues and warnings.
        """
        result = ValidationResult(
            paper_id=paper_id,
            title=title,
            valid=True,
            confidence=extraction_data.get("extraction_confidence", 0.0),
        )

        # Check confidence
        if result.confidence < self.min_confidence:
            result.issues.append(
                f"Confidence too low: {result.confidence:.2f} < {self.min_confidence}"
            )
            result.valid = False

        if result.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            result.warnings.append(f"Low confidence: {result.confidence:.2f}")

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            value = extraction_data.get(field)
            has_value = self._has_meaningful_value(value)
            result.field_coverage[field] = has_value

            if not has_value:
                result.issues.append(f"Missing required field: {field}")
                result.valid = False

        # Check key_claims if required
        if self.require_claims:
            claims = extraction_data.get("key_claims", [])
            result.field_coverage["key_claims"] = bool(claims)
            if not claims:
                result.issues.append("Missing key_claims")
                result.valid = False

        # Check recommended fields
        for field in self.RECOMMENDED_FIELDS:
            if field == "key_claims" and self.require_claims:
                continue
            value = extraction_data.get(field)
            has_value = self._has_meaningful_value(value)
            result.field_coverage[field] = has_value

            if not has_value:
                result.warnings.append(f"Missing recommended field: {field}")

        # Check methodology structure
        methodology = extraction_data.get("methodology", {})
        if isinstance(methodology, dict):
            if not methodology.get("approach"):
                result.warnings.append("Methodology missing approach")
            if not methodology.get("sample_size"):
                result.warnings.append("Methodology missing sample_size info")

        # Check key_findings quality
        findings = extraction_data.get("key_findings", [])
        if findings and len(findings) < 2:
            result.warnings.append("Only one key finding extracted")

        return result

    def _has_meaningful_value(self, value) -> bool:
        """Check if a value is meaningful (non-empty, non-null)."""
        if value is None:
            return False
        if isinstance(value, str):
            return len(value.strip()) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True

    def validate_all(
        self,
        extractions: dict[str, dict],
        papers: dict[str, dict] | None = None,
    ) -> tuple[list[ValidationResult], ValidationSummary]:
        """Validate all extractions.

        Args:
            extractions: Dictionary of paper_id -> extraction data.
            papers: Optional dictionary of paper_id -> paper data.

        Returns:
            Tuple of (list of results, summary).
        """
        results = []
        issue_counter = Counter()

        for paper_id, ext_data in extractions.items():
            # Get extraction (handle nested structure)
            extraction = ext_data.get("extraction", ext_data)

            # Get title
            title = "Unknown"
            if papers and paper_id in papers:
                title = papers[paper_id].get("title", "Unknown")

            result = self.validate_extraction(paper_id, title, extraction)
            results.append(result)

            # Count issues
            for issue in result.issues:
                issue_counter[issue] += 1

        # Build summary
        summary = ValidationSummary(
            total_extractions=len(results),
            valid_count=sum(1 for r in results if r.valid),
            invalid_count=sum(1 for r in results if not r.valid),
            warning_count=sum(1 for r in results if r.warnings),
        )

        # Average confidence
        confidences = [r.confidence for r in results if r.confidence > 0]
        summary.avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Field coverage rates
        all_fields = set()
        for r in results:
            all_fields.update(r.field_coverage.keys())

        for field in all_fields:
            covered = sum(
                1 for r in results if r.field_coverage.get(field, False)
            )
            summary.field_coverage_rates[field] = covered / len(results) if results else 0

        # Common issues
        summary.common_issues = issue_counter.most_common(10)

        # Low confidence papers
        summary.low_confidence_papers = [
            r.paper_id
            for r in results
            if r.confidence < self.LOW_CONFIDENCE_THRESHOLD
        ][:20]

        return results, summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate extraction results and generate quality report"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=project_root / "data" / "index",
        help="Path to index directory",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for validation report (JSON)",
    )
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="Show details of invalid extractions",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show all warnings",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Load extractions
    extractions_file = args.index_dir / "extractions.json"
    if not extractions_file.exists():
        logger.error(f"Extractions file not found: {extractions_file}")
        return 1

    extractions_data = safe_read_json(extractions_file, default={})
    if isinstance(extractions_data, dict) and "extractions" in extractions_data:
        extractions = extractions_data["extractions"]
    else:
        extractions = extractions_data

    if not extractions:
        logger.error("No extractions found to validate")
        return 1

    logger.info(f"Loaded {len(extractions)} extractions to validate")

    # Load papers for titles
    papers_file = args.index_dir / "papers.json"
    papers = {}
    if papers_file.exists():
        papers_data = safe_read_json(papers_file, default={})
        if isinstance(papers_data, dict) and "papers" in papers_data:
            papers = {p["paper_id"]: p for p in papers_data["papers"] if "paper_id" in p}

    # Run validation
    validator = ExtractionValidator(min_confidence=args.min_confidence)
    results, summary = validator.validate_all(extractions, papers)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal extractions: {summary.total_extractions}")
    print(f"Valid: {summary.valid_count} ({summary.valid_count/summary.total_extractions*100:.1f}%)")
    print(f"Invalid: {summary.invalid_count} ({summary.invalid_count/summary.total_extractions*100:.1f}%)")
    print(f"With warnings: {summary.warning_count}")
    print(f"Average confidence: {summary.avg_confidence:.3f}")

    print("\nField Coverage Rates:")
    for field, rate in sorted(summary.field_coverage_rates.items()):
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {field:20s} [{bar}] {rate*100:5.1f}%")

    if summary.common_issues:
        print("\nMost Common Issues:")
        for issue, count in summary.common_issues:
            print(f"  {count:3d}x {issue}")

    if summary.low_confidence_papers:
        print(f"\nLow Confidence Papers ({len(summary.low_confidence_papers)}):")
        for paper_id in summary.low_confidence_papers[:5]:
            result = next(r for r in results if r.paper_id == paper_id)
            print(f"  - {result.title[:50]}... (conf: {result.confidence:.2f})")
        if len(summary.low_confidence_papers) > 5:
            print(f"  ... and {len(summary.low_confidence_papers) - 5} more")

    # Show invalid details
    if args.show_invalid:
        invalid_results = [r for r in results if not r.valid]
        if invalid_results:
            print("\n" + "-" * 60)
            print("INVALID EXTRACTIONS")
            print("-" * 60)
            for result in invalid_results:
                print(f"\n{result.title[:60]}...")
                print(f"  Paper ID: {result.paper_id}")
                print(f"  Confidence: {result.confidence:.2f}")
                print("  Issues:")
                for issue in result.issues:
                    print(f"    - {issue}")

    # Show warnings
    if args.show_warnings:
        results_with_warnings = [r for r in results if r.warnings]
        if results_with_warnings:
            print("\n" + "-" * 60)
            print("WARNINGS")
            print("-" * 60)
            for result in results_with_warnings[:20]:
                print(f"\n{result.title[:60]}...")
                for warning in result.warnings:
                    print(f"  ⚠ {warning}")

    # Save report
    if args.output:
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_extractions": summary.total_extractions,
                "valid_count": summary.valid_count,
                "invalid_count": summary.invalid_count,
                "warning_count": summary.warning_count,
                "avg_confidence": summary.avg_confidence,
                "field_coverage_rates": summary.field_coverage_rates,
                "common_issues": [{"issue": i, "count": c} for i, c in summary.common_issues],
                "low_confidence_papers": summary.low_confidence_papers,
            },
            "results": [
                {
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "valid": r.valid,
                    "confidence": r.confidence,
                    "issues": r.issues,
                    "warnings": r.warnings,
                    "field_coverage": r.field_coverage,
                }
                for r in results
            ],
        }
        safe_write_json(args.output, report)
        print(f"\nReport saved to: {args.output}")

    print("\n" + "=" * 60)

    # Return exit code based on validity
    return 0 if summary.invalid_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
