"""LLM analysis and extraction module."""

from src.analysis.gap_detection import (
    GapDetectionConfig,
    analyze_gap_report,
    format_gap_report_markdown,
    load_gap_report,
    save_gap_report,
)

try:
    from src.analysis.llm_client import LLMClient
except ModuleNotFoundError:  # Optional dependencies may be unavailable in tests
    LLMClient = None

from src.analysis.schemas import (
    EvidenceType,
    ExtractionResult,
    KeyClaim,
    KeyFinding,
    Methodology,
    PaperExtraction,
    SignificanceLevel,
    SupportType,
)

try:
    from src.analysis.section_extractor import ExtractionStats, SectionExtractor
except ImportError:  # Optional dependencies may be unavailable in tests
    ExtractionStats = None
    SectionExtractor = None

__all__ = [
    # Schemas
    "PaperExtraction",
    "ExtractionResult",
    "Methodology",
    "KeyFinding",
    "KeyClaim",
    "EvidenceType",
    "SignificanceLevel",
    "SupportType",
    # Gap detection
    "GapDetectionConfig",
    "analyze_gap_report",
    "load_gap_report",
    "format_gap_report_markdown",
    "save_gap_report",
]

if LLMClient is not None:
    __all__.insert(0, "LLMClient")

if SectionExtractor is not None and ExtractionStats is not None:
    __all__.extend(["SectionExtractor", "ExtractionStats"])
