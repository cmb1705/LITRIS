"""LLM analysis and extraction module."""

from src.analysis.llm_client import LLMClient
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
from src.analysis.section_extractor import ExtractionStats, SectionExtractor

__all__ = [
    # Client
    "LLMClient",
    # Schemas
    "PaperExtraction",
    "ExtractionResult",
    "Methodology",
    "KeyFinding",
    "KeyClaim",
    "EvidenceType",
    "SignificanceLevel",
    "SupportType",
    # Extractor
    "SectionExtractor",
    "ExtractionStats",
]
