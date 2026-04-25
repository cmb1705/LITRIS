"""LLM analysis and extraction module.

Package-level exports are resolved lazily so importing a lightweight
submodule such as ``src.analysis.constants`` does not initialize optional
LLM or extraction backends.
"""

__all__ = [
    # Schemas (SemanticAnalysis is the primary analysis type)
    "SemanticAnalysis",
    "DimensionedExtraction",
    "PaperExtraction",  # retained for backward compatibility
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
    # Research question generation
    "ResearchQuestionConfig",
    "QuestionStyle",
    "QuestionScope",
    "build_prompts_from_gap_report",
    "LLMClient",
    "SectionExtractor",
    "ExtractionStats",
]

_EXPORTS = {
    "DimensionedExtraction": "src.analysis.schemas",
    "EvidenceType": "src.analysis.schemas",
    "ExtractionResult": "src.analysis.schemas",
    "KeyClaim": "src.analysis.schemas",
    "KeyFinding": "src.analysis.schemas",
    "Methodology": "src.analysis.schemas",
    "PaperExtraction": "src.analysis.schemas",
    "SemanticAnalysis": "src.analysis.schemas",
    "SignificanceLevel": "src.analysis.schemas",
    "SupportType": "src.analysis.schemas",
    "GapDetectionConfig": "src.analysis.gap_detection",
    "analyze_gap_report": "src.analysis.gap_detection",
    "format_gap_report_markdown": "src.analysis.gap_detection",
    "load_gap_report": "src.analysis.gap_detection",
    "save_gap_report": "src.analysis.gap_detection",
    "QuestionScope": "src.analysis.research_questions",
    "QuestionStyle": "src.analysis.research_questions",
    "ResearchQuestionConfig": "src.analysis.research_questions",
    "build_prompts_from_gap_report": "src.analysis.research_questions",
    "LLMClient": "src.analysis.llm_client",
    "ExtractionStats": "src.analysis.section_extractor",
    "SectionExtractor": "src.analysis.section_extractor",
}


def __getattr__(name: str) -> object:
    """Resolve legacy package exports on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from importlib import import_module

        value = getattr(import_module(module_name), name)
    except (ImportError, ModuleNotFoundError):
        if name in {"LLMClient", "ExtractionStats", "SectionExtractor"}:
            value = None
        else:
            raise
    globals()[name] = value
    return value
