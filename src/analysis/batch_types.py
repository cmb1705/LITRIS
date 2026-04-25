"""Shared types for provider batch extraction clients."""

from dataclasses import dataclass, field

from src.zotero.models import PaperMetadata


@dataclass
class BatchRequest:
    """A single semantic extraction request in a provider batch."""

    custom_id: str
    paper: PaperMetadata
    prompt: str
    pass_number: int = 1


@dataclass
class PaperPassResults:
    """Accumulator for per-paper pass results during batch reassembly."""

    answers: dict[str, str | None] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: list[str] = field(default_factory=list)
