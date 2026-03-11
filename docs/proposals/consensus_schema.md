# Consensus Schema and Aggregation Strategy

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-03-10 |
| Task | li-7v6 |
| Implementation | src/analysis/llm_council.py |

---

## 1. Overview

The LLM Council enables multi-provider consensus extraction by querying
multiple LLM providers in parallel and aggregating their responses into a
single consensus extraction. Inspired by [karpathy/llm-council](https://github.com/karpathy/llm-council),
simplified for structured academic paper extraction.

### Design Principles

1. **Resilience**: Provider failures do not block extraction.
2. **Quality**: Multiple perspectives improve extraction accuracy.
3. **Cost awareness**: Per-provider cost limits prevent budget overruns.
4. **Transparency**: Every provider response is recorded for audit.

---

## 2. Consensus Strategies

Each extraction field uses a strategy appropriate to its data type:

| Strategy | Description | Applied To |
|----------|-------------|------------|
| LONGEST | Take the longest/most detailed response | thesis_statement, theoretical_framework, conclusions, contribution_summary, extraction_notes |
| UNION | Combine all unique values | research_questions, limitations, future_directions, keywords, discipline_tags, key_findings, key_claims |
| MAJORITY_VOTE | Most common value wins | methodology (approach, design) |
| AVERAGE | Average numeric values | extraction_confidence |
| INTERSECTION | Keep only values all providers agree on | Reserved for high-confidence fields |
| FIRST_VALID | Take first non-empty response | Fallback for sparse fields |
| WEIGHTED | Weight by provider reliability | Available via provider weights |

### Strategy Rationale

- **LONGEST for prose fields**: Longer thesis statements and conclusions
  tend to be more complete and nuanced. Truncated responses lose context.
- **UNION for list fields**: Different providers surface different
  keywords, findings, and directions. Union maximizes recall.
- **MAJORITY_VOTE for methodology**: Approach and design are categorical;
  the most common answer is likely correct.
- **AVERAGE for confidence**: Smooths out individual provider biases.

---

## 3. Provider Configuration Schema

```python
@dataclass
class ProviderConfig:
    name: str                       # Provider identifier (anthropic, openai, google, ollama)
    weight: float = 1.0             # Reliability weight for weighted consensus
    timeout: int = 120              # Per-provider timeout in seconds
    max_cost: float | None = None   # Max cost per extraction (USD)
    enabled: bool = True            # Toggle without removing from config
```

### Provider Weights

Weights affect:
- Weighted average for extraction_confidence
- Can be extended to weighted voting for categorical fields

Recommended starting weights based on extraction quality benchmarks:

| Provider | Suggested Weight | Rationale |
|----------|-----------------|-----------|
| Anthropic (Claude) | 1.2 | Strong structured output compliance |
| OpenAI (GPT) | 1.0 | Baseline |
| Google (Gemini) | 0.8 | Good but less consistent on enums |
| Ollama (local) | 0.7 | Depends on model size |

---

## 4. Council Configuration Schema

```python
@dataclass
class CouncilConfig:
    providers: list[ProviderConfig]     # Provider list
    min_responses: int = 2              # Minimum responses for consensus
    fallback_to_single: bool = True     # Use single response if min not met
    parallel: bool = True               # Parallel provider execution
    timeout: int = 180                  # Overall council timeout (seconds)
    consensus_threshold: float = 0.5    # Threshold for majority decisions
```

### Fallback Behavior

```
Responses >= min_responses  --> Build consensus (full aggregation)
Responses < min_responses   --> fallback_to_single=True  --> Use best single response (confidence=0.5)
                            --> fallback_to_single=False --> Return failure
```

---

## 5. Response and Result Schemas

### Per-Provider Response

```python
@dataclass
class ProviderResponse:
    provider: str                              # Provider name
    extraction: PaperExtraction | None         # Extraction result
    success: bool                              # Whether extraction succeeded
    error: str | None = None                   # Error message if failed
    duration_seconds: float = 0.0              # Wall-clock time
    cost: float = 0.0                          # Estimated cost (USD)
```

### Council Result

```python
@dataclass
class CouncilResult:
    paper_id: str                              # Paper identifier
    consensus: PaperExtraction | None          # Aggregated extraction
    provider_responses: list[ProviderResponse] # All individual responses
    success: bool                              # Whether consensus was built
    consensus_confidence: float = 0.0          # 0-1 confidence score
    total_duration_seconds: float = 0.0        # Wall-clock time
    total_cost: float = 0.0                    # Sum of provider costs
    errors: list[str]                          # Error messages
```

---

## 6. Consensus Confidence Calculation

Confidence combines response rate and inter-provider agreement:

```
confidence = 0.6 * response_rate + 0.4 * avg_agreement
```

Where:
- `response_rate` = successful responses / total providers
- `avg_agreement` = average of field-specific agreement scores

### Agreement Metrics

| Field | Metric | Description |
|-------|--------|-------------|
| thesis_statement | First-5-word overlap | Inverse of unique first-word tuples ratio |
| keywords | Jaccard similarity | Intersection/union of keyword sets |

---

## 7. Aggregation for Nested Objects

### Methodology Merge

```
approach     --> majority_vote(all approaches)
design       --> majority_vote(all designs)
data_sources --> union(all data_sources lists)
analysis_methods --> union(all analysis_methods lists)
sample_size  --> longest(all sample_sizes)
time_period  --> longest(all time_periods)
```

### Key Findings Merge

Union of all findings, deduplicated by normalized finding text
(lowercase, stripped). First occurrence preserved.

### Key Claims Merge

Union of all claims, deduplicated by normalized claim text.
First occurrence preserved.

### Discipline Tags

Union of all tags, normalized to lowercase, deduplicated.

---

## 8. Example Usage

```python
from src.analysis.llm_council import LLMCouncil, CouncilConfig, ProviderConfig

config = CouncilConfig(
    providers=[
        ProviderConfig(name="anthropic", weight=1.2),
        ProviderConfig(name="openai", weight=1.0),
        ProviderConfig(name="google", weight=0.8),
    ],
    min_responses=2,
    parallel=True,
    timeout=180,
)

council = LLMCouncil(config)
result = council.extract(
    paper_id="paper_001",
    title="Example Paper Title",
    authors="Author et al.",
    year=2024,
    item_type="journalArticle",
    text="Full paper text...",
)

if result.success:
    print(f"Confidence: {result.consensus_confidence:.2f}")
    print(f"Providers responded: {len([r for r in result.provider_responses if r.success])}")
    print(f"Total cost: ${result.total_cost:.4f}")
```

---

## 9. Limitations and Future Work

1. **String deduplication**: Current dedup uses exact normalized matching;
   semantic similarity would catch paraphrases.
2. **Confidence granularity**: Per-field confidence scores would help
   identify which parts of the consensus are strong vs weak.
3. **Cost tracking**: Actual cost estimation requires API response
   metadata (token counts, pricing tiers).
4. **Streaming**: Current implementation waits for all providers;
   streaming could surface partial results faster.
