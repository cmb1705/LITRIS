<!-- markdownlint-disable MD013 MD030 MD060 -->
# LITRIS Redesign: Technical Progress Report

**Date:** March 2026 \| **Version:** 0.2.0

This report is a historical redesign summary from March 2026. It describes the
state of the migration work at that time, not every current runtime default.
LITRIS now uses profile-driven semantic dimensions, canonical full-text
snapshots for verbatim retrieval, and updated extraction-cascade ordering beyond
what this report originally captured.

***

## Overview

LITRIS (Literature Review Indexing System) is an AI-assisted literature review tool that extracts structured insights from academic papers and indexes them for semantic search. The system supports corpora of any size across arbitrary academic disciplines. This report summarizes the infrastructure redesign completed in March 2026, which replaced the original single-pass extraction pipeline with a multi-dimensional analysis framework and introduced multi-provider consensus capabilities.

## 40-Dimension Semantic Analysis

The extraction schema was redesigned from a flat structure (thesis, methodology, key findings) to a 40-dimension SemanticAnalysis framework organized across six analytical passes:

| Pass                            | Focus                                                                  |
|---------------------------------|------------------------------------------------------------------------|
| Research Core (q01-q05)         | Research question, thesis, claims, evidence, limitations               |
| Methodology (q06-q10)           | Paradigm, methods, data, reproducibility, framework                    |
| Context & Discourse (q11-q16)   | Traditions, citations, assumptions, counterarguments, novelty, stance  |
| Meta & Audience (q17-q24)       | Field, audience, implications, future work, quality, contribution      |
| Scholarly Positioning (q25-q31) | Institutional context, timing, paradigm influence, interdisciplinarity |
| Impact & Gaps (q32-q40)         | Deployment gap, infrastructure, power dynamics, omissions, emergence   |

At the time of this report, each paper was analyzed through six sequential LLM
calls, one per pass, using the paper's full text. The current system preserves
legacy `q01` through `q40` compatibility, but the active runtime now stores
answers in profile-driven canonical `dimensions` maps keyed by stable dimension
IDs.

## Multi-Provider Consensus (LLM Council)

A multi-provider extraction architecture was implemented to address quality concerns with single-provider extraction. In a 5-paper benchmark using the prior system, the longest-string aggregation strategy selected the most verbose provider 93% of the time regardless of content quality.

The redesigned council introduces three aggregation strategies:

-   **Quality-weighted** (default): Scores responses by sentence structure, citation patterns, and named entities, multiplied by provider reliability weight. In a 4-paper live comparison, this strategy achieved 100% average dimension coverage versus 95% with the prior longest-string approach.
-   **Union merge**: Deduplicates and merges unique sentences across providers for list-like dimensions (e.g., key claims, limitations).
-   **Longest**: Preserved for backward compatibility.

A comparative test across 8 papers using Claude Opus 4.6 and GPT-5.4 found that OpenAI produced higher individual coverage (98.8% vs 91.9%) while Anthropic produced more structured, citation-rich text. The council's consensus extractions blend both strengths, filling dimensions that neither provider catches alone. For example, one paper's council consensus filled three dimensions (power dynamics, dual-use concerns, policy recommendations) that were absent from both individual extractions.

## Gap-Filling Extraction

Rather than running the full council on every paper (which doubles extraction cost), a targeted gap-filling mechanism was implemented. After primary extraction, papers below a configurable coverage threshold are re-extracted only on the specific passes containing missing dimensions, using a secondary provider. This achieves approximately 80% of the council's quality benefit at 10-30% of the cost.

## Extraction Pipeline Improvements

The PDF text extraction cascade was originally reordered to prioritize speed:

1.  Companion markdown sidecars when present
2.  arXiv HTML
3.  ar5iv
4.  OpenDataLoader
5.  PyMuPDF
6.  Marker

OCR is now handled as a fallback within the PDF extraction path rather than as a
simple standalone top-level tier.

An OCR assessment comparing Tesseract and GLM-OCR (a 0.9B-parameter multimodal model) on 8 papers with failed text extraction found that both methods recovered 800-3,400 words from pages where PyMuPDF extracted zero. Enabling OCR in the cascade recovers approximately 101 papers (6.5% of the corpus) that were previously excluded from the index due to insufficient text.

## Build Infrastructure

The index build pipeline now supports parallel extraction with adaptive rate
limiting, resumable backfill checkpoints, canonical full-text snapshot reuse,
and portable dimension-profile backfills. Multiple LLM workers process papers
concurrently, reducing parallelism when rate-limit signals are detected and
resuming from checkpoints after interruptions.

A per-dimension reasoning effort classification maps each of the 40 dimensions as either structural (answer is explicit in the text) or inferential (requires synthesis or critical reading), establishing the foundation for differentiated compute allocation in future builds.

## Current Status

The full corpus re-extraction is underway using GPT-5.4 with highest reasoning effort, gap-filling enabled at 90% coverage threshold, and 4 parallel workers. The system checkpoints every 10 papers and auto-recovers from interruptions. Early results show 91-100% dimension coverage across processed papers.

| Metric                  | Value                                               |
|-------------------------|-----------------------------------------------------|
| Corpus size             | User-defined (scales to thousands of papers)         |
| Extraction dimensions   | 40 (6-pass pipeline)                                |
| Test coverage           | 980 tests, CI green across 3 OS x 3 Python versions |
| Aggregation strategies  | 3 (quality-weighted, union, longest)                |
| LLM providers supported | 5 (Anthropic, OpenAI, Google, Ollama, llama.cpp)    |
