# Embedding Model Evaluation for LITRIS

**Issue**: LITRIS-79l
**Date**: 2026-03-06, updated 2026-03-06
**Status**: Research complete (revised)

## Context

Current model: `sentence-transformers/all-MiniLM-L6-v2` (384d, MTEB 56.09, 256-token context).
Goal: Evaluate the full landscape of embedding models -- free and paid -- and recommend
the best option for a library of ~500 academic publications.

**Cost basis**: 500 papers x 9 chunks x 500 tokens = ~2.25M tokens per full index build.

## Existing Subscription Analysis

| Provider | Embeddings? | Details |
| -------- | ----------- | ------- |
| Anthropic (Claude) | No | No embeddings API. Recommends Voyage AI (paid). |
| OpenAI | Yes, but paid | text-embedding-3-large: $0.13/1M tokens. NOT free with ChatGPT sub. |
| Google (Gemini) | Yes, free tier | gemini-embedding-001: free tier available but 2048-token input limit. Note: `text-embedding-004` deprecated Jan 2026. Rate limits: ~5-15 RPM on free tier. |

## Full Model Comparison (sorted by MTEB average)

### Tier 1: Heavyweight Leaders (MTEB 68+)

Models that top the overall leaderboard but require significant RAM (8B+ parameters).

| Model | Dims | MTEB Avg | Context | Params | Cost | Local? | Ollama? | License | 500-paper cost |
| ----- | ---- | -------- | ------- | ------ | ---- | ------ | ------- | ------- | -------------- |
| NV-Embed-v2 | 4096 | 72.31 | 32K | 7.85B | Free | Yes | No | CC BY-NC 4.0 | $0 (non-commercial) |
| Qwen3-Embedding-8B | 4096 | 70.58 | 32K | 8B | Free | Yes | Yes | Apache 2.0 | $0 |
| gemini-embedding-001 | 3072 | ~68 | 2K | API | Free tier | No | No | Proprietary | $0 (free tier) |
| voyage-4-large (MoE) | 2048 | ~68.6 | 32K | MoE | $0.06/1M | No | No | Proprietary | ~$0.14 |

### Tier 2: Strong Contenders (MTEB 64-67)

The sweet spot for quality vs. resource tradeoff.

| Model | Dims | MTEB Avg | Context | Params | Cost | Local? | Ollama? | License | 500-paper cost |
| ----- | ---- | -------- | ------- | ------ | ---- | ------ | ------- | ------- | -------------- |
| voyage-3.5 | 2048 | ~66 | 32K | API | $0.06/1M | No | No | Proprietary | ~$0.14 |
| jina-embeddings-v3 | 1024 | 65.52 | 8K | 570M | Free* | Yes | No | CC BY-NC 4.0 | $0 (non-commercial) |
| **gte-large-en-v1.5** | **1024** | **65.39** | **8K** | **434M** | **Free** | **Yes** | **No** | **Apache 2.0** | **$0** |
| Cohere embed-v4 | 1536 | 65.2 | 512 | API | $0.12/1M | No | No | Proprietary | ~$0.27 |
| mxbai-embed-large-v1 | 1024 | 64.68 | 512 | 334M | Free | Yes | Yes | Apache 2.0 | $0 |
| Qwen3-Embedding-0.6B | 1024 | 64.33 | 32K | 0.6B | Free | Yes | Yes | Apache 2.0 | $0 |
| OpenAI emb-3-large | 3072 | 64.6 | 8K | API | $0.13/1M | No | No | Proprietary | ~$0.29 |
| bge-large-en-v1.5 | 1024 | 64.23 | 512 | 335M | Free | Yes | No | MIT | $0 |
| voyage-3-large | 2048 | ~64 | 32K | API | $0.18/1M | No | No | Proprietary | ~$0.41 |

### Tier 3: Lightweight / Budget (MTEB 56-63)

Smaller models or older-generation options.

| Model | Dims | MTEB Avg | Context | Params | Cost | Local? | Ollama? | License | 500-paper cost |
| ----- | ---- | -------- | ------- | ------ | ---- | ------ | ------- | ------- | -------------- |
| EmbeddingGemma-300M | 768 | ~63 | 2K | 300M | Free | Yes | Yes | Gemma (commercial OK) | $0 |
| BGE-M3 | 1024 | 63.0 | 8K | 568M | Free | Yes | No | MIT | $0 |
| nomic-embed-text-v1.5 | 768 | 62.28 | 8K | 137M | Free | Yes | Yes | Apache 2.0 | $0 |
| OpenAI emb-3-small | 1536 | 62.3 | 8K | API | $0.02/1M | No | No | Proprietary | ~$0.05 |
| e5-large-v2 | 1024 | ~61.5 | 512 | 335M | Free | Yes | No | MIT | $0 |
| **all-MiniLM-L6-v2 (current)** | **384** | **56.09** | **256** | **22M** | **Free** | **Yes** | **Yes** | **Apache 2.0** | **$0** |

*jina-embeddings-v3 and NV-Embed-v2: non-commercial licenses (CC BY-NC 4.0).

## Cost Estimates for 500 Publications

Based on 500 papers x 9 chunks x 500 tokens = 2.25M tokens per full index build:

| Model | Per-build cost | 10 rebuilds | Annual (monthly rebuild) |
| ----- | -------------- | ----------- | ------------------------ |
| All local models | $0.00 | $0.00 | $0.00 |
| OpenAI emb-3-small | $0.05 | $0.45 | $0.54 |
| voyage-3.5 | $0.14 | $1.35 | $1.62 |
| voyage-4-large | $0.14 | $1.35 | $1.62 |
| Cohere embed-v4 | $0.27 | $2.70 | $3.24 |
| OpenAI emb-3-large | $0.29 | $2.93 | $3.51 |
| voyage-3-large | $0.41 | $4.05 | $4.86 |

All paid API models cost under $5/year even with monthly rebuilds. Cost is not a differentiator.

## Ollama Embedding Models (Zero-Cost Local)

All run locally via `ollama serve`. ChromaDB integrates natively via `OllamaEmbeddingFunction`.

| Ollama Model | Base Model | Dims | Context | Size |
| ------------ | ---------- | ---- | ------- | ---- |
| `qwen3-embedding` (8b) | Qwen3-Embedding-8B | 4096 | 32K | ~16GB |
| `qwen3-embedding` (0.6b) | Qwen3-Embedding-0.6B | 1024 | 32K | ~1.2GB |
| `embeddinggemma` | EmbeddingGemma-300M | 768 | 2K | ~600MB |
| `nomic-embed-text` | nomic-embed-text-v1.5 | 768 | 8K | ~274MB |
| `mxbai-embed-large` | mxbai-embed-large-v1 | 1024 | 512 | ~670MB |
| `snowflake-arctic-embed2` | arctic-embed-l-v2.0 | 1024 | 8K | ~670MB |
| `all-minilm` | all-MiniLM-L6-v2 | 384 | 256 | ~46MB |

## Query Prefix Requirements

Some models require prefixes for optimal retrieval:

- **Qwen3-Embedding**: Instruction-based (task description in query)
- **e5-large-v2**: `"query:"` for queries, `"passage:"` for documents (trained on S2ORC academic corpus)
- **nomic-embed-text-v1.5**: `"search_query:"` / `"search_document:"`
- **bge-large-en-v1.5**: `"Represent this sentence:"` for queries
- **gte-large-en-v1.5**: No prefix needed
- **mxbai-embed-large-v1**: No prefix needed

## Target Hardware

### Workstation (embedding host)

| Spec | Value | Impact |
| ---- | ----- | ------ |
| CPU | 16-core workstation-class CPU with AVX2 | Fast CPU inference path, 16 threads for parallel encoding |
| RAM | 64 GB system RAM | Any model fits comfortably, including Qwen3-8B (16 GB) |
| GPU | CUDA-capable GPU with 16 GB VRAM | CUDA acceleration: 10-50x faster than CPU for embedding |
| Role | Runs embedding during index builds | Full 500-paper re-index: ~1-2 min (GPU), ~3-5 min (CPU) |

All recommended models fit in the 16 GB VRAM. sentence-transformers auto-detects CUDA.

### Low-Power Self-Hosted Node

| Spec | Value | Impact |
| ---- | ----- | ------ |
| CPU | Low-power 4-core CPU (2.0-2.7 GHz class, no AVX/AVX2) | Low single-thread perf, no AVX/AVX2 |
| RAM | 10 GB system RAM | Limits model size to ~4-6 GB working set |
| GPU/NPU | None | No hardware acceleration for inference |
| AVX support | **None** | Ollama falls back to slowest CPU path (2-3x penalty) |
| Storage | Large local disk volume | Ample for models and indexes |
| Docker | Container runtime | Supported but AVX-dependent containers may fail |

### Hardware Constraints

1. **No AVX = slow Ollama inference.** Ollama's CPU library hierarchy is: `cpu_avx2` (fastest)
   -> `cpu_avx` -> `cpu` (slowest). A low-power host without AVX gets the slowest path, meaning 2-3x slower
   embedding generation than a modern desktop CPU.

2. **10 GB RAM ceiling.** After OS overhead (~1-2 GB), Docker overhead, ChromaDB, and the
   Discord bot, roughly 4-6 GB is available for embedding model inference. This rules out
   any model requiring >4 GB of RAM (Qwen3-8B at ~16 GB, NV-Embed-v2 at ~30 GB).

3. **No GPU/NPU.** All inference is CPU-bound on a 10W TDP mobile chip. Embedding 4,500 chunks
   through a 0.6B model will take 30-60+ minutes, not 2-5 minutes.

4. **Embedding is a batch operation, not real-time.** Index builds happen infrequently (when new
   papers are added). Slow embedding is tolerable if search is fast -- and ChromaDB vector
   search is instant regardless of how long embedding took.

### What This Rules Out

- **Qwen3-Embedding-8B** -- needs ~16 GB RAM, won't fit
- **NV-Embed-v2** -- 7.85B params, ~30 GB, completely infeasible
- **gte-large-en-v1.5 via sentence-transformers** -- 434M params loads ~1.6 GB for model weights,
  but sentence-transformers adds PyTorch overhead (~2-3 GB). Tight but possible.
- **Any Ollama LLM for Open WebUI conversation** -- a 7B model needs ~8 GB and will be
  intolerably slow on Celeron without AVX. Open WebUI should use a cloud API instead.

## Recommendations

### Strategy: Embed locally on workstation, search on self-hosted node

The best approach given the low-power host constraints:

1. **Build the index on your workstation** (16 CPU cores, 64 GB RAM, 16 GB VRAM GPU) -- fast GPU embedding with any model
2. **Copy the ChromaDB directory to the target host** -- vector search is lightweight
3. **Run ChromaDB + Discord bot + MCP server on the self-hosted node** -- these are all low-resource
4. **Open WebUI on the self-hosted node uses cloud API** (Anthropic/OpenAI) for conversation, not local Ollama

This decouples the expensive one-time embedding step from the always-on search service.

### Primary: Alibaba-NLP/gte-large-en-v1.5

Best choice when embedding runs on the workstation:

- MTEB 65.39 -- highest free, permissively licensed model in sentence-transformers
- 8,192-token context -- handles long academic text
- 1024 dimensions
- Apache 2.0 license
- sentence-transformers drop-in replacement (embeds 4,500 chunks in ~1-2 min on a 16 GB VRAM GPU)
- ~1.6 GB on disk
- Requires `trust_remote_code=True`

### Alternative: Qwen3-Embedding-0.6B (if embedding must run on the self-hosted node)

If the NAS must handle embedding autonomously (e.g., auto-ingesting new papers):

- MTEB 64.33 -- 1 point below gte-large, still 8 points above current
- **32K-token context** -- most generous of any free model
- 1024 dimensions, Apache 2.0, available in Ollama
- ~1.2 GB via Ollama -- fits in a 10 GB RAM host
- **Caveat**: Embedding 4,500 chunks will take 30-60+ min on a low-power CPU without AVX.
  Acceptable for overnight batch jobs, not for interactive use.

### Best Paid API: voyage-3.5

If you want to skip local embedding entirely:

- Outperforms all local models on retrieval quality
- 32K context, 2048 dimensions
- $0.06/1M tokens ($0.14 per full index build of 500 papers)
- First 200M tokens free per account (covers ~88 full index builds)
- The self-hosted node just runs ChromaDB for search; embedding happens via API call from either machine
- **Downside**: Requires internet for index builds, not reproducible offline

### NOT Recommended for low-power self-hosted node

| Model | Why Not |
| ----- | ------- |
| Qwen3-Embedding-8B | Needs 16 GB RAM; the reference host has 10 GB total |
| NV-Embed-v2 | 7.85B params, ~30 GB RAM, non-commercial license |
| Any Ollama LLM (7B+) | Too slow on a low-power non-AVX CPU for conversational use |

### Decision Matrix (low-power host adjusted)

| Scenario | Recommended Model | Why |
| -------- | ----------------- | --- |
| Embed on workstation, search on self-hosted node | **gte-large-en-v1.5** | Best quality, fast on workstation, ChromaDB is portable |
| Self-hosted node must embed autonomously | Qwen3-Embedding-0.6B | Fits in RAM, Ollama, slow but works as batch job |
| Skip local embedding entirely | voyage-3.5 | Best quality, $0.14/build, 200M free tokens |
| Cheapest API fallback | OpenAI emb-3-small | $0.05/build, adequate quality |
| Minimal resources on self-hosted node | nomic-embed-text-v1.5 | 137M params, smallest viable upgrade |

## Re-Embedding Frequency

Embedding is **not** a continuous operation. It runs only when the index needs updating.

| Trigger | Scope | Frequency | Time (reference GPU) |
| ------- | ----- | --------- | ------------------- |
| New papers added to Zotero | Incremental (new papers only) | As-needed (weekly/monthly) | Seconds (5-20 papers) |
| Embedding model switch | Full re-index (all papers) | One-time or rare | ~1-2 min (500 papers) |
| Chunking strategy change | Full re-index | Rare (1-2x/year max) | ~1-2 min |
| Extraction schema change affecting chunks | Full re-index | Rare | ~1-2 min |

`build_index.py` uses content-hash change detection -- unchanged papers are skipped automatically.
For routine use (adding a few papers), the workstation completes embedding in seconds.

## Migration Plan

Switching models requires a full re-index:

1. Install Ollama: <https://ollama.com/download> (if using Ollama backend)
2. Pull the model: `ollama pull qwen3-embedding:8b`
3. Update `config.yaml`: model, dimension, backend, query_prefix
4. Delete existing ChromaDB collection (or use `--rebuild-embeddings`)
5. Run `python scripts/build_index.py --rebuild-embeddings` (on workstation)
6. Copy `data/chroma/` directory to the target host
7. Re-embedding ~4,500 chunks: ~1-2 min on workstation (GPU), ~45 min on a low-power CPU-only host

Integration point: `src/indexing/embeddings.py` (`EmbeddingGenerator` class).

```yaml
# config.yaml -- Qwen3-8B via Ollama (implemented):
embeddings:
  model: "qwen3-embedding:8b"
  dimension: 4096
  backend: "ollama"
  ollama_base_url: "http://localhost:11434"
  query_prefix: "Instruct: Retrieve academic papers relevant to the query\nQuery: "
  document_prefix: null

# config.yaml -- Qwen3-0.6B via Ollama:
embeddings:
  model: "qwen3-embedding:0.6b"
  dimension: 1024
  backend: "ollama"
  ollama_base_url: "http://localhost:11434"
  query_prefix: "Instruct: Retrieve academic papers relevant to the query\nQuery: "

# config.yaml -- gte-large via sentence-transformers:
embeddings:
  model: "Alibaba-NLP/gte-large-en-v1.5"
  dimension: 1024
  backend: "sentence-transformers"
```

Note: gte-large requires `trust_remote_code=True` in SentenceTransformer constructor.

## Sources

- [MTEB Leaderboard (Hugging Face)](https://huggingface.co/spaces/mteb/leaderboard)
- [Qwen3 Embedding Blog](https://qwenlm.github.io/blog/qwen3-embedding/)
- [Qwen3-Embedding-0.6B (Hugging Face)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3-Embedding on Ollama](https://ollama.com/library/qwen3-embedding)
- [NV-Embed-v2 (Hugging Face)](https://huggingface.co/nvidia/NV-Embed-v2)
- [NVIDIA Embedding Blog](https://developer.nvidia.com/blog/nvidia-text-embedding-model-tops-mteb-leaderboard/)
- [Voyage AI Pricing](https://docs.voyageai.com/docs/pricing)
- [voyage-3.5 Blog](https://blog.voyageai.com/2025/05/20/voyage-3-5/)
- [voyage-4-large MoE Blog](https://blog.voyageai.com/2026/03/03/moe-voyage-4-large/)
- [Cohere Embed v4 (Ailog)](https://app.ailog.fr/en/blog/news/cohere-embed-v4)
- [Cohere Pricing (MetaCTO)](https://www.metacto.com/blogs/cohere-pricing-explained-a-deep-dive-into-integration-development-costs)
- [OpenAI Pricing](https://platform.openai.com/docs/pricing)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [EmbeddingGemma Blog (Google)](https://developers.googleblog.com/introducing-embeddinggemma/)
- [EmbeddingGemma on Ollama](https://ollama.com/library/embeddinggemma)
- [Best Open-Source Embedding Models 2026 (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [10 Best Embedding Models 2026 (Openxcell)](https://www.openxcell.com/blog/best-embedding-models/)
- [Top Embedding Models on MTEB (Modal)](https://modal.com/blog/mteb-leaderboard-article)
- [Ollama Embedding Models](https://ollama.com/search?c=embedding)
- Model cards on Hugging Face for each model
