# Local LLM Integration Guide

LITRIS supports local LLM inference through two providers:

- **Ollama**: Server-based inference with easy model management
- **llama.cpp**: Direct model loading for maximum control

Both options provide free inference on your own hardware with no API costs.

## Quick Comparison

| Feature | Ollama | llama.cpp |
| ------- | ------ | --------- |
| Setup complexity | Easy | Moderate |
| Model management | Built-in | Manual download |
| Server required | Yes (local) | No |
| GPU support | Automatic | Manual config |
| Memory efficiency | Good | Excellent |
| Multiple models | Easy switching | One at a time |

## Ollama Integration

### Overview

Ollama provides a simple way to run LLMs locally with automatic GPU detection and easy model management. It runs as a local server that LITRIS connects to.

### Installation

1. Download and install Ollama from [ollama.com](https://ollama.com/)

2. Install the Python client:

```bash
pip install ollama
```

3. Pull a model:

```bash
# Recommended for paper extraction
ollama pull llama3

# Larger, more capable model
ollama pull llama3.1:70b

# Smaller, faster model
ollama pull llama3.2:3b
```

### Configuration

Configure LITRIS to use Ollama in `config.yaml`:

```yaml
extraction:
  provider: "ollama"
  model: "llama3"
  mode: "api"
```

Or use command-line options:

```bash
python scripts/build_index.py --provider ollama --model llama3
```

### Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

### Supported Models

Ollama supports many models. Recommended for paper extraction:

| Model | Size | Description |
| ----- | ---- | ----------- |
| `llama3.3` | 70B | Best quality, requires high-end GPU |
| `llama3.1:70b` | 70B | Excellent quality |
| `llama3.1:8b` | 8B | Good balance of quality and speed |
| `llama3` | 8B | Good general-purpose model |
| `llama3.2:3b` | 3B | Fast, lower quality |
| `mistral` | 7B | Efficient, good quality |
| `mixtral` | 8x7B | MoE model, high capability |
| `qwen2.5` | Various | Strong multilingual support |

### Usage Example

```python
from src.analysis.llm_factory import create_llm_client

# Create Ollama client
client = create_llm_client(
    provider="ollama",
    model="llama3",
)

# Check connection
if client.check_connection():
    print("Ollama server is running")
    print(f"Available models: {client.list_local_models()}")
else:
    print("Start Ollama server: ollama serve")
```

### Troubleshooting

**Connection refused**

```text
Error: Connection refused at http://localhost:11434
```

Solution: Start the Ollama server:

```bash
ollama serve
```

**Model not found**

```text
Error: model 'llama3' not found
```

Solution: Pull the model:

```bash
ollama pull llama3
```

**Slow inference**

- Ensure GPU is being used: `ollama run llama3 --verbose`
- Try a smaller model: `llama3.2:3b` or `mistral`
- Reduce context length in config

## llama.cpp Integration

### Overview

llama.cpp provides direct model loading without a server, offering maximum control over inference parameters. Best for advanced users who want fine-grained control or minimal dependencies.

### Installation

```bash
pip install llama-cpp-python
```

For GPU acceleration (recommended):

```bash
# CUDA (NVIDIA GPUs)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Metal (Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# ROCm (AMD GPUs)
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Model Download

Download GGUF models from Hugging Face. Recommended sources:

- [TheBloke](https://huggingface.co/TheBloke) - Quantized versions of popular models
- [QuantFactory](https://huggingface.co/QuantFactory) - High-quality quantizations
- [Official model repos](https://huggingface.co/meta-llama) - Original model authors

Example download:

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir ./models
```

### Configuration

Configure LITRIS to use llama.cpp in `config.yaml`:

```yaml
extraction:
  provider: "llamacpp"
  model: "llama-3"
  mode: "api"
  llamacpp:
    model_path: "/path/to/model.gguf"
    n_ctx: 8192
    n_gpu_layers: -1  # -1 = all layers on GPU
```

Or use command-line options:

```bash
python scripts/build_index.py --provider llamacpp --model-path /path/to/model.gguf
```

### Configuration Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `model_path` | Path to GGUF model file | Required |
| `n_ctx` | Context window size | 8192 |
| `n_gpu_layers` | Layers to offload to GPU (-1 = all) | -1 |
| `verbose` | Enable verbose logging | False |

### Quantization Levels

GGUF models come in different quantization levels:

| Quantization | Size | Quality | Speed |
| ------------ | ---- | ------- | ----- |
| Q8_0 | Largest | Best | Slowest |
| Q6_K | Large | Excellent | Slow |
| Q5_K_M | Medium | Very good | Medium |
| Q4_K_M | Small | Good | Fast |
| Q4_0 | Smallest | Acceptable | Fastest |

For paper extraction, `Q5_K_M` or `Q4_K_M` is recommended.

### Usage Example

```python
from src.analysis.llm_factory import create_llm_client

# Create llama.cpp client
client = create_llm_client(
    provider="llamacpp",
    model_path="/models/llama-3-8b.Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,  # All layers on GPU
)

# Extract from a paper
result = client.extract(
    paper_id="test-001",
    title="Example Paper",
    authors="Smith, J.",
    year=2024,
    item_type="journalArticle",
    text="Paper text content...",
)
```

### Troubleshooting

**Model file not found**

```text
ValueError: Model file not found: /path/to/model.gguf
```

Solution: Verify the model path is correct and the file exists.

**Out of memory**

```text
CUDA out of memory
```

Solutions:
- Reduce `n_gpu_layers` to partially offload to CPU
- Use a smaller quantization (Q4_0 instead of Q8_0)
- Use a smaller model

**Slow CPU inference**

Solution: Ensure GPU acceleration is compiled:

```bash
# Check if CUDA is available
python -c "from llama_cpp import Llama; print('llama.cpp loaded')"
```

If GPU not detected, reinstall with CUDA support (see Installation).

## Hardware Requirements

### Minimum Requirements

| Component | Ollama | llama.cpp |
| --------- | ------ | --------- |
| RAM | 8GB | 8GB |
| VRAM | 4GB | 4GB |
| Storage | 5GB per model | 5GB per model |

### Recommended for Paper Extraction

| Model Size | RAM | VRAM | Notes |
| ---------- | --- | ---- | ----- |
| 7B (Q4) | 8GB | 6GB | Fast, acceptable quality |
| 7B (Q8) | 16GB | 8GB | Better quality |
| 13B (Q4) | 16GB | 10GB | Good balance |
| 70B (Q4) | 64GB | 48GB | Best quality |

### Performance Tips

1. **Use GPU acceleration** - 10-50x faster than CPU
2. **Match quantization to VRAM** - Larger quantization = better quality
3. **Set appropriate context size** - Larger papers need more context
4. **Batch processing** - Process papers sequentially to avoid memory issues

## Cost Comparison

| Provider | Cost per Paper | Monthly (500 papers) |
| -------- | -------------- | -------------------- |
| Anthropic Batch API | ~$0.06 | ~$31 |
| OpenAI API | ~$0.10 | ~$50 |
| Gemini API | ~$0.03 | ~$15 |
| Ollama | $0 (electricity) | $0 (electricity) |
| llama.cpp | $0 (electricity) | $0 (electricity) |

Local inference is essentially free once hardware is available.

## Choosing Between Ollama and llama.cpp

**Use Ollama if:**

- You want easy setup and model management
- You need to switch between models frequently
- You prefer a server-based architecture
- You're new to local LLMs

**Use llama.cpp if:**

- You want maximum control over inference
- You need minimal dependencies
- You're running in a containerized environment
- You need to fine-tune memory usage

## Running the Smoketest

Verify your local LLM setup:

```bash
python scripts/smoketest_local_llm.py
```

This tests:
- Factory integration
- Client imports
- Basic initialization
- Connection (Ollama only)
