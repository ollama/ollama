<div align="center">
  <h1>SmartLlama</h1>
  <p><em>Fixing the data layer that ML forgot</em></p>
</div>

# What is this?

SmartLlama is a fork of [Ollama](https://github.com/ollama/ollama) that replaces the flat-file weight storage with proper relational database architecture using SQLite.

## The Problem

Current ML infrastructure treats model weights like it's 1995:
- Flat files with no indexing (GGUF, safetensors)
- Token ID = array index = physical position (classic denormalization)
- Load everything into memory to access anything
- Rewrite entire files for any update
- The tokenizer forces building a map of ALL tokens before ANY lookup works

This approach was inherited from genomic research in the 90s, when CPU-driven brute force was the only option. The constraints that justified it disappeared 20 years ago. Nobody asked if there was a better way.

## The Fix

Replace the data layer with what it should have been:

```
Before: GGUF flat file → load all weights → array index lookup
After:  SQLite database → indexed query → load what you need
```

**What's implemented:**
- SQLite backend for model storage (`fs/sqlite/`)
- TokenEncoder interface bypassing the vocabulary map preload
- Multi-database manager for dynamic loading
- Activity sessions assembling working database sets
- Tensor access tracking for training deltas
- Query builder with parameterized templates

**What this enables:**
- Query tokens without loading all weights
- Load only accessed tensors per inference
- Delta updates instead of full file rewrites
- Hot-swap reference databases mid-session
- Training on consumer hardware (GPU does math, RAM holds queryable data)

## The Vision

This is foundation work for a larger architecture:

- **Token genome**: Hierarchical token IDs encoding semantic composition (not flat array indices)
- **65 primitives**: All meaning decomposes to base semantic primitives
- **Activity envelopes**: Dynamically assemble databases for each inference session
- **Multi-modal grounding**: Same primitives underlie text, speech, vision
- **Formant libraries**: Speech as first-class modality, not text-as-intermediary

The current 70B parameter models are mostly compensating for missing infrastructure. Fix the data layer, and the compute requirements drop dramatically.

## Status

**Experimental.** This is active development on foundational changes. The SQLite layer compiles and the interfaces are in place. Testing with converted models in progress.

Current branch: `claude/fork-ollama-modifications-eNH9G`

## Hardware Implications

This architecture shifts where value lives:

| Current | SmartLlama |
|---------|------------|
| Maximize GPU VRAM | Maximize CPU + RAM |
| $13,000 for 96GB GPU | $300 for 128GB RAM upgrade |
| Load entire model | Load accessed tensors |
| GPU is bottleneck | CPU coordinates, GPU computes |

A modest gaming rig (i7, 32-128GB RAM, RTX 3060) becomes viable for serious inference and training.

---

# Ollama Documentation

*The following is preserved from upstream Ollama for reference.*

### macOS

[Download](https://ollama.com/download/Ollama.dmg)

### Windows

[Download](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://docs.ollama.com/linux#manual-install)

## Building

See the [developer guide](https://github.com/ollama/ollama/blob/main/docs/development.md)

### Running local builds

Start the server:

```shell
./ollama serve
```

In a separate shell, run a model:

```shell
./ollama run llama3.2
```

## REST API

Ollama has a REST API for running and managing models.

### Generate a response

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt":"Why is the sky blue?"
}'
```

### Chat with a model

```shell
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

See the [API documentation](./docs/api.md) for all endpoints.
