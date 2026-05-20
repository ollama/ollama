# Gemma 4 Multi-Token Prediction (MTP)

Gemma 4 models ship with companion MTP assistant models that accelerate inference by predicting multiple tokens per forward pass. The assistant shares the target model's KV cache and operates directly on its hidden states via query-only attention, making it significantly more memory-efficient than traditional speculative decoding.

## Requirements

- NVIDIA GPU with CUDA support (e.g., RTX A6000, A100)
- Ollama built from this fork
- Gemma 4 GGUF with bundled assistant tensors (tensors prefixed with `draft.*`)

## How it works

The MTP assistant is a lightweight model (~10x smaller than the target) that predicts multiple tokens ahead. It shares the target's KV cache -- its attention layers compute only queries and read K/V directly from the target's cache. This means no second KV cache, no separate prefill, and minimal extra memory.

The inference cycle:
1. Target model forward pass on current token -> logits + hidden state
2. Draft N tokens using the assistant (single-position: RoPE position stays fixed, hidden state advances)
3. Verify all N draft tokens in one batched target forward pass
4. Accept the prefix that matches, reject the rest
5. Emit all accepted tokens + the correct next token

Greedy MTP (temperature=0) is mathematically equivalent to standard autoregressive decoding -- the output is identical, just faster.

## Quickstart

### With pre-built GGUFs

If you have a Gemma 4 GGUF that already contains `draft.*` assistant tensors:

```bash
ollama create mymodel -f Modelfile
ollama run mymodel
```

MTP engages automatically for greedy decoding (temperature=0). No flags needed.

### From HuggingFace safetensors

Download the target model and its MTP assistant. The assistant follows the `{model}-assistant` naming convention:

```
models/
  gemma-4-27B-it/
    config.json
    model-00001-of-00006.safetensors
    ...
  gemma-4-27B-it-assistant/
    config.json
    model.safetensors
```

Create with the DRAFT directive:

```
FROM ./gemma-4-27B-it
DRAFT ./gemma-4-27B-it-assistant
```

```bash
ollama create mymodel -f Modelfile
```

The converter bundles both target and assistant tensors into a single GGUF.

## When MTP activates

MTP is used when all of these are true:
- The model has `draft.*` tensors loaded (DraftModel is non-nil)
- Temperature is 0 (greedy decoding)
- Logprobs are not requested
- The sequence has not hit its prediction limit

## Tuning

The default draft count is 4 tokens per cycle. This can be adjusted via environment variables (coming soon).

## Supported models

Models with these HuggingFace architectures have MTP support:
- Target: `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`
- Assistant: `Gemma4AssistantForCausalLM`
