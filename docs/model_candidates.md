# Model Candidates for s390x Testing

## Overview

This document catalogs small language models with GGUF variants that are ideal candidates for testing Ollama on s390x architecture. These models are specifically selected for their compact size, making them suitable for initial testing, validation, and development work on resource-constrained or specialized architectures.

## Why Small Models for s390x Testing?

Small models (< 5B parameters) are crucial for s390x architecture testing for several reasons:

1. **Faster Iteration**: Smaller models load and run faster, enabling quicker testing cycles during development
2. **Resource Efficiency**: s390x systems may have different memory and compute characteristics; small models reduce resource requirements
3. **Architecture Validation**: Small models allow us to validate core functionality (GGUF loading, inference, quantization) without excessive overhead
4. **Debugging**: Issues are easier to isolate and debug with smaller model files and simpler architectures
5. **CI/CD Integration**: Compact models can be integrated into automated testing pipelines more easily

## Model Candidates

| Model Name | Size | Type/Purpose | HuggingFace Link | Notes |
|------------|------|--------------|------------------|-------|
| Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking | 1B | Instruction-tuned, Thinking | [Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF](https://huggingface.co/Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF) | Gemma-based with thinking capabilities, good for testing instruction following |
| Qwen2.5-0.5B-Instruct | 0.5B | Instruction-tuned | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | Smallest model in the list, excellent for quick validation tests |
| LFM2.5-1.2B-Instruct | 1.2B | Instruction-tuned | [LiquidAI/LFM2.5-1.2B-Instruct-GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) | Liquid AI's foundation model, tests alternative architectures |
| Gemma-3-1B-it | 1B | Instruction-tuned | [unsloth/gemma-3-1b-it-GGUF](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF) | Standard Gemma 3 variant, baseline for Gemma architecture testing |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Reasoning, Distilled | [unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) | DeepSeek reasoning model distilled to Qwen, tests reasoning capabilities |
| Granite-Speech-4.1-2B | 2B | Speech/Audio | [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | IBM's speech model, tests multimodal capabilities (may need GGUF conversion) |

## References

- [Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp s390x Support](https://github.com/ggerganov/llama.cpp)
- [HuggingFace Model Hub](https://huggingface.co/models)

## Changelog

- **2026-06-16**: Initial document creation with 6 model candidates