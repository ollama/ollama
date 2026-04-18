---
name: llm-inference-instructions
description: "Use when: working on model inference, implementing backends, handling LLM operations, or optimizing token generation in the llm/ package"
applyTo: "llm/**"
---

# LLM Inference Engine Instructions

## Overview
The `llm/` package handles model inference, tokenization, backend selection, and token streaming.

## Key Components

- **Backend selection**: CPU, CUDA, Metal (MLX), etc.
- **Model loading**: Loading weights and initializing inference
- **Token generation**: Streaming token-by-token responses
- **Context management**: GPU memory management and model unloading

## Backend Pattern

Each backend (mlx, cuda, cpu) provides:

```go
// Backend interface
type Backend interface {
    LoadModel(ctx context.Context, modelPath string) (Model, error)
    UnloadModel(model Model) error
    GenerateTokens(ctx context.Context, prompt string) <-chan Token
}
```

## Model Loading

```go
// Good: Load with context and timeout
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
defer cancel()

model, err := backend.LoadModel(ctx, modelPath)
if err != nil {
    return fmt.Errorf("failed to load model: %w", err)
}
defer model.Close()
```

## Token Generation & Streaming

```go
// Token stream pattern:
func (m *Model) GenerateTokens(ctx context.Context, prompt string) <-chan Token {
    tokenChan := make(chan Token)
    go func() {
        defer close(tokenChan)
        for {
            select {
            case <-ctx.Done():
                return
            default:
                token := m.NextToken()
                select {
                case tokenChan <- token:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    return tokenChan
}

// Consuming tokens:
for token := range model.GenerateTokens(ctx, prompt) {
    fmt.Print(token.Text)
    if err := processToken(token); err != nil {
        return err
    }
}
```

## Memory Management

- Always unload models after use to free GPU/CPU memory
- Use defer statements to ensure cleanup
- Monitor memory usage during long sessions
- Implement model swapping for multi-model scenarios

```go
// Pattern: Load, use, unload
model, err := LoadModel(modelPath)
if err != nil {
    return err
}
defer model.Unload()

// Use model
for token := range model.Generate(ctx, prompt) {
    // Process token
}
```

## Backend Selection Logic

```go
// Determine best backend based on:
// 1. Hardware availability (GPU detected?)
// 2. User preference (env var OLLAMA_BACKEND)
// 3. Model requirements (does model need GPU?)
// 4. Performance (benchmark results)
```

## Tokenization

```go
// Tokenizer interface:
type Tokenizer interface {
    Encode(text string) ([]int, error)
    Decode(tokens []int) (string, error)
}

// Model-specific tokenizers in tokenizer/ package
```

## Error Handling in Inference

```go
// Critical errors (must be handled):
- Model not found
- Insufficient memory
- Backend not available
- Context cancelled

// Wrap all errors with context:
if err != nil {
    return fmt.Errorf("inference failed for model %s: %w", modelName, err)
}
```

## Performance Optimization

- Use batch processing when possible
- Pre-allocate buffers for token streams
- Profile hot paths with pprof
- Cache model metadata
- Minimize context switching

## Testing Inference

```go
func TestGenerateTokens(t *testing.T) {
    model, err := LoadTestModel()
    require.NoError(t, err)
    defer model.Unload()
    
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    tokenCount := 0
    for range model.GenerateTokens(ctx, "test prompt") {
        tokenCount++
        if tokenCount > 100 { // Safety limit
            break
        }
    }
    
    assert.Greater(t, tokenCount, 0)
}
```

## Context Cancellation

- Always check `ctx.Done()` in tight loops
- Propagate context through function calls
- Return immediately on cancellation
- Clean up resources on context cancel

```go
// Pattern: Cancellation-aware loop
for {
    select {
    case <-ctx.Done():
        return ctx.Err()
    case token := <-tokenStream:
        process(token)
    }
}
```

## Model Architecture Support

When adding support for new architecture (e.g., Llama, Qwen):
1. Add converter in `convert/convert_modelname.go`
2. Register in `convert.go`
3. Add architecture-specific parameters in `template/`
4. Implement tokenizer if needed
5. Add tests for inference
6. Document in `docs/`
