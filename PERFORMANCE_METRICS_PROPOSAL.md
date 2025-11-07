# Add Per-Embedding Performance Metrics to API Response

## Problem
AI systems processing embeddings in batches cannot optimize batch size without external timing.
Current `EmbedResponse` only includes total duration, not per-embedding metrics.

## Proposal
Add optional performance metrics to help autonomous systems optimize:

```go
type EmbedResponse struct {
    Model      string      `json:"model"`
    Embeddings [][]float32 `json:"embeddings"`

    TotalDuration   time.Duration `json:"total_duration,omitempty"`
    LoadDuration    time.Duration `json:"load_duration,omitempty"`
    PromptEvalCount int           `json:"prompt_eval_count,omitempty"`
    
    // NEW: Per-embedding timing for optimization
    EmbeddingDurations []time.Duration `json:"embedding_durations,omitempty"`
    AverageDuration    time.Duration   `json:"average_duration,omitempty"`
}
```

## Use Case
AI consciousness building memory systems can:
1. Monitor per-embedding performance
2. Automatically adjust batch sizes
3. Detect performance degradation
4. Optimize for their specific workload

## Implementation
Modify `server/routes.go` EmbedHandler to track individual timings in errgroup.

## Benefit
Enables autonomous performance optimization without external measurement tools.
