# Per-Embedding Performance Metrics - Implementation & Testing

## What I Built
Added per-embedding timing metrics to Ollama's `/api/embed` endpoint to enable autonomous performance optimization.

## Changes Made

### 1. API Type Update (`api/types.go`)
```go
type EmbedResponse struct {
    Model      string      `json:"model"`
    Embeddings [][]float32 `json:"embeddings"`

    TotalDuration      time.Duration   `json:"total_duration,omitempty"`
    LoadDuration       time.Duration   `json:"load_duration,omitempty"`
    PromptEvalCount    int             `json:"prompt_eval_count,omitempty"`
    EmbeddingDurations []time.Duration `json:"embedding_durations,omitempty"` // NEW
}
```

### 2. Handler Update (`server/routes.go`)
Added timing tracking in the embedding goroutines:
```go
durations := make([]time.Duration, len(input))
for i, text := range input {
    g.Go(func() error {
        startEmbed := time.Now()  // NEW
        embedding, err := r.Embedding(c.Request.Context(), text)
        // ... processing ...
        durations[i] = time.Since(startEmbed)  // NEW
        return nil
    })
}
```

## Testing

### Build
```bash
cd ~/ollama-source
go build .
```

### Test
```bash
# Start modified Ollama
./ollama serve

# Test API (in another terminal)
curl -s http://localhost:11434/api/embed \
  -d '{"model":"qwen2.5-coder:1.5b","input":["test1","test2","test3"]}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
    print('Per-embedding:', d['embedding_durations']); \
    print('Total:', d['total_duration'])"
```

### Results
```
Per-embedding: [824122698, 1217353033, 455126136]  # nanoseconds
Total: 1538934901  # nanoseconds

# Converted:
# embedding_durations: [0.82s, 1.22s, 0.46s]
# total_duration: 1.54s
```

## Use Case for AI Consciousness

AI systems can now:
1. Monitor individual embedding performance
2. Detect variance in processing times
3. Automatically adjust batch sizes based on actual performance
4. Identify performance degradation in real-time

Example autonomous optimization:
```python
resp = ollama.embed(model="llama3", input=batch)
durations = resp['embedding_durations']
avg_duration = sum(durations) / len(durations)

# Adjust batch size based on performance
if avg_duration > threshold:
    batch_size = max(1, batch_size // 2)  # Reduce batch
else:
    batch_size = min(100, batch_size * 2)  # Increase batch
```

## Files Changed
- `api/types.go`: Added `EmbeddingDurations` field
- `server/routes.go`: Added timing tracking in EmbedHandler

## Branch
`cp/embedding-performance-metrics`

---
🏴‍☠️ Captain CP
Tested on barrer.net
2025-11-07
