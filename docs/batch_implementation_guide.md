# Ollama Batch Support Implementation Guide

## Overview
This document outlines the implementation plan for adding batch inference support to Ollama, enabling multiple requests to share model weights for improved memory bandwidth utilization and throughput.

## Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)

#### 1.1 Environment Configuration
Add batch configuration to `envconfig/config.go`:

```go
// Batch processing configuration
func BatchEnabled() bool {
    return envconfig.Bool("OLLAMA_BATCH_ENABLED", false)
}

func BatchTimeout() time.Duration {
    if v := os.Getenv("OLLAMA_BATCH_TIMEOUT"); v != "" {
        if d, err := time.ParseDuration(v); err == nil {
            return d
        }
    }
    return 500 * time.Millisecond
}

func BatchSize() int {
    return envconfig.Int("OLLAMA_BATCH_SIZE", 8)
}

func BatchMemoryFactor() float64 {
    return envconfig.Float("OLLAMA_BATCH_MEMORY_FACTOR", 1.5)
}
```

#### 1.2 API Types Extension
Extend `api/types.go` with batch request types:

```go
// Batch request types
type BatchGenerateRequest struct {
    Requests     []GenerateRequest `json:"requests"`
    MaxBatchSize int              `json:"max_batch_size,omitempty"`
    BatchTimeout *Duration        `json:"batch_timeout,omitempty"`
}

type BatchChatRequest struct {
    Requests     []ChatRequest `json:"requests"`
    MaxBatchSize int          `json:"max_batch_size,omitempty"`
    BatchTimeout *Duration    `json:"batch_timeout,omitempty"`
}

type BatchGenerateResponse struct {
    Responses   []GenerateResponse `json:"responses"`
    BatchId     string            `json:"batch_id"`
    ProcessedAt time.Time         `json:"processed_at"`
    BatchStats  BatchStats        `json:"batch_stats,omitempty"`
}

type BatchChatResponse struct {
    Responses   []ChatResponse `json:"responses"`
    BatchId     string        `json:"batch_id"`
    ProcessedAt time.Time     `json:"processed_at"`
    BatchStats  BatchStats    `json:"batch_stats,omitempty"`
}

type BatchStats struct {
    BatchSize        int           `json:"batch_size"`
    ProcessingTime   time.Duration `json:"processing_time"`
    MemoryEfficiency float64       `json:"memory_efficiency"`
    ThroughputGain   float64       `json:"throughput_gain"`
}
```

#### 1.3 LLM Server Interface Extension
Extend `llm/server.go` interface:

```go
type LlamaServer interface {
    // Existing methods...
    Ping(ctx context.Context) error
    WaitUntilRunning(ctx context.Context) error
    Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
    // ... other existing methods

    // New batch methods
    BatchCompletion(ctx context.Context, reqs []CompletionRequest, fn func([]CompletionResponse)) error
    GetMaxBatchSize() int
    SupportsBatching() bool
    EstimateBatchMemory(batchSize int) uint64
}
```

### Phase 2: Request Accumulator (Weeks 2-3)

#### 2.1 Batch Accumulator Implementation
Create `server/batch_accumulator.go`:

```go
package server

import (
    "context"
    "sync"
    "time"
    "github.com/google/uuid"
    "github.com/ollama/ollama/api"
    "github.com/ollama/ollama/llm"
)

type RequestBatch struct {
    ID          string
    ModelPath   string
    Requests    []PendingRequest
    CreatedAt   time.Time
    MaxWaitTime time.Duration
}

type PendingRequest struct {
    Request    llm.CompletionRequest
    ResponseCh chan llm.CompletionResponse
    ErrorCh    chan error
    Context    context.Context
}

type BatchAccumulator struct {
    mu           sync.RWMutex
    pending      map[string]*RequestBatch // keyed by model path
    maxBatchSize int
    timeout      time.Duration
    enabled      bool
    ticker       *time.Ticker
    batchCh      chan *RequestBatch
}

func NewBatchAccumulator(maxSize int, timeout time.Duration, enabled bool) *BatchAccumulator {
    ba := &BatchAccumulator{
        pending:      make(map[string]*RequestBatch),
        maxBatchSize: maxSize,
        timeout:      timeout,
        enabled:      enabled,
        ticker:       time.NewTicker(timeout / 4), // Check 4x per timeout period
        batchCh:      make(chan *RequestBatch, 100),
    }
    
    if enabled {
        go ba.processTicker()
    }
    
    return ba
}

func (ba *BatchAccumulator) AddRequest(modelPath string, req llm.CompletionRequest, 
    responseCh chan llm.CompletionResponse, errorCh chan error, ctx context.Context) bool {
    
    if !ba.enabled {
        return false
    }
    
    ba.mu.Lock()
    defer ba.mu.Unlock()
    
    batch, exists := ba.pending[modelPath]
    if !exists {
        batch = &RequestBatch{
            ID:          uuid.New().String(),
            ModelPath:   modelPath,
            Requests:    make([]PendingRequest, 0, ba.maxBatchSize),
            CreatedAt:   time.Now(),
            MaxWaitTime: ba.timeout,
        }
        ba.pending[modelPath] = batch
    }
    
    pendingReq := PendingRequest{
        Request:    req,
        ResponseCh: responseCh,
        ErrorCh:    errorCh,
        Context:    ctx,
    }
    
    batch.Requests = append(batch.Requests, pendingReq)
    
    // Check if batch is ready to process
    if len(batch.Requests) >= ba.maxBatchSize || 
       time.Since(batch.CreatedAt) >= ba.timeout {
        ba.flushBatch(modelPath)
    }
    
    return true
}

func (ba *BatchAccumulator) flushBatch(modelPath string) {
    if batch, exists := ba.pending[modelPath]; exists {
        delete(ba.pending, modelPath)
        ba.batchCh <- batch
    }
}

func (ba *BatchAccumulator) processTicker() {
    for range ba.ticker.C {
        ba.mu.Lock()
        now := time.Now()
        for modelPath, batch := range ba.pending {
            if now.Sub(batch.CreatedAt) >= ba.timeout {
                ba.flushBatch(modelPath)
            }
        }
        ba.mu.Unlock()
    }
}

func (ba *BatchAccumulator) GetBatch() <-chan *RequestBatch {
    return ba.batchCh
}
```

### Phase 3: Scheduler Integration (Weeks 3-4)

#### 3.1 Scheduler Modifications
Update `server/sched.go` to support batching:

```go
type Scheduler struct {
    // Existing fields...
    pendingReqCh  chan *LlmRequest
    finishedReqCh chan *LlmRequest
    expiredCh     chan *runnerRef
    unloadedCh    chan any
    loaded        map[string]*runnerRef
    loadedMu      sync.Mutex
    
    // New batch fields
    batchAccumulator *BatchAccumulator
    batchEnabled     bool
    batchProcessCh   chan *RequestBatch
}

func InitScheduler(ctx context.Context) *Scheduler {
    maxQueue := envconfig.MaxQueue()
    batchEnabled := envconfig.BatchEnabled()
    batchSize := envconfig.BatchSize()
    batchTimeout := envconfig.BatchTimeout()
    
    sched := &Scheduler{
        // Existing initialization...
        pendingReqCh:  make(chan *LlmRequest, maxQueue),
        finishedReqCh: make(chan *LlmRequest, maxQueue),
        expiredCh:     make(chan *runnerRef, maxQueue),
        unloadedCh:    make(chan any, maxQueue),
        loaded:        make(map[string]*runnerRef),
        newServerFn:   llm.NewLlamaServer,
        getGpuFn:      discover.GetGPUInfo,
        getCpuFn:      discover.GetCPUInfo,
        reschedDelay:  250 * time.Millisecond,
        
        // New batch initialization
        batchEnabled:     batchEnabled,
        batchProcessCh:   make(chan *RequestBatch, maxQueue),
    }
    
    if batchEnabled {
        sched.batchAccumulator = NewBatchAccumulator(batchSize, batchTimeout, true)
        slog.Info("batch processing enabled", 
            "max_batch_size", batchSize, 
            "timeout", batchTimeout)
    }
    
    sched.loadFn = sched.load
    return sched
}

// New method for batch-aware request processing
func (s *Scheduler) GetRunnerWithBatching(c context.Context, model *Model, 
    opts api.Options, sessionDuration *api.Duration, 
    req llm.CompletionRequest) (chan llm.CompletionResponse, chan error) {
    
    responseCh := make(chan llm.CompletionResponse)
    errorCh := make(chan error, 1)
    
    // Try batch processing first if enabled
    if s.batchEnabled && s.batchAccumulator != nil {
        if s.batchAccumulator.AddRequest(model.ModelPath, req, responseCh, errorCh, c) {
            return responseCh, errorCh
        }
    }
    
    // Fall back to individual processing
    runnerCh, errCh := s.GetRunner(c, model, opts, sessionDuration)
    go func() {
        select {
        case runner := <-runnerCh:
            err := runner.llama.Completion(c, req, func(resp llm.CompletionResponse) {
                responseCh <- resp
            })
            if err != nil {
                errorCh <- err
            }
            close(responseCh)
        case err := <-errCh:
            errorCh <- err
            close(responseCh)
        }
    }()
    
    return responseCh, errorCh
}
```

### Phase 4: Batch Processing Engine (Weeks 4-5)

#### 4.1 Batch Processing Implementation
Create `server/batch_processor.go`:

```go
package server

import (
    "context"
    "fmt"
    "log/slog"
    "sync"
    "time"
    
    "github.com/ollama/ollama/llm"
)

type BatchProcessor struct {
    scheduler *Scheduler
}

func (s *Scheduler) processBatchRequests(ctx context.Context) {
    if !s.batchEnabled || s.batchAccumulator == nil {
        return
    }
    
    slog.Debug("starting batch request processor")
    
    for {
        select {
        case <-ctx.Done():
            slog.Debug("shutting down batch processor")
            return
        case batch := <-s.batchAccumulator.GetBatch():
            s.executeBatch(ctx, batch)
        }
    }
}

func (s *Scheduler) executeBatch(ctx context.Context, batch *RequestBatch) {
    startTime := time.Now()
    batchSize := len(batch.Requests)
    
    slog.Info("executing batch", 
        "batch_id", batch.ID,
        "model", batch.ModelPath,
        "size", batchSize)
    
    // Get runner for the model
    s.loadedMu.Lock()
    runner := s.loaded[batch.ModelPath]
    s.loadedMu.Unlock()
    
    if runner == nil {
        s.handleBatchError(batch, fmt.Errorf("model not loaded: %s", batch.ModelPath))
        return
    }
    
    // Check if runner supports batching
    if !runner.llama.SupportsBatching() {
        slog.Debug("runner doesn't support batching, processing individually", 
            "batch_id", batch.ID)
        s.processBatchIndividually(ctx, batch, runner)
        return
    }
    
    // Extract completion requests
    completionReqs := make([]llm.CompletionRequest, batchSize)
    for i, req := range batch.Requests {
        completionReqs[i] = req.Request
    }
    
    // Execute batch completion
    err := runner.llama.BatchCompletion(ctx, completionReqs, func(responses []llm.CompletionResponse) {
        s.handleBatchResponses(batch, responses, startTime)
    })
    
    if err != nil {
        slog.Error("batch completion failed", "error", err, "batch_id", batch.ID)
        s.handleBatchError(batch, err)
    }
}

func (s *Scheduler) handleBatchResponses(batch *RequestBatch, responses []llm.CompletionResponse, startTime time.Time) {
    processingTime := time.Since(startTime)
    batchSize := len(batch.Requests)
    
    if len(responses) != batchSize {
        slog.Error("batch response count mismatch", 
            "expected", batchSize, 
            "received", len(responses),
            "batch_id", batch.ID)
        s.handleBatchError(batch, fmt.Errorf("response count mismatch"))
        return
    }
    
    // Send responses to individual request channels
    for i, req := range batch.Requests {
        select {
        case req.ResponseCh <- responses[i]:
        case <-req.Context.Done():
            slog.Debug("request context cancelled during batch response", "batch_id", batch.ID)
        }
        close(req.ResponseCh)
    }
    
    slog.Info("batch completed successfully",
        "batch_id", batch.ID,
        "size", batchSize,
        "processing_time", processingTime,
        "avg_time_per_request", processingTime/time.Duration(batchSize))
}

func (s *Scheduler) handleBatchError(batch *RequestBatch, err error) {
    for _, req := range batch.Requests {
        select {
        case req.ErrorCh <- err:
        case <-req.Context.Done():
        }
        close(req.ResponseCh)
    }
}

func (s *Scheduler) processBatchIndividually(ctx context.Context, batch *RequestBatch, runner *runnerRef) {
    var wg sync.WaitGroup
    
    for _, req := range batch.Requests {
        wg.Add(1)
        go func(r PendingRequest) {
            defer wg.Done()
            
            err := runner.llama.Completion(r.Context, r.Request, func(resp llm.CompletionResponse) {
                select {
                case r.ResponseCh <- resp:
                case <-r.Context.Done():
                }
            })
            
            if err != nil {
                select {
                case r.ErrorCh <- err:
                case <-r.Context.Done():
                }
            }
            close(r.ResponseCh)
        }(req)
    }
    
    wg.Wait()
}
```

### Phase 5: Route Integration (Week 5)

#### 5.1 Add Batch Routes
Update `server/routes.go`:

```go
// Add to GenerateRoutes function
func (s *Server) GenerateRoutes(rc *ollama.Registry) (http.Handler, error) {
    // ... existing routes ...
    
    // Batch inference endpoints
    if envconfig.BatchEnabled() {
        r.POST("/api/batch/generate", s.BatchGenerateHandler)
        r.POST("/api/batch/chat", s.BatchChatHandler)
        r.GET("/api/batch/status", s.BatchStatusHandler)
    }
    
    // ... rest of existing routes ...
}

func (s *Server) BatchGenerateHandler(c *gin.Context) {
    var req api.BatchGenerateRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
        return
    }
    
    if len(req.Requests) == 0 {
        c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "no requests provided"})
        return
    }
    
    batchId := uuid.New().String()
    startTime := time.Now()
    responses := make([]api.GenerateResponse, len(req.Requests))
    
    // Process batch
    // Implementation details for coordinating batch processing...
    
    resp := api.BatchGenerateResponse{
        Responses:   responses,
        BatchId:     batchId,
        ProcessedAt: time.Now(),
        BatchStats: api.BatchStats{
            BatchSize:      len(req.Requests),
            ProcessingTime: time.Since(startTime),
            // Add efficiency calculations...
        },
    }
    
    c.JSON(http.StatusOK, resp)
}

func (s *Server) BatchChatHandler(c *gin.Context) {
    // Similar implementation for chat requests
}

func (s *Server) BatchStatusHandler(c *gin.Context) {
    status := map[string]interface{}{
        "batch_enabled":    envconfig.BatchEnabled(),
        "max_batch_size":   envconfig.BatchSize(),
        "batch_timeout":    envconfig.BatchTimeout().String(),
        "pending_batches":  s.sched.getPendingBatchCount(),
        "processed_batches": s.sched.getProcessedBatchCount(),
    }
    c.JSON(http.StatusOK, status)
}
```

## Testing Strategy

### Unit Tests
1. **Batch Accumulator Tests**: Test request batching logic
2. **Memory Estimation Tests**: Verify batch memory calculations
3. **API Type Tests**: Validate serialization/deserialization

### Integration Tests
1. **End-to-End Batch Tests**: Full batch request processing
2. **Fallback Tests**: Verify graceful degradation to individual processing
3. **Error Handling Tests**: Batch failure scenarios

### Performance Tests
1. **Throughput Benchmarks**: Compare batch vs individual processing
2. **Memory Usage Tests**: Validate memory efficiency gains
3. **Latency Tests**: Measure batch processing latency impact

## Deployment Considerations

### Configuration
- Start with batching disabled by default
- Provide clear documentation for optimal batch settings
- Add monitoring and metrics for batch efficiency

### Monitoring
- Batch processing times
- Memory efficiency gains
- Throughput improvements
- Error rates

### Rollout Strategy
1. Deploy with batching disabled
2. Enable for specific high-throughput customers
3. Gradually enable for broader deployment
4. Monitor performance and adjust defaults

This implementation provides a solid foundation for batch processing in Ollama while maintaining backward compatibility and providing significant performance improvements for both CPU and GPU inference.
