package api

import (
	"time"
)

// BatchGenerateRequest represents a request to process multiple generate requests as a batch
type BatchGenerateRequest struct {
	// Requests is the list of individual generate requests to process
	Requests []GenerateRequest `json:"requests"`

	// MaxBatchSize optionally overrides the server's default maximum batch size
	MaxBatchSize int `json:"max_batch_size,omitempty"`

	// BatchTimeout optionally overrides the server's default batch timeout
	BatchTimeout *Duration `json:"batch_timeout,omitempty"`

	// Priority can be used to prioritize certain batches (higher values = higher priority)
	Priority int `json:"priority,omitempty"`
}

// BatchChatRequest represents a request to process multiple chat requests as a batch
type BatchChatRequest struct {
	// Requests is the list of individual chat requests to process
	Requests []ChatRequest `json:"requests"`

	// MaxBatchSize optionally overrides the server's default maximum batch size
	MaxBatchSize int `json:"max_batch_size,omitempty"`

	// BatchTimeout optionally overrides the server's default batch timeout
	BatchTimeout *Duration `json:"batch_timeout,omitempty"`

	// Priority can be used to prioritize certain batches (higher values = higher priority)
	Priority int `json:"priority,omitempty"`
}

// BatchGenerateResponse represents the response from a batch generate request
type BatchGenerateResponse struct {
	// Responses contains the individual responses corresponding to each request
	Responses []GenerateResponse `json:"responses"`

	// BatchId is a unique identifier for this batch
	BatchId string `json:"batch_id"`

	// ProcessedAt indicates when the batch was processed
	ProcessedAt time.Time `json:"processed_at"`

	// BatchStats provides performance metrics for the batch
	BatchStats BatchStats `json:"batch_stats,omitempty"`
}

// BatchChatResponse represents the response from a batch chat request
type BatchChatResponse struct {
	// Responses contains the individual responses corresponding to each request
	Responses []ChatResponse `json:"responses"`

	// BatchId is a unique identifier for this batch
	BatchId string `json:"batch_id"`

	// ProcessedAt indicates when the batch was processed
	ProcessedAt time.Time `json:"processed_at"`

	// BatchStats provides performance metrics for the batch
	BatchStats BatchStats `json:"batch_stats,omitempty"`
}

// BatchEmbedRequest represents a request to process multiple embed requests as a batch
type BatchEmbedRequest struct {
	// Requests is the list of individual embed requests to process
	Requests []EmbedRequest `json:"requests"`

	// MaxBatchSize optionally overrides the server's default maximum batch size
	MaxBatchSize int `json:"max_batch_size,omitempty"`

	// BatchTimeout optionally overrides the server's default batch timeout
	BatchTimeout *Duration `json:"batch_timeout,omitempty"`

	// Priority can be used to prioritize certain batches (higher values = higher priority)
	Priority int `json:"priority,omitempty"`
}

// BatchEmbedResponse represents the response from a batch embed request
type BatchEmbedResponse struct {
	// Responses contains the individual responses corresponding to each request
	Responses []EmbedResponse `json:"responses"`

	// BatchId is a unique identifier for this batch
	BatchId string `json:"batch_id"`

	// ProcessedAt indicates when the batch was processed
	ProcessedAt time.Time `json:"processed_at"`

	// BatchStats provides performance metrics for the batch
	BatchStats BatchStats `json:"batch_stats,omitempty"`
}

// BatchStats provides performance and efficiency metrics for batch processing
type BatchStats struct {
	// BatchSize is the number of requests processed in this batch
	BatchSize int `json:"batch_size"`

	// ProcessingTime is the total time taken to process the batch
	ProcessingTime time.Duration `json:"processing_time"`

	// WaitTime is the time spent waiting to accumulate the batch
	WaitTime time.Duration `json:"wait_time"`

	// MemoryEfficiency represents the memory bandwidth savings (1.0 = no savings, 4.0 = 4x efficiency)
	MemoryEfficiency float64 `json:"memory_efficiency"`

	// ThroughputGain represents the throughput improvement over individual processing
	ThroughputGain float64 `json:"throughput_gain"`

	// AvgTimePerRequest is the average processing time per request in the batch
	AvgTimePerRequest time.Duration `json:"avg_time_per_request"`

	// BatchMethod indicates how the batch was processed ("batched", "individual", "mixed")
	BatchMethod string `json:"batch_method"`
}

// BatchStatusResponse provides information about the current state of batch processing
type BatchStatusResponse struct {
	// BatchEnabled indicates whether batch processing is currently enabled
	BatchEnabled bool `json:"batch_enabled"`

	// MaxBatchSize is the maximum number of requests that can be batched together
	MaxBatchSize int `json:"max_batch_size"`

	// BatchTimeout is the maximum time to wait for accumulating requests
	BatchTimeout Duration `json:"batch_timeout"`

	// CurrentBatches is the number of batches currently being processed
	CurrentBatches int `json:"current_batches"`

	// PendingRequests is the number of individual requests waiting to be batched
	PendingRequests int `json:"pending_requests"`

	// ProcessedBatches is the total number of batches processed since server start
	ProcessedBatches int64 `json:"processed_batches"`

	// AverageEfficiency is the average memory efficiency across all processed batches
	AverageEfficiency float64 `json:"average_efficiency"`

	// SupportedModels lists the models that support batch processing
	SupportedModels []string `json:"supported_models,omitempty"`
}

// BatchErrorResponse represents an error that occurred during batch processing
type BatchErrorResponse struct {
	// BatchId is the identifier of the failed batch
	BatchId string `json:"batch_id"`

	// Error is the error message
	Error string `json:"error"`

	// FailedRequests indicates which requests in the batch failed (by index)
	FailedRequests []int `json:"failed_requests,omitempty"`

	// PartialResults contains any successful responses if some requests succeeded
	PartialResults []interface{} `json:"partial_results,omitempty"`
}
