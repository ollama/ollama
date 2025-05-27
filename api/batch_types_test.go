package api

import (
	"encoding/json"
	"testing"
	"time"
)

func TestBatchGenerateRequestSerialization(t *testing.T) {
	req := BatchGenerateRequest{
		Requests: []GenerateRequest{
			{Model: "test-model", Prompt: "Hello"},
			{Model: "test-model", Prompt: "World"},
		},
		MaxBatchSize: 8,
		Priority:     1,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal BatchGenerateRequest: %v", err)
	}

	var unmarshaled BatchGenerateRequest
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal BatchGenerateRequest: %v", err)
	}

	if len(unmarshaled.Requests) != 2 {
		t.Errorf("Expected 2 requests, got %d", len(unmarshaled.Requests))
	}

	if unmarshaled.MaxBatchSize != 8 {
		t.Errorf("Expected MaxBatchSize 8, got %d", unmarshaled.MaxBatchSize)
	}

	if unmarshaled.Priority != 1 {
		t.Errorf("Expected Priority 1, got %d", unmarshaled.Priority)
	}
}

func TestBatchChatRequestSerialization(t *testing.T) {
	req := BatchChatRequest{
		Requests: []ChatRequest{
			{Model: "test-model", Messages: []Message{{Role: "user", Content: "Hello"}}},
			{Model: "test-model", Messages: []Message{{Role: "user", Content: "World"}}},
		},
		MaxBatchSize: 4,
		Priority:     2,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal BatchChatRequest: %v", err)
	}

	var unmarshaled BatchChatRequest
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal BatchChatRequest: %v", err)
	}

	if len(unmarshaled.Requests) != 2 {
		t.Errorf("Expected 2 requests, got %d", len(unmarshaled.Requests))
	}

	if unmarshaled.MaxBatchSize != 4 {
		t.Errorf("Expected MaxBatchSize 4, got %d", unmarshaled.MaxBatchSize)
	}
}

func TestBatchEmbedRequestSerialization(t *testing.T) {
	req := BatchEmbedRequest{
		Requests: []EmbedRequest{
			{Model: "embed-model", Input: "text1"},
			{Model: "embed-model", Input: "text2"},
		},
		MaxBatchSize: 16,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal BatchEmbedRequest: %v", err)
	}

	var unmarshaled BatchEmbedRequest
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal BatchEmbedRequest: %v", err)
	}

	if len(unmarshaled.Requests) != 2 {
		t.Errorf("Expected 2 requests, got %d", len(unmarshaled.Requests))
	}

	if unmarshaled.MaxBatchSize != 16 {
		t.Errorf("Expected MaxBatchSize 16, got %d", unmarshaled.MaxBatchSize)
	}
}

func TestBatchGenerateResponseSerialization(t *testing.T) {
	response := BatchGenerateResponse{
		Responses: []GenerateResponse{
			{Model: "test-model", Response: "Hello response", Done: true},
			{Model: "test-model", Response: "World response", Done: true},
		},
		BatchId:     "batch-123",
		ProcessedAt: time.Now(),
		BatchStats: BatchStats{
			BatchSize:        2,
			ProcessingTime:   100 * time.Millisecond,
			WaitTime:         50 * time.Millisecond,
			MemoryEfficiency: 3.2,
			ThroughputGain:   2.5,
			BatchMethod:      "batched",
		},
	}

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("Failed to marshal BatchGenerateResponse: %v", err)
	}

	var unmarshaled BatchGenerateResponse
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal BatchGenerateResponse: %v", err)
	}

	if len(unmarshaled.Responses) != 2 {
		t.Errorf("Expected 2 responses, got %d", len(unmarshaled.Responses))
	}

	if unmarshaled.BatchId != "batch-123" {
		t.Errorf("Expected BatchId 'batch-123', got '%s'", unmarshaled.BatchId)
	}

	if unmarshaled.BatchStats.BatchSize != 2 {
		t.Errorf("Expected BatchSize 2, got %d", unmarshaled.BatchStats.BatchSize)
	}

	if unmarshaled.BatchStats.MemoryEfficiency != 3.2 {
		t.Errorf("Expected MemoryEfficiency 3.2, got %f", unmarshaled.BatchStats.MemoryEfficiency)
	}
}

func TestBatchStatsCalculations(t *testing.T) {
	stats := BatchStats{
		BatchSize:      4,
		ProcessingTime: 400 * time.Millisecond,
		WaitTime:       100 * time.Millisecond,
	}

	// Calculate average time per request
	stats.AvgTimePerRequest = stats.ProcessingTime / time.Duration(stats.BatchSize)
	expected := 100 * time.Millisecond
	if stats.AvgTimePerRequest != expected {
		t.Errorf("Expected AvgTimePerRequest %v, got %v", expected, stats.AvgTimePerRequest)
	}

	// Test efficiency calculations
	stats.MemoryEfficiency = 4.0 // 4x memory efficiency
	stats.ThroughputGain = 3.5   // 3.5x throughput gain

	if stats.MemoryEfficiency < 1.0 {
		t.Errorf("MemoryEfficiency should be >= 1.0, got %f", stats.MemoryEfficiency)
	}

	if stats.ThroughputGain < 1.0 {
		t.Errorf("ThroughputGain should be >= 1.0, got %f", stats.ThroughputGain)
	}
}

func TestBatchStatusResponseSerialization(t *testing.T) {
	status := BatchStatusResponse{
		BatchEnabled:      true,
		MaxBatchSize:      8,
		BatchTimeout:      Duration{Duration: 500 * time.Millisecond},
		CurrentBatches:    2,
		PendingRequests:   5,
		ProcessedBatches:  42,
		AverageEfficiency: 2.8,
		SupportedModels:   []string{"llama2", "codellama"},
	}

	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("Failed to marshal BatchStatusResponse: %v", err)
	}

	var unmarshaled BatchStatusResponse
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal BatchStatusResponse: %v", err)
	}

	if !unmarshaled.BatchEnabled {
		t.Error("Expected BatchEnabled to be true")
	}

	if unmarshaled.MaxBatchSize != 8 {
		t.Errorf("Expected MaxBatchSize 8, got %d", unmarshaled.MaxBatchSize)
	}

	if len(unmarshaled.SupportedModels) != 2 {
		t.Errorf("Expected 2 supported models, got %d", len(unmarshaled.SupportedModels))
	}
}

func TestBatchErrorResponseSerialization(t *testing.T) {
	errorResp := BatchErrorResponse{
		BatchId:        "batch-456",
		Error:          "timeout occurred",
		FailedRequests: []int{1, 3},
		PartialResults: []interface{}{
			map[string]string{"response": "partial response"},
		},
	}

	data, err := json.Marshal(errorResp)
	if err != nil {
		t.Fatalf("Failed to marshal BatchErrorResponse: %v", err)
	}

	var unmarshaled BatchErrorResponse
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal BatchErrorResponse: %v", err)
	}

	if unmarshaled.BatchId != "batch-456" {
		t.Errorf("Expected BatchId 'batch-456', got '%s'", unmarshaled.BatchId)
	}

	if unmarshaled.Error != "timeout occurred" {
		t.Errorf("Expected Error 'timeout occurred', got '%s'", unmarshaled.Error)
	}

	if len(unmarshaled.FailedRequests) != 2 {
		t.Errorf("Expected 2 failed requests, got %d", len(unmarshaled.FailedRequests))
	}
}

func TestBatchTimeoutSerialization(t *testing.T) {
	req := BatchGenerateRequest{
		Requests:     []GenerateRequest{{Model: "test", Prompt: "test"}},
		BatchTimeout: &Duration{Duration: 200 * time.Millisecond},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request with timeout: %v", err)
	}

	var unmarshaled BatchGenerateRequest
	err = json.Unmarshal(data, &unmarshaled)
	if err != nil {
		t.Fatalf("Failed to unmarshal request with timeout: %v", err)
	}

	if unmarshaled.BatchTimeout == nil {
		t.Fatal("Expected BatchTimeout to be non-nil")
	}

	if unmarshaled.BatchTimeout.Duration != 200*time.Millisecond {
		t.Errorf("Expected BatchTimeout 200ms, got %v", unmarshaled.BatchTimeout.Duration)
	}
}
