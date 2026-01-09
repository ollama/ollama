package usage

import (
	"testing"
)

func TestNew(t *testing.T) {
	stats := New()
	if stats == nil {
		t.Fatal("New() returned nil")
	}
}

func TestRecord(t *testing.T) {
	stats := New()

	stats.Record(&Request{
		Model:            "llama3:8b",
		Endpoint:         "chat",
		Architecture:     "llama",
		APIType:          "native",
		PromptTokens:     100,
		CompletionTokens: 50,
		UsedTools:        true,
		StructuredOutput: false,
	})

	// Check totals
	payload := stats.View()
	if payload.Totals.Requests != 1 {
		t.Errorf("expected 1 request, got %d", payload.Totals.Requests)
	}
	if payload.Totals.InputTokens != 100 {
		t.Errorf("expected 100 prompt tokens, got %d", payload.Totals.InputTokens)
	}
	if payload.Totals.OutputTokens != 50 {
		t.Errorf("expected 50 completion tokens, got %d", payload.Totals.OutputTokens)
	}
	if payload.Features.ToolCalls != 1 {
		t.Errorf("expected 1 tool call, got %d", payload.Features.ToolCalls)
	}
	if payload.Features.StructuredOutput != 0 {
		t.Errorf("expected 0 structured outputs, got %d", payload.Features.StructuredOutput)
	}
}

func TestGetModelStats(t *testing.T) {
	stats := New()

	// Record requests for multiple models
	stats.Record(&Request{
		Model:            "llama3:8b",
		PromptTokens:     100,
		CompletionTokens: 50,
	})
	stats.Record(&Request{
		Model:            "llama3:8b",
		PromptTokens:     200,
		CompletionTokens: 100,
	})
	stats.Record(&Request{
		Model:            "mistral:7b",
		PromptTokens:     50,
		CompletionTokens: 25,
	})

	modelStats := stats.GetModelStats()

	// Check llama3:8b stats
	llama := modelStats["llama3:8b"]
	if llama == nil {
		t.Fatal("expected llama3:8b stats")
	}
	if llama.Requests != 2 {
		t.Errorf("expected 2 requests for llama3:8b, got %d", llama.Requests)
	}
	if llama.TokensInput != 300 {
		t.Errorf("expected 300 input tokens for llama3:8b, got %d", llama.TokensInput)
	}
	if llama.TokensOutput != 150 {
		t.Errorf("expected 150 output tokens for llama3:8b, got %d", llama.TokensOutput)
	}

	// Check mistral:7b stats
	mistral := modelStats["mistral:7b"]
	if mistral == nil {
		t.Fatal("expected mistral:7b stats")
	}
	if mistral.Requests != 1 {
		t.Errorf("expected 1 request for mistral:7b, got %d", mistral.Requests)
	}
	if mistral.TokensInput != 50 {
		t.Errorf("expected 50 input tokens for mistral:7b, got %d", mistral.TokensInput)
	}
	if mistral.TokensOutput != 25 {
		t.Errorf("expected 25 output tokens for mistral:7b, got %d", mistral.TokensOutput)
	}
}

func TestRecordError(t *testing.T) {
	stats := New()

	stats.RecordError()
	stats.RecordError()

	payload := stats.View()
	if payload.Totals.Errors != 2 {
		t.Errorf("expected 2 errors, got %d", payload.Totals.Errors)
	}
}

func TestView(t *testing.T) {
	stats := New()

	stats.Record(&Request{
		Model:        "llama3:8b",
		Endpoint:     "chat",
		Architecture: "llama",
		APIType:      "native",
	})

	// First view
	_ = stats.View()

	// View should not reset counters
	payload := stats.View()
	if payload.Totals.Requests != 1 {
		t.Errorf("View should not reset counters, expected 1 request, got %d", payload.Totals.Requests)
	}
}

func TestSnapshot(t *testing.T) {
	stats := New()

	stats.Record(&Request{
		Model:            "llama3:8b",
		Endpoint:         "chat",
		PromptTokens:     100,
		CompletionTokens: 50,
	})

	// Snapshot should return data and reset counters
	snapshot := stats.Snapshot()
	if snapshot.Totals.Requests != 1 {
		t.Errorf("expected 1 request in snapshot, got %d", snapshot.Totals.Requests)
	}

	// After snapshot, counters should be reset
	payload2 := stats.View()
	if payload2.Totals.Requests != 0 {
		t.Errorf("expected 0 requests after snapshot, got %d", payload2.Totals.Requests)
	}
}

func TestConcurrentAccess(t *testing.T) {
	stats := New()

	done := make(chan bool)

	// Concurrent writes
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				stats.Record(&Request{
					Model:            "llama3:8b",
					PromptTokens:     10,
					CompletionTokens: 5,
				})
			}
			done <- true
		}()
	}

	// Concurrent reads
	for i := 0; i < 5; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				_ = stats.View()
				_ = stats.GetModelStats()
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 15; i++ {
		<-done
	}

	payload := stats.View()
	if payload.Totals.Requests != 1000 {
		t.Errorf("expected 1000 requests, got %d", payload.Totals.Requests)
	}
}
