package server

import (
	"context"
	"os"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestModelRequestStore(t *testing.T) {
	// Create a temporary store for testing
	store, err := NewModelRequestStore()
	if err != nil {
		t.Fatalf("Failed to create model request store: %v", err)
	}

	// Test creating a model request
	req := &api.ModelRequestCreateRequest{
		Model:       "test-model:1.0",
		Description: "A test model for Ollama Cloud",
		Reason:      "Good for testing",
	}

	ctx := context.Background()
	modelReq, err := store.CreateRequest(ctx, req)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	if modelReq.Model != req.Model {
		t.Errorf("Expected model %s, got %s", req.Model, modelReq.Model)
	}

	if modelReq.Status != "pending" {
		t.Errorf("Expected status 'pending', got %s", modelReq.Status)
	}

	if modelReq.VoteCount != 1 {
		t.Errorf("Expected vote count 1, got %d", modelReq.VoteCount)
	}

	// Test getting the request
	retrieved, err := store.GetRequest(ctx, modelReq.ID)
	if err != nil {
		t.Fatalf("Failed to get request: %v", err)
	}

	if retrieved.ID != modelReq.ID {
		t.Errorf("Expected ID %s, got %s", modelReq.ID, retrieved.ID)
	}

	// Test listing requests
	requests := store.ListRequests(ctx)
	if len(requests) != 1 {
		t.Errorf("Expected 1 request, got %d", len(requests))
	}

	// Test voting for a request
	voted, err := store.VoteRequest(ctx, modelReq.ID)
	if err != nil {
		t.Fatalf("Failed to vote on request: %v", err)
	}

	if voted.VoteCount != 2 {
		t.Errorf("Expected vote count 2 after voting, got %d", voted.VoteCount)
	}

	// Test invalid model request
	invalidReq := &api.ModelRequestCreateRequest{
		Model: "", // Empty model name
	}

	_, err = store.CreateRequest(ctx, invalidReq)
	if err == nil {
		t.Error("Expected error for empty model name, got nil")
	}

	// Cleanup
	path, _ := modelRequestsStorePath()
	os.Remove(path)
}

func TestModelRequestValidation(t *testing.T) {
	ctx := context.Background()
	store, err := NewModelRequestStore()
	if err != nil {
		t.Fatalf("Failed to create model request store: %v", err)
	}

	tests := []struct {
		name      string
		request   *api.ModelRequestCreateRequest
		shouldErr bool
	}{
		{
			name: "valid request",
			request: &api.ModelRequestCreateRequest{
				Model:       "llama-3:70b",
				Description: "Large language model",
				Reason:      "For production use",
			},
			shouldErr: false,
		},
		{
			name: "empty model name",
			request: &api.ModelRequestCreateRequest{
				Model: "",
			},
			shouldErr: true,
		},
		{
			name: "model with description only",
			request: &api.ModelRequestCreateRequest{
				Model:       "mistral-7b",
				Description: "Fast model",
			},
			shouldErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := store.CreateRequest(ctx, tt.request)
			if (err != nil) != tt.shouldErr {
				t.Errorf("expected error: %v, got: %v", tt.shouldErr, err)
			}
		})
	}

	// Cleanup
	path, _ := modelRequestsStorePath()
	os.Remove(path)
}
