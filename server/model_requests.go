package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/api"
)

// ModelRequestStore manages model requests for Ollama Cloud.
type ModelRequestStore struct {
	mu       sync.RWMutex
	requests map[string]*api.ModelRequest
	filepath string
}

// NewModelRequestStore creates a new model request store.
func NewModelRequestStore() (*ModelRequestStore, error) {
	path, err := modelRequestsStorePath()
	if err != nil {
		return nil, err
	}

	store := &ModelRequestStore{
		requests: make(map[string]*api.ModelRequest),
		filepath: path,
	}

	// Load existing requests from disk
	if err := store.load(); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Warn("failed to load model requests", "error", err)
		}
	}

	return store, nil
}

// CreateRequest creates a new model request.
func (s *ModelRequestStore) CreateRequest(ctx context.Context, req *api.ModelRequestCreateRequest) (*api.ModelRequest, error) {
	if req.Model == "" {
		return nil, errors.New("model name is required")
	}

	now := time.Now()
	modelReq := &api.ModelRequest{
		ID:        uuid.New().String(),
		Model:     req.Model,
		Description: req.Description,
		Reason:    req.Reason,
		Status:    "pending",
		CreatedAt: now,
		UpdatedAt: now,
		VoteCount: 1,
	}

	s.mu.Lock()
	s.requests[modelReq.ID] = modelReq
	s.mu.Unlock()

	if err := s.persist(); err != nil {
		slog.Warn("failed to persist model request", "error", err)
	}

	slog.Info("model request created", "id", modelReq.ID, "model", modelReq.Model)
	return modelReq, nil
}

// GetRequest retrieves a model request by ID.
func (s *ModelRequestStore) GetRequest(ctx context.Context, id string) (*api.ModelRequest, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	req, ok := s.requests[id]
	if !ok {
		return nil, fmt.Errorf("model request not found: %s", id)
	}

	return req, nil
}

// ListRequests returns all model requests.
func (s *ModelRequestStore) ListRequests(ctx context.Context) []api.ModelRequest {
	s.mu.RLock()
	defer s.mu.RUnlock()

	requests := make([]api.ModelRequest, 0, len(s.requests))
	for _, req := range s.requests {
		requests = append(requests, *req)
	}

	return requests
}

// VoteRequest increments the vote count for a model request.
func (s *ModelRequestStore) VoteRequest(ctx context.Context, id string) (*api.ModelRequest, error) {
	s.mu.Lock()
	req, ok := s.requests[id]
	if !ok {
		s.mu.Unlock()
		return nil, fmt.Errorf("model request not found: %s", id)
	}

	req.VoteCount++
	req.UpdatedAt = time.Now()
	s.mu.Unlock()

	if err := s.persist(); err != nil {
		slog.Warn("failed to persist model request vote", "error", err)
	}

	return req, nil
}

// load loads model requests from disk.
func (s *ModelRequestStore) load() error {
	data, err := os.ReadFile(s.filepath)
	if err != nil {
		return err
	}

	var response api.ModelRequestsResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for _, req := range response.Requests {
		r := req
		s.requests[req.ID] = &r
	}

	slog.Debug("loaded model requests", "path", s.filepath, "count", len(s.requests))
	return nil
}

// persist saves model requests to disk.
func (s *ModelRequestStore) persist() error {
	s.mu.RLock()
	requests := make([]api.ModelRequest, 0, len(s.requests))
	for _, req := range s.requests {
		requests = append(requests, *req)
	}
	s.mu.RUnlock()

	response := api.ModelRequestsResponse{Requests: requests}
	data, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(s.filepath), 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(filepath.Dir(s.filepath), ".model-requests-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	if err := os.Rename(tmpPath, s.filepath); err != nil {
		return err
	}

	slog.Debug("persisted model requests", "path", s.filepath, "count", len(requests))
	return nil
}

// modelRequestsStorePath returns the path to the model requests store.
func modelRequestsStorePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "cache", "model-requests.json"), nil
}
