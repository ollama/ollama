package server

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

var (
	ErrResponseNotFound   = errors.New("response not found")
	ErrChainDepthExceeded = errors.New("response chain depth limit exceeded")
)

const (
	DefaultResponseTTL   = 30 * time.Minute
	DefaultMaxResponses  = 1024
	DefaultMaxChainDepth = 100
	ResponseGCInterval   = 60 * time.Second
)

// newResponseID generates an ID matching the OpenAI format: "resp_" + 32 hex chars.
func newResponseID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return "resp_" + hex.EncodeToString(b)
}

// StoredResponse is an immutable snapshot of a completed response.
type StoredResponse struct {
	ID                 string
	PreviousResponseID string
	Model              string
	Instructions       string
	InputMessages      []api.Message // messages from the request input
	OutputMessages     []api.Message // assistant response
	CreatedAt          time.Time
}

// ResponseStore manages stored responses for previous_response_id chaining.
// It is safe for concurrent use.
type ResponseStore struct {
	mu            sync.RWMutex
	responses     map[string]*StoredResponse
	maxResponses  int
	maxChainDepth int
	ttl           time.Duration
	stopGC        chan struct{}
}

// NewResponseStore creates a response store and starts background GC.
func NewResponseStore(maxResponses int, ttl time.Duration) *ResponseStore {
	if maxResponses <= 0 {
		maxResponses = DefaultMaxResponses
	}
	if ttl <= 0 {
		ttl = DefaultResponseTTL
	}

	rs := &ResponseStore{
		responses:     make(map[string]*StoredResponse),
		maxResponses:  maxResponses,
		maxChainDepth: DefaultMaxChainDepth,
		ttl:           ttl,
		stopGC:        make(chan struct{}),
	}
	go rs.gcLoop()
	return rs
}

// Store saves a response with the given parameters.
// This signature satisfies middleware.ResponseStoreInterface.
func (rs *ResponseStore) Store(id, previousID, model, instructions string, input, output []api.Message) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if len(rs.responses) >= rs.maxResponses {
		rs.evictOldest()
	}
	rs.responses[id] = &StoredResponse{
		ID:                 id,
		PreviousResponseID: previousID,
		Model:              model,
		Instructions:       instructions,
		InputMessages:      input,
		OutputMessages:     output,
		CreatedAt:          time.Now(),
	}
}

// storeResponse saves a StoredResponse directly. Used internally and in tests.
func (rs *ResponseStore) storeResponse(resp *StoredResponse) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	if len(rs.responses) >= rs.maxResponses {
		rs.evictOldest()
	}
	rs.responses[resp.ID] = resp
}

// Get retrieves a stored response by ID.
func (rs *ResponseStore) Get(id string) (*StoredResponse, error) {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	resp, ok := rs.responses[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrResponseNotFound, id)
	}
	return resp, nil
}

// BuildHistory walks the previous_response_id chain and reconstructs the full
// conversation history. Messages are returned in chronological order.
// The instructions from the chain are NOT included — the caller must supply
// their own instructions per the OpenAI spec.
func (rs *ResponseStore) BuildHistory(id string) ([]api.Message, error) {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	// Walk the chain backwards, collecting responses
	var chain []*StoredResponse
	current := id
	seen := make(map[string]bool)

	for current != "" {
		if seen[current] {
			return nil, fmt.Errorf("circular reference detected in response chain at %s", current)
		}
		if len(chain) >= rs.maxChainDepth {
			return nil, fmt.Errorf("%w: exceeded %d", ErrChainDepthExceeded, rs.maxChainDepth)
		}

		resp, ok := rs.responses[current]
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrResponseNotFound, current)
		}

		seen[current] = true
		chain = append(chain, resp)
		current = resp.PreviousResponseID
	}

	// Reverse to get chronological order
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}

	// Build message history: [input1, output1, input2, output2, ...]
	var messages []api.Message
	for _, resp := range chain {
		messages = append(messages, resp.InputMessages...)
		messages = append(messages, resp.OutputMessages...)
	}

	return messages, nil
}

// Delete removes a response by ID.
func (rs *ResponseStore) Delete(id string) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	delete(rs.responses, id)
}

// Len returns the number of stored responses.
func (rs *ResponseStore) Len() int {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	return len(rs.responses)
}

// evictOldest removes the response with the oldest CreatedAt timestamp.
// Must be called with rs.mu held.
func (rs *ResponseStore) evictOldest() {
	var oldestID string
	var oldestTime time.Time

	for id, resp := range rs.responses {
		if oldestID == "" || resp.CreatedAt.Before(oldestTime) {
			oldestID = id
			oldestTime = resp.CreatedAt
		}
	}

	if oldestID != "" {
		delete(rs.responses, oldestID)
	}
}

func (rs *ResponseStore) gcLoop() {
	ticker := time.NewTicker(ResponseGCInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			rs.gc()
		case <-rs.stopGC:
			return
		}
	}
}

func (rs *ResponseStore) gc() {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	cutoff := time.Now().Add(-rs.ttl)
	for id, resp := range rs.responses {
		if resp.CreatedAt.Before(cutoff) {
			delete(rs.responses, id)
		}
	}
}

// Stop stops the background GC goroutine.
func (rs *ResponseStore) Stop() {
	close(rs.stopGC)
}
