package server

import (
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestNewResponseID(t *testing.T) {
	id := newResponseID()
	if !strings.HasPrefix(id, "resp_") {
		t.Fatalf("expected resp_ prefix, got %s", id)
	}
	// "resp_" (5) + 32 hex chars = 37 total
	if len(id) != 37 {
		t.Fatalf("expected 37 chars, got %d: %s", len(id), id)
	}

	// Uniqueness
	ids := make(map[string]bool)
	for range 10000 {
		rid := newResponseID()
		if ids[rid] {
			t.Fatalf("duplicate ID: %s", rid)
		}
		ids[rid] = true
	}
}

func TestResponseStoreStoreAndGet(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	resp := &StoredResponse{
		ID:             "resp_test123",
		Model:          "gemma3:4b",
		InputMessages:  []api.Message{{Role: "user", Content: "hello"}},
		OutputMessages: []api.Message{{Role: "assistant", Content: "hi"}},
		CreatedAt:      time.Now(),
	}

	rs.storeResponse(resp)

	got, err := rs.Get("resp_test123")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "resp_test123" {
		t.Fatalf("expected resp_test123, got %s", got.ID)
	}
	if got.Model != "gemma3:4b" {
		t.Fatalf("expected gemma3:4b, got %s", got.Model)
	}
}

func TestResponseStoreGetNotFound(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	_, err := rs.Get("resp_nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent response")
	}
	if !strings.Contains(err.Error(), "response not found") {
		t.Fatalf("expected 'response not found' error, got: %v", err)
	}
}

func TestResponseStoreBuildHistorySingleLink(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	rs.storeResponse(&StoredResponse{
		ID:             "resp_a",
		InputMessages:  []api.Message{{Role: "user", Content: "What is 2+2?"}},
		OutputMessages: []api.Message{{Role: "assistant", Content: "4"}},
		CreatedAt:      time.Now(),
	})

	rs.storeResponse(&StoredResponse{
		ID:                 "resp_b",
		PreviousResponseID: "resp_a",
		InputMessages:      []api.Message{{Role: "user", Content: "And 3+3?"}},
		OutputMessages:     []api.Message{{Role: "assistant", Content: "6"}},
		CreatedAt:          time.Now(),
	})

	msgs, err := rs.BuildHistory("resp_b")
	if err != nil {
		t.Fatal(err)
	}

	// Should have: user(2+2) + assistant(4) + user(3+3) + assistant(6)
	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(msgs))
	}
	if msgs[0].Content != "What is 2+2?" {
		t.Fatalf("expected first message 'What is 2+2?', got %q", msgs[0].Content)
	}
	if msgs[1].Content != "4" {
		t.Fatalf("expected second message '4', got %q", msgs[1].Content)
	}
	if msgs[2].Content != "And 3+3?" {
		t.Fatalf("expected third message 'And 3+3?', got %q", msgs[2].Content)
	}
	if msgs[3].Content != "6" {
		t.Fatalf("expected fourth message '6', got %q", msgs[3].Content)
	}
}

func TestResponseStoreBuildHistoryMultipleLinks(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	rs.storeResponse(&StoredResponse{
		ID:             "resp_1",
		InputMessages:  []api.Message{{Role: "user", Content: "A"}},
		OutputMessages: []api.Message{{Role: "assistant", Content: "a"}},
		CreatedAt:      time.Now(),
	})
	rs.storeResponse(&StoredResponse{
		ID:                 "resp_2",
		PreviousResponseID: "resp_1",
		InputMessages:      []api.Message{{Role: "user", Content: "B"}},
		OutputMessages:     []api.Message{{Role: "assistant", Content: "b"}},
		CreatedAt:          time.Now(),
	})
	rs.storeResponse(&StoredResponse{
		ID:                 "resp_3",
		PreviousResponseID: "resp_2",
		InputMessages:      []api.Message{{Role: "user", Content: "C"}},
		OutputMessages:     []api.Message{{Role: "assistant", Content: "c"}},
		CreatedAt:          time.Now(),
	})

	msgs, err := rs.BuildHistory("resp_3")
	if err != nil {
		t.Fatal(err)
	}

	if len(msgs) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(msgs))
	}
	expected := []string{"A", "a", "B", "b", "C", "c"}
	for i, msg := range msgs {
		if msg.Content != expected[i] {
			t.Errorf("message %d: expected %q, got %q", i, expected[i], msg.Content)
		}
	}
}

func TestResponseStoreBuildHistoryNotFound(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	_, err := rs.BuildHistory("resp_doesnotexist")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "response not found") {
		t.Fatalf("expected 'response not found', got: %v", err)
	}
}

func TestResponseStoreBuildHistoryCircularReference(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	rs.storeResponse(&StoredResponse{
		ID:                 "resp_x",
		PreviousResponseID: "resp_y",
		InputMessages:      []api.Message{{Role: "user", Content: "X"}},
		OutputMessages:     []api.Message{{Role: "assistant", Content: "x"}},
		CreatedAt:          time.Now(),
	})
	rs.storeResponse(&StoredResponse{
		ID:                 "resp_y",
		PreviousResponseID: "resp_x",
		InputMessages:      []api.Message{{Role: "user", Content: "Y"}},
		OutputMessages:     []api.Message{{Role: "assistant", Content: "y"}},
		CreatedAt:          time.Now(),
	})

	_, err := rs.BuildHistory("resp_x")
	if err == nil {
		t.Fatal("expected circular reference error")
	}
	if !strings.Contains(err.Error(), "circular reference") {
		t.Fatalf("expected 'circular reference' error, got: %v", err)
	}
}

func TestResponseStoreBuildHistoryDepthLimit(t *testing.T) {
	rs := NewResponseStore(200, time.Hour)
	defer rs.Stop()

	// Create a chain of 101 responses (exceeds default limit of 100)
	prev := ""
	for i := range 101 {
		id := newResponseID()
		rs.storeResponse(&StoredResponse{
			ID:                 id,
			PreviousResponseID: prev,
			InputMessages:      []api.Message{{Role: "user", Content: "msg"}},
			OutputMessages:     []api.Message{{Role: "assistant", Content: "reply"}},
			CreatedAt:          time.Now(),
		})
		prev = id
		_ = i
	}

	_, err := rs.BuildHistory(prev)
	if err == nil {
		t.Fatal("expected depth limit error")
	}
	if !strings.Contains(err.Error(), "chain depth limit exceeded") {
		t.Fatalf("expected 'chain depth limit exceeded', got: %v", err)
	}
}

func TestResponseStoreTTLExpiry(t *testing.T) {
	rs := NewResponseStore(100, 100*time.Millisecond)
	defer rs.Stop()

	rs.storeResponse(&StoredResponse{
		ID:        "resp_old",
		CreatedAt: time.Now(),
	})

	time.Sleep(200 * time.Millisecond)
	rs.gc() // Force GC

	_, err := rs.Get("resp_old")
	if err == nil {
		t.Fatal("expected response to be expired")
	}
}

func TestResponseStoreMaxEntries(t *testing.T) {
	rs := NewResponseStore(3, time.Hour)
	defer rs.Stop()

	for i := range 5 {
		rs.storeResponse(&StoredResponse{
			ID:        newResponseID(),
			CreatedAt: time.Now().Add(time.Duration(i) * time.Second),
		})
	}

	if rs.Len() != 3 {
		t.Fatalf("expected 3 responses after eviction, got %d", rs.Len())
	}
}

func TestResponseStoreConcurrentAccess(t *testing.T) {
	rs := NewResponseStore(10000, time.Hour)
	defer rs.Stop()

	var wg sync.WaitGroup

	// Concurrent stores
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				rs.storeResponse(&StoredResponse{
					ID:             newResponseID(),
					InputMessages:  []api.Message{{Role: "user", Content: "concurrent"}},
					OutputMessages: []api.Message{{Role: "assistant", Content: "reply"}},
					CreatedAt:      time.Now(),
				})
			}
		}()
	}

	// Concurrent reads
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				rs.Len()
			}
		}()
	}

	wg.Wait()

	if rs.Len() == 0 {
		t.Fatal("expected responses after concurrent writes")
	}
}

func TestResponseStoreDelete(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	rs.storeResponse(&StoredResponse{ID: "resp_del", CreatedAt: time.Now()})
	rs.Delete("resp_del")

	_, err := rs.Get("resp_del")
	if err == nil {
		t.Fatal("expected not found after delete")
	}
}

func TestResponseStoreInstructionsNotInherited(t *testing.T) {
	rs := NewResponseStore(100, time.Hour)
	defer rs.Stop()

	// Response A has instructions embedded in input (as system message)
	rs.storeResponse(&StoredResponse{
		ID: "resp_with_instructions",
		InputMessages: []api.Message{
			{Role: "system", Content: "You are a pirate."},
			{Role: "user", Content: "Hello"},
		},
		OutputMessages: []api.Message{{Role: "assistant", Content: "Ahoy!"}},
		Instructions:   "You are a pirate.",
		CreatedAt:      time.Now(),
	})

	// BuildHistory returns messages but NOT the instructions field
	msgs, err := rs.BuildHistory("resp_with_instructions")
	if err != nil {
		t.Fatal(err)
	}

	// The system message IS in the history (it was part of input)
	// But the caller is responsible for adding their own instructions
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" {
		t.Fatalf("expected system message first, got %s", msgs[0].Role)
	}
}
