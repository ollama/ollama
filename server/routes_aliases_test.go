package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestAliasShadowingRejected(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("HOME", t.TempDir())

	s := Server{}
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "shadowed-model",
		RemoteHost: "example.com",
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	w = createRequest(t, s.CreateAliasHandler, aliasEntry{Alias: "shadowed-model", Target: "other-model"})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
}

func TestAliasResolvesForChatRemote(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("HOME", t.TempDir())

	var remoteModel string
	rs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req api.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		remoteModel = req.Model

		w.Header().Set("Content-Type", "application/json")
		resp := api.ChatResponse{
			Model:      req.Model,
			Done:       true,
			DoneReason: "load",
		}
		if err := json.NewEncoder(w).Encode(&resp); err != nil {
			t.Fatal(err)
		}
	}))
	defer rs.Close()

	p, err := url.Parse(rs.URL)
	if err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_REMOTES", p.Hostname())

	s := Server{}
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "target-model",
		RemoteHost: rs.URL,
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	w = createRequest(t, s.CreateAliasHandler, aliasEntry{Alias: "alias-model", Target: "target-model"})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	w = createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "alias-model",
		Messages: []api.Message{{Role: "user", Content: "hi"}},
		Stream:   &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp api.ChatResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if resp.Model != "alias-model" {
		t.Fatalf("expected response model to be alias-model, got %q", resp.Model)
	}

	if remoteModel != "test" {
		t.Fatalf("expected remote model to be 'test', got %q", remoteModel)
	}
}

func TestPrefixAliasBasicMatching(t *testing.T) {
	tmpDir := t.TempDir()
	store, err := createStore(filepath.Join(tmpDir, "server.json"))
	if err != nil {
		t.Fatal(err)
	}

	// Create a prefix alias: "myprefix-" -> "targetmodel"
	targetName := model.ParseName("targetmodel")

	// Set a prefix alias (using "myprefix-" as the pattern)
	store.mu.Lock()
	store.prefixEntries = append(store.prefixEntries, aliasEntry{
		Alias:          "myprefix-",
		Target:         "targetmodel",
		PrefixMatching: true,
	})
	store.mu.Unlock()

	// Test that "myprefix-foo" resolves to "targetmodel"
	testName := model.ParseName("myprefix-foo")
	resolved, wasResolved, err := store.ResolveName(testName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved")
	}
	if resolved.DisplayShortest() != targetName.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q, got %q", targetName.DisplayShortest(), resolved.DisplayShortest())
	}

	// Test that "otherprefix-foo" does not resolve
	otherName := model.ParseName("otherprefix-foo")
	_, wasResolved, err = store.ResolveName(otherName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if wasResolved {
		t.Fatal("expected name not to be resolved")
	}

	// Test that exact alias takes precedence
	exactAlias := model.ParseName("myprefix-exact")
	exactTarget := model.ParseName("exacttarget")
	if err := store.Set(exactAlias, exactTarget, false); err != nil {
		t.Fatal(err)
	}

	resolved, wasResolved, err = store.ResolveName(exactAlias)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved")
	}
	if resolved.DisplayShortest() != exactTarget.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q (exact match), got %q", exactTarget.DisplayShortest(), resolved.DisplayShortest())
	}
}

func TestPrefixAliasLongestMatchWins(t *testing.T) {
	tmpDir := t.TempDir()
	store, err := createStore(filepath.Join(tmpDir, "server.json"))
	if err != nil {
		t.Fatal(err)
	}

	// Add two prefix aliases with overlapping patterns
	store.mu.Lock()
	store.prefixEntries = []aliasEntry{
		{Alias: "abc-", Target: "short-target", PrefixMatching: true},
		{Alias: "abc-def-", Target: "long-target", PrefixMatching: true},
	}
	store.sortPrefixEntriesLocked()
	store.mu.Unlock()

	// "abc-def-ghi" should match the longer prefix "abc-def-"
	testName := model.ParseName("abc-def-ghi")
	resolved, wasResolved, err := store.ResolveName(testName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved")
	}
	expectedLongTarget := model.ParseName("long-target")
	if resolved.DisplayShortest() != expectedLongTarget.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q (longest prefix match), got %q", expectedLongTarget.DisplayShortest(), resolved.DisplayShortest())
	}

	// "abc-xyz" should match the shorter prefix "abc-"
	testName2 := model.ParseName("abc-xyz")
	resolved, wasResolved, err = store.ResolveName(testName2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved")
	}
	expectedShortTarget := model.ParseName("short-target")
	if resolved.DisplayShortest() != expectedShortTarget.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q, got %q", expectedShortTarget.DisplayShortest(), resolved.DisplayShortest())
	}
}

func TestPrefixAliasChain(t *testing.T) {
	tmpDir := t.TempDir()
	store, err := createStore(filepath.Join(tmpDir, "server.json"))
	if err != nil {
		t.Fatal(err)
	}

	// Create a chain: prefix "test-" -> "intermediate" -> "final"
	intermediate := model.ParseName("intermediate")
	final := model.ParseName("final")

	// Add prefix alias
	store.mu.Lock()
	store.prefixEntries = []aliasEntry{
		{Alias: "test-", Target: "intermediate", PrefixMatching: true},
	}
	store.mu.Unlock()

	// Add exact alias for the intermediate step
	if err := store.Set(intermediate, final, false); err != nil {
		t.Fatal(err)
	}

	// "test-foo" should resolve through the chain to "final"
	testName := model.ParseName("test-foo")
	resolved, wasResolved, err := store.ResolveName(testName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved")
	}
	if resolved.DisplayShortest() != final.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q, got %q", final.DisplayShortest(), resolved.DisplayShortest())
	}
}

func TestPrefixAliasCRUD(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("HOME", t.TempDir())

	s := Server{}

	// Create a prefix alias via API
	w := createRequest(t, s.CreateAliasHandler, aliasEntry{
		Alias:          "myprefix-",
		Target:         "llama2",
		PrefixMatching: true,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var createResp aliasEntry
	if err := json.NewDecoder(w.Body).Decode(&createResp); err != nil {
		t.Fatal(err)
	}
	if !createResp.PrefixMatching {
		t.Fatal("expected prefix_matching to be true in response")
	}

	// List aliases and verify the prefix alias is included
	w = createRequest(t, s.ListAliasesHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var listResp aliasListResponse
	if err := json.NewDecoder(w.Body).Decode(&listResp); err != nil {
		t.Fatal(err)
	}

	found := false
	for _, a := range listResp.Aliases {
		if a.PrefixMatching && a.Target == "llama2" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected to find prefix alias in list")
	}

	// Delete the prefix alias
	w = createRequest(t, s.DeleteAliasHandler, aliasDeleteRequest{Alias: "myprefix-"})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify it's deleted
	w = createRequest(t, s.ListAliasesHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	if err := json.NewDecoder(w.Body).Decode(&listResp); err != nil {
		t.Fatal(err)
	}

	for _, a := range listResp.Aliases {
		if a.PrefixMatching {
			t.Fatal("expected prefix alias to be deleted")
		}
	}
}

func TestPrefixAliasCaseInsensitive(t *testing.T) {
	tmpDir := t.TempDir()
	store, err := createStore(filepath.Join(tmpDir, "server.json"))
	if err != nil {
		t.Fatal(err)
	}

	// Add a prefix alias with mixed case
	store.mu.Lock()
	store.prefixEntries = []aliasEntry{
		{Alias: "MyPrefix-", Target: "targetmodel", PrefixMatching: true},
	}
	store.mu.Unlock()

	// Test that matching is case-insensitive
	testName := model.ParseName("myprefix-foo")
	resolved, wasResolved, err := store.ResolveName(testName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved (case-insensitive)")
	}
	expectedTarget := model.ParseName("targetmodel")
	if resolved.DisplayShortest() != expectedTarget.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q, got %q", expectedTarget.DisplayShortest(), resolved.DisplayShortest())
	}

	// Test uppercase request
	testName2 := model.ParseName("MYPREFIX-BAR")
	_, wasResolved, err = store.ResolveName(testName2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected name to be resolved (uppercase)")
	}
}

func TestPrefixAliasLocalModelPrecedence(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("HOME", t.TempDir())

	s := Server{}

	// Create a local model that would match a prefix alias
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:      "myprefix-localmodel",
		RemoteHost: "example.com",
		From:       "test",
		Info: map[string]any{
			"capabilities": []string{"completion"},
		},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Create a prefix alias that would match the local model name
	w = createRequest(t, s.CreateAliasHandler, aliasEntry{
		Alias:          "myprefix-",
		Target:         "someothermodel",
		PrefixMatching: true,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify that resolving "myprefix-localmodel" returns the local model, not the alias target
	store, err := s.aliasStore()
	if err != nil {
		t.Fatal(err)
	}

	localModelName := model.ParseName("myprefix-localmodel")
	resolved, wasResolved, err := store.ResolveName(localModelName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if wasResolved {
		t.Fatalf("expected local model to take precedence (wasResolved should be false), but got resolved to %q", resolved.DisplayShortest())
	}
	if resolved.DisplayShortest() != localModelName.DisplayShortest() {
		t.Fatalf("expected resolved name to be local model %q, got %q", localModelName.DisplayShortest(), resolved.DisplayShortest())
	}

	// Also verify that a non-local model matching the prefix DOES resolve to the alias target
	nonLocalName := model.ParseName("myprefix-nonexistent")
	resolved, wasResolved, err = store.ResolveName(nonLocalName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wasResolved {
		t.Fatal("expected non-local model to resolve via prefix alias")
	}
	expectedTarget := model.ParseName("someothermodel")
	if resolved.DisplayShortest() != expectedTarget.DisplayShortest() {
		t.Fatalf("expected resolved name to be %q, got %q", expectedTarget.DisplayShortest(), resolved.DisplayShortest())
	}
}
