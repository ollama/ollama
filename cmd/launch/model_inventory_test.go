package launch

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/ollama/ollama/api"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestModelInventoryResolveRefreshesLocalMiss(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		calls++
		if calls == 1 {
			fmt.Fprint(w, `{"models":[]}`)
			return
		}
		fmt.Fprint(w, `{"models":[{"name":"new-model","size":123,"details":{"context_length":65536,"embedding_length":1024},"capabilities":["vision","tools"]}]}`)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	inventory := newModelInventory(api.NewClient(u, srv.Client()))

	got := inventory.Resolve(context.Background(), []string{"new-model"})
	if calls != 2 {
		t.Fatalf("List calls = %d, want 2", calls)
	}
	if len(got) != 1 {
		t.Fatalf("Resolve returned %d models, want 1", len(got))
	}
	if got[0].Name != "new-model" {
		t.Fatalf("Name = %q, want new-model", got[0].Name)
	}
	if got[0].ContextLength != 65_536 || got[0].EmbeddingLength != 1_024 {
		t.Fatalf("metadata = context %d embedding %d, want refreshed metadata", got[0].ContextLength, got[0].EmbeddingLength)
	}
	if !got[0].HasCapability(modelpkg.CapabilityVision) || !got[0].ToolCapable {
		t.Fatalf("capabilities = %v toolCapable=%v, want refreshed capabilities", got[0].Capabilities, got[0].ToolCapable)
	}
}

func TestModelInventoryResolveDoesNotRefreshCloudMiss(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		calls++
		fmt.Fprint(w, `{"models":[]}`)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	inventory := newModelInventory(api.NewClient(u, srv.Client()))

	got := inventory.Resolve(context.Background(), []string{"glm-5.1:cloud"})
	if calls != 1 {
		t.Fatalf("List calls = %d, want 1", calls)
	}
	if len(got) != 1 {
		t.Fatalf("Resolve returned %d models, want 1", len(got))
	}
	if got[0].Name != "glm-5.1:cloud" || !got[0].Remote {
		t.Fatalf("resolved model = %#v, want cloud fallback", got[0])
	}
	if got[0].ContextLength <= 0 || got[0].MaxOutputTokens <= 0 {
		t.Fatalf("cloud limits not applied: %#v", got[0])
	}
}

func TestModelInventoryEnrichUnresolvedFromShow(t *testing.T) {
	showCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/show":
			showCalls++
			fmt.Fprint(w, `{"capabilities":["thinking","tools"],"model_info":{"kimi-k3.context_length":1048576,"kimi-k3.embedding_length":7168},"details":{"family":"kimi-k3"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	inventory := newModelInventory(api.NewClient(u, srv.Client()))

	resolved := inventory.Resolve(context.Background(), []string{"kimi-k3:cloud"})
	got := inventory.enrichUnresolvedFromShow(context.Background(), resolved)
	if len(got) != 1 {
		t.Fatalf("enrichUnresolvedFromShow returned %d models, want 1", len(got))
	}
	if showCalls != 1 {
		t.Fatalf("Show calls = %d, want 1", showCalls)
	}
	if got[0].ContextLength != 1_048_576 {
		t.Fatalf("ContextLength = %d, want 1048576", got[0].ContextLength)
	}
	if got[0].Details.ContextLength != 1_048_576 {
		t.Fatalf("Details.ContextLength = %d, want 1048576", got[0].Details.ContextLength)
	}
	if !got[0].HasCapability(modelpkg.CapabilityThinking) || !got[0].ToolCapable {
		t.Fatalf("capabilities = %v toolCapable=%v, want thinking and tools from Show", got[0].Capabilities, got[0].ToolCapable)
	}
	if !got[0].Remote {
		t.Fatal("Remote = false, want cloud fallback to stay remote")
	}
}

func TestModelInventoryEnrichKeepsFallbackWhenShowFails(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			fmt.Fprint(w, `{"models":[]}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	inventory := newModelInventory(api.NewClient(u, srv.Client()))

	resolved := inventory.Resolve(context.Background(), []string{"newcloud:cloud"})
	got := inventory.enrichUnresolvedFromShow(context.Background(), resolved)
	if len(got) != 1 {
		t.Fatalf("enrichUnresolvedFromShow returned %d models, want 1", len(got))
	}
	if got[0].ContextLength != 0 || len(got[0].Capabilities) != 0 {
		t.Fatalf("fallback metadata = %#v, want zero values when Show is unavailable", got[0])
	}
	if !got[0].Remote {
		t.Fatal("Remote = false, want fallback marked remote")
	}
}

func TestModelInventoryEnrichSkipsShowForListedModels(t *testing.T) {
	showCalls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"local:latest","details":{"context_length":32768},"capabilities":["completion"]}]}`)
		case "/api/show":
			showCalls++
			fmt.Fprint(w, `{}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	u, _ := url.Parse(srv.URL)
	inventory := newModelInventory(api.NewClient(u, srv.Client()))

	resolved := inventory.Resolve(context.Background(), []string{"local"})
	got := inventory.enrichUnresolvedFromShow(context.Background(), resolved)
	if len(got) != 1 || got[0].ContextLength != 32_768 {
		t.Fatalf("resolved = %#v, want local model metadata from the list", got)
	}
	if showCalls != 0 {
		t.Fatalf("Show calls = %d, want 0 for models resolved from the list", showCalls)
	}
}
