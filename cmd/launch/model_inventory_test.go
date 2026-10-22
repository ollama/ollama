package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

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

func TestFallbackLaunchModelUsesServerMetadata(t *testing.T) {
	useServerContextLookup(t)

	var shown []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			http.NotFound(w, r)
			return
		}
		var req api.ShowRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode show request: %v", err)
		}
		shown = append(shown, req.Model)
		fmt.Fprint(w, `{"capabilities":["vision","thinking","completion","tools"],"thinking":{"values":[false,"low","high"],"default":"high"},"model_info":{"deepseek4.context_length":1048576}}`)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	t.Run("cloud model missing from limits", func(t *testing.T) {
		shown = nil
		got := fallbackLaunchModel("deepseek-v4-pro-inventory-test:cloud")
		if got.ContextLength != 1_048_576 {
			t.Fatalf("ContextLength = %d, want 1048576", got.ContextLength)
		}
		if !got.ToolCapable || !got.HasCapability(modelpkg.CapabilityVision) || !got.HasCapability(modelpkg.CapabilityThinking) {
			t.Fatalf("capabilities = %v, want vision, thinking and tools", got.Capabilities)
		}
		if got.Thinking == nil || got.Thinking.Default != "high" || len(got.Thinking.Values) != 3 {
			t.Fatalf("Thinking = %#v, want the server's descriptor", got.Thinking)
		}
		if len(shown) != 1 || shown[0] != "deepseek-v4-pro-inventory-test:cloud" {
			t.Fatalf("show calls = %v, want one for the model", shown)
		}

		shown = nil
		if got := fallbackLaunchModel("deepseek-v4-pro-inventory-test:cloud"); got.ContextLength != 1_048_576 {
			t.Fatalf("cached ContextLength = %d, want 1048576", got.ContextLength)
		}
		if len(shown) != 0 {
			t.Fatalf("show calls = %v, want none when cached", shown)
		}
	})

	t.Run("cloud model with known limits keeps the table context", func(t *testing.T) {
		shown = nil
		got := fallbackLaunchModel("kimi-k2.5:cloud")
		if got.ContextLength != 262_144 || got.MaxOutputTokens != 262_144 {
			t.Fatalf("limits = %d/%d, want the table's 262144/262144", got.ContextLength, got.MaxOutputTokens)
		}
		if !got.HasCapability(modelpkg.CapabilityVision) {
			t.Fatalf("capabilities = %v, want the server's capabilities", got.Capabilities)
		}
	})

	t.Run("local model", func(t *testing.T) {
		shown = nil
		got := fallbackLaunchModel("llama3.2")
		if got.ContextLength != 0 || len(got.Capabilities) != 0 {
			t.Fatalf("model = %#v, want no server metadata", got)
		}
		if len(shown) != 0 {
			t.Fatalf("show calls = %v, want none for a local model", shown)
		}
	})

	t.Run("resolved cloud model without metadata", func(t *testing.T) {
		shown = nil
		models := []LaunchModel{{Name: "glm-5.2-inventory-test:cloud", Remote: true}}
		resolved, _ := resolveLaunchModels([]string{"glm-5.2-inventory-test:cloud"}, models)
		if len(resolved) != 1 || resolved[0].ContextLength != 1_048_576 || !resolved[0].ToolCapable {
			t.Fatalf("resolved = %#v, want ContextLength 1048576 and tool capability", resolved)
		}
	})
}

func TestServerModelMetadataStopsAfterUnreachableServer(t *testing.T) {
	useServerContextLookup(t)

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		time.Sleep(300 * time.Millisecond)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	oldTimeout := modelShowTimeout
	modelShowTimeout = 50 * time.Millisecond
	t.Cleanup(func() { modelShowTimeout = oldTimeout })

	for _, name := range []string{"unreachable-a:cloud", "unreachable-b:cloud", "unreachable-a:cloud"} {
		if _, ok := cachedServerModelMetadata(name); ok {
			t.Fatalf("lookup for %s succeeded against a hanging server", name)
		}
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("show requests = %d, want 1 before the launcher stops asking", got)
	}
}
