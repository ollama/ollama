package launch

import (
	"context"
	"encoding/json"
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

func TestFallbackLaunchModelUsesServerContextLength(t *testing.T) {
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
		fmt.Fprint(w, `{"model_info":{"deepseek4.context_length":1048576}}`)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	t.Run("cloud model missing from limits", func(t *testing.T) {
		shown = nil
		got := fallbackLaunchModel("deepseek-v4-pro-inventory-test:cloud")
		if got.ContextLength != 1_048_576 {
			t.Fatalf("ContextLength = %d, want 1048576", got.ContextLength)
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

	t.Run("cloud model with known limits", func(t *testing.T) {
		shown = nil
		got := fallbackLaunchModel("kimi-k2.5:cloud")
		if got.ContextLength != 262_144 {
			t.Fatalf("ContextLength = %d, want 262144", got.ContextLength)
		}
		if len(shown) != 0 {
			t.Fatalf("show calls = %v, want none for a tabled model", shown)
		}
	})

	t.Run("local model", func(t *testing.T) {
		shown = nil
		got := fallbackLaunchModel("llama3.2")
		if got.ContextLength != 0 {
			t.Fatalf("ContextLength = %d, want 0", got.ContextLength)
		}
		if len(shown) != 0 {
			t.Fatalf("show calls = %v, want none for a local model", shown)
		}
	})

	t.Run("resolved cloud model without metadata", func(t *testing.T) {
		shown = nil
		models := []LaunchModel{{Name: "glm-5.2-inventory-test:cloud", Remote: true}}
		resolved, _ := resolveLaunchModels([]string{"glm-5.2-inventory-test:cloud"}, models)
		if len(resolved) != 1 || resolved[0].ContextLength != 1_048_576 {
			t.Fatalf("resolved = %#v, want ContextLength 1048576", resolved)
		}
	})
}
