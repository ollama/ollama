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
		fmt.Fprint(w, `{"models":[{"name":"new-model","size":123,"details":{"context_length":65536,"projected_context_length":32768,"embedding_length":1024},"capabilities":["vision","tools"]}]}`)
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
	if got[0].ContextLength != 32_768 || got[0].EmbeddingLength != 1_024 {
		t.Fatalf("metadata = context %d embedding %d, want refreshed metadata", got[0].ContextLength, got[0].EmbeddingLength)
	}
	if !got[0].HasCapability(modelpkg.CapabilityVision) || !got[0].ToolCapable {
		t.Fatalf("capabilities = %v toolCapable=%v, want refreshed capabilities", got[0].Capabilities, got[0].ToolCapable)
	}
}

func TestLaunchModelFromListResponseContextLength(t *testing.T) {
	tests := []struct {
		name    string
		details api.ModelDetails
		want    int
	}{
		{
			name:    "projected",
			details: api.ModelDetails{ContextLength: 131_072, ProjectedContextLength: 32_768},
			want:    32_768,
		},
		{
			name:    "architectural fallback",
			details: api.ModelDetails{ContextLength: 131_072},
			want:    131_072,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := launchModelFromListResponse(api.ListModelResponse{Name: "test", Details: tt.details})
			if got.ContextLength != tt.want {
				t.Fatalf("ContextLength = %d, want %d", got.ContextLength, tt.want)
			}
		})
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
