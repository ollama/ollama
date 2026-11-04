package launch

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestSameModelRefLatestWithSource(t *testing.T) {
	for _, tt := range []struct {
		a, b string
		want bool
	}{
		{"ornith:latest:local", "ornith:latest", true},
		{"ornith:latest:local", "ornith:local", true},
		{"ornith:latest:cloud", "ornith:cloud", true},
		{"ornith:latest-cloud", "ornith:cloud", true},
		{" ORNITH:LATEST:LOCAL ", "ornith", true},
		{"ornith:latest:local", "ornith:cloud", false},
		{"ornith:latest:cloud", "ornith:local", false},
		{"ornith:8b:local", "ornith:latest", false},
		{"ornith:latest:local", "other:latest", false},
		{"ornith:cloud:local", "ornith:local", false},
	} {
		t.Run(tt.a+"/"+tt.b, func(t *testing.T) {
			if got := SameModelRef(tt.a, tt.b); got != tt.want {
				t.Errorf("SameModelRef(%q, %q) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
			if got := SameModelRef(tt.b, tt.a); got != tt.want {
				t.Errorf("SameModelRef(%q, %q) = %v, want %v", tt.b, tt.a, got, tt.want)
			}
		})
	}
}

func TestProcessContextWindowMatchesLatestWithSource(t *testing.T) {
	for _, field := range []string{"name", "model"} {
		t.Run(field, func(t *testing.T) {
			running := api.ProcessModelResponse{ContextLength: 8192}
			if field == "name" {
				running.Name = "ornith:latest"
			} else {
				running.Model = "ornith:latest"
			}
			got := processContextWindow("ornith:latest:local", &api.ProcessResponse{
				Models: []api.ProcessModelResponse{
					{Name: "other:latest", ContextLength: 32768}, running,
				},
			})
			if got != 8192 {
				t.Fatalf("context window = %d, want 8192", got)
			}
		})
	}
}

func TestLoadedContextWindowMatchesLatestWithSource(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ps" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"models":[{"name":"ornith:latest","context_length":8192}]}`)
	}))
	defer server.Close()
	base, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := api.NewClient(base, server.Client())
	if got := LoadedContextWindow(context.Background(), client, "ornith:latest:local"); got != 8192 {
		t.Fatalf("loaded context window = %d, want 8192", got)
	}
}
