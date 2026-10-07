package launch

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestProcessContextWindowMatchesLatestAlias(t *testing.T) {
	got := processContextWindow("ornith", &api.ProcessResponse{
		Models: []api.ProcessModelResponse{
			{Name: "other:latest", Model: "other:latest", ContextLength: 32768},
			{Name: "ornith:latest", Model: "ornith:latest", ContextLength: 262144},
		},
	})
	if got != 262144 {
		t.Fatalf("context window = %d, want 262144", got)
	}
}
