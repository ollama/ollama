//go:build integration

package integration

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TODO - this would ideally be in the llm package, but that would require some refactoring of interfaces in the server
//        package to avoid circular dependencies

var (
	stream = false
	req    = [2]api.GenerateRequest{
		{
			Model:  smol,
			Prompt: "why is the ocean blue?",
			Stream: &stream,
			Options: map[string]any{
				"seed":        42,
				"temperature": 0.0,
			},
		}, {
			Model:  smol,
			Prompt: "what is the origin of the us thanksgiving holiday?",
			Stream: &stream,
			Options: map[string]any{
				"seed":        42,
				"temperature": 0.0,
			},
		},
	}
	resp = [2][]string{
		{"sunlight", "scattering", "interact"},
		{"england", "english", "massachusetts", "pilgrims"},
	}
)

func TestIntegrationSimple(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	GenerateTestHelper(ctx, t, req[0], resp[0])
}
