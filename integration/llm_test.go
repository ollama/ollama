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
			Model:  "orca-mini",
			Prompt: "why is the ocean blue?",
			Stream: &stream,
			Options: map[string]interface{}{
				"seed":        42,
				"temperature": 0.0,
			},
		}, {
			Model:  "orca-mini",
			Prompt: "what is the origin of the us thanksgiving holiday?",
			Stream: &stream,
			Options: map[string]interface{}{
				"seed":        42,
				"temperature": 0.0,
			},
		},
	}
	resp = [2][]string{
		{"sunlight"},
		{"england", "english", "massachusetts", "pilgrims"},
	}
)

func TestIntegrationSimpleOrcaMini(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	GenerateTestHelper(ctx, t, req[0], resp[0])
}
