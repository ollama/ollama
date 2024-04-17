//go:build integration

package integration

import (
	"context"
	"net/http"
	"sync"
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
		[]string{"sunlight"},
		[]string{"england", "english", "massachusetts", "pilgrims"},
	}
)

func TestIntegrationSimpleOrcaMini(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	GenerateTestHelper(ctx, t, &http.Client{}, req[0], resp[0])
}

// TODO
// The server always loads a new runner and closes the old one, which forces serial execution
// At present this test case fails with concurrency problems.  Eventually we should try to
// get true concurrency working with n_parallel support in the backend
func TestIntegrationConcurrentPredictOrcaMini(t *testing.T) {
	var wg sync.WaitGroup
	wg.Add(len(req))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*120)
	defer cancel()
	for i := 0; i < len(req); i++ {
		go func(i int) {
			defer wg.Done()
			GenerateTestHelper(ctx, t, &http.Client{}, req[i], resp[i])
		}(i)
	}
	wg.Wait()
}

// TODO - create a parallel test with 2 different models once we support concurrency
