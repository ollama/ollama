package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestList(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Setenv("OLLAMA_MODELS", t.TempDir())

	expectNames := []string{
		"mistral:7b-instruct-q4_0",
		"zephyr:7b-beta-q5_K_M",
		"apple/OpenELM:latest",
		"boreas:2b-code-v1.5-q6_K",
		"notus:7b-v1-IQ2_S",
		// TODO: host:port currently fails on windows (#4107)
		// "localhost:5000/library/eurus:700b-v0.5-iq3_XXS",
		"mynamespace/apeliotes:latest",
		"myhost/mynamespace/lips:code",
	}

	var s Server
	for _, n := range expectNames {
		_, digest := createBinFile(t, nil, nil)

		createRequest(t, s.CreateHandler, api.CreateRequest{
			Name:  n,
			Files: map[string]string{"test.gguf": digest},
		})
	}

	w := createRequest(t, s.ListHandler, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ListResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if len(resp.Models) != len(expectNames) {
		t.Fatalf("expected %d models, actual %d", len(expectNames), len(resp.Models))
	}

	actualNames := make([]string, len(resp.Models))
	for i, m := range resp.Models {
		actualNames[i] = m.Name
	}

	slices.Sort(actualNames)
	slices.Sort(expectNames)

	if !slices.Equal(actualNames, expectNames) {
		t.Fatalf("expected slices to be equal %v", actualNames)
	}
}

func BenchmarkListHandler(b *testing.B) {
	gin.SetMode(gin.TestMode)

	// Test with higher model counts to simulate real-world scenarios
	modelCounts := []int{50, 100, 250, 500, 1000, 2000}

	for _, count := range modelCounts {
		b.Run(fmt.Sprintf("models_%d", count), func(b *testing.B) {
			benchmarkListWithModelCount(b, count)
		})
	}
}

func benchmarkListWithModelCount(b *testing.B, modelCount int) {
	// Setup
	tempDir := b.TempDir()
	b.Setenv("OLLAMA_MODELS", tempDir)

	var s Server

	// Create the specified number of models
	b.Logf("Creating %d models for benchmark...", modelCount)
	for i := range modelCount {
		modelName := fmt.Sprintf("testmodel%d:latest", i)
		_, digest := createBinFile(b, nil, nil)

		createRequest(b, s.CreateHandler, api.CreateRequest{
			Name:  modelName,
			Files: map[string]string{"test.gguf": digest},
		})

		// Log progress for large numbers
		if modelCount >= 500 && i%100 == 0 {
			b.Logf("Created %d/%d models", i, modelCount)
		}
	}

	b.Logf("Setup complete, starting benchmark with %d models", modelCount)

	// Reset timer to exclude setup time
	b.ResetTimer()

	// Run the actual benchmark
	for i := 0; i < b.N; i++ {
		w := createRequest(b, s.ListHandler, nil)
		if w.Code != http.StatusOK {
			b.Fatalf("expected status code 200, actual %d", w.Code)
		}

		// Optional: Verify we got the expected number of models
		if i == 0 { // Only check on first iteration to avoid overhead
			var resp api.ListResponse
			if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
				b.Fatal(err)
			}
			if len(resp.Models) != modelCount {
				b.Fatalf("expected %d models, got %d", modelCount, len(resp.Models))
			}
		}
	}
}
