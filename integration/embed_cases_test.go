//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func registerEmbeddingCases(models []string) {
	registerEmbeddingCasesWithFallback(models, false)
}

func registerLibraryEmbeddingCases(models []string) {
	registerEmbeddingCasesWithFallback(models, true)
}

func registerEmbeddingCasesWithFallback(models []string, smokeMissing bool) {
	testCases, err := loadEmbeddingTestCases()
	if err != nil {
		registerIntegrationCases(integrationCase{
			Key:   "embed/testdata",
			Case:  "embed",
			Model: "testdata",
			Run: func(t *testing.T) {
				t.Fatalf("failed to load embedding test data: %s", err)
			},
		})
		return
	}

	if testModel != "" {
		models = []string{testModel}
	}

	cases := make([]integrationCase, 0, len(models))
	for _, model := range models {
		model := model
		expected, ok := embeddingExpected(testCases, model)
		if !ok {
			if smokeMissing || testModel != "" {
				cases = append(cases, embeddingSmokeCase(model))
				continue
			}
			cases = append(cases, integrationCase{
				Key:   "embed/" + model,
				Case:  "embed",
				Model: model,
				Run: func(t *testing.T) {
					t.Skipf("no embedding expectation for model %s", model)
				},
			})
			continue
		}

		cases = append(cases, embeddingCase(model, expected))
	}
	registerIntegrationCases(cases...)
}

func embeddingSmokeCase(model string) integrationCase {
	return integrationCase{
		Key:   "embed/" + model,
		Case:  "embed",
		Model: model,
		Run: func(t *testing.T) {
			runEmbeddingSmokeModel(t, model)
		},
	}
}

func embeddingCase(model string, expected []float64) integrationCase {
	return integrationCase{
		Key:   "embed/" + model,
		Case:  "embed",
		Model: model,
		Run: func(t *testing.T) {
			runEmbeddingModel(t, model, expected)
		},
	}
}

func loadEmbeddingTestCases() (map[string][]float64, error) {
	data, err := os.ReadFile(filepath.Join("testdata", "embed.json"))
	if err != nil {
		return nil, err
	}
	testCases := map[string][]float64{}
	if err := json.Unmarshal(data, &testCases); err != nil {
		return nil, err
	}
	return testCases, nil
}

func embeddingExpected(testCases map[string][]float64, model string) ([]float64, bool) {
	if expected, ok := testCases[model]; ok {
		return expected, true
	}
	if !strings.Contains(model, ":") {
		expected, ok := testCases[model+":latest"]
		return expected, ok
	}
	return nil, false
}

func runEmbeddingModel(t *testing.T, model string, expected []float64) {
	t.Helper()

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	if time.Since(started) > softTimeout {
		t.Skip("skipping remaining tests to avoid excessive runtime")
	}
	pullOrSkip(ctx, t, client, model)
	skipIfModelTooLargeForSweepVRAM(ctx, t, client, model)

	req := api.EmbeddingRequest{
		Model:     model,
		Prompt:    "why is the sky blue?",
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
		Options: map[string]any{
			"temperature": 0,
			"seed":        123,
		},
	}
	resp, err := client.Embeddings(ctx, &req)
	if err != nil {
		t.Fatalf("embeddings call failed %s", err)
	}
	defer func() {
		client.Generate(ctx, &api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
	}()
	if len(resp.Embedding) == 0 {
		t.Errorf("zero length embedding response")
	}
	if len(expected) != len(resp.Embedding) {
		expStr := make([]string, len(resp.Embedding))
		for i, v := range resp.Embedding {
			expStr[i] = fmt.Sprintf("%0.6f", v)
		}
		// When adding new models, use this output to populate the testdata/embed.json
		fmt.Printf("expected\n%s\n", strings.Join(expStr, ", "))
		t.Fatalf("expected %d, got %d", len(expected), len(resp.Embedding))
	}
	sim := cosineSimilarity(resp.Embedding, expected)
	if sim < 0.99 {
		t.Fatalf("expected %v, got %v (similarity: %f)", expected[0:5], resp.Embedding[0:5], sim)
	}
}

func runEmbeddingSmokeModel(t *testing.T, model string) {
	t.Helper()

	softTimeout, hardTimeout := getTimeouts(t)
	ctx, cancel := context.WithTimeout(context.Background(), hardTimeout)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	if time.Since(started) > softTimeout {
		t.Skip("skipping remaining tests to avoid excessive runtime")
	}
	requireCapability(ctx, t, client, model, "embedding")
	skipIfModelTooLargeForSweepVRAM(ctx, t, client, model)

	req := api.EmbedRequest{
		Model:     model,
		Input:     []string{"cat", "kitten", "dog"},
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
	}
	resp, err := embedTestHelper(ctx, client, t, req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		client.Generate(ctx, &api.GenerateRequest{Model: req.Model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
	}()
	if len(resp.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(resp.Embeddings))
	}
	for i, embedding := range resp.Embeddings {
		if len(embedding) == 0 {
			t.Fatalf("embedding %d was empty", i)
		}
	}

	cosRelated := cosineSimilarity(resp.Embeddings[0], resp.Embeddings[1])
	cosUnrelated := cosineSimilarity(resp.Embeddings[0], resp.Embeddings[2])
	if cosRelated <= cosUnrelated {
		t.Fatalf("expected related terms to be closer than unrelated terms: cat/kitten=%f cat/dog=%f", cosRelated, cosUnrelated)
	}
}
