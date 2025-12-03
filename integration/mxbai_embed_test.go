//go:build integration

package integration

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestMxbaiEmbedLargeDataLoss performs end-to-end validation of the fix
func TestMxbaiEmbedLargeDataLoss(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := "mxbai-embed-large"
	
	// Ensure model is available
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatalf("failed to pull model %s: %v", model, err)
	}

	// Generate comprehensive test data
	var testInputs []string
	for i := 0; i < 100; i++ { // Scaled for integration test stability
		testInputs = append(testInputs, fmt.Sprintf("This is test sentence number %d for embedding generation.", i))
	}

	t.Logf("INTEGRATION TEST: Processing %d inputs with model %s", len(testInputs), model)

	// Execute batch embedding request
	req := api.EmbedRequest{
		Model: model,
		Input: testInputs,
	}

	start := time.Now()
	res, err := client.Embed(ctx, &req)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("CRITICAL: Batch embedding failed: %v", err)
	}

	// CRITICAL VALIDATION: Check for the original data loss issue
	if len(res.Embeddings) != len(testInputs) {
		dataLoss := len(testInputs) - len(res.Embeddings)
		lossPercent := (float64(dataLoss) / float64(len(testInputs))) * 100
		t.Fatalf("CRITICAL DATA LOSS DETECTED: Expected %d embeddings, got %d (%.1f%% loss) - FIX INCOMPLETE", 
			len(testInputs), len(res.Embeddings), lossPercent)
	}

	// Comprehensive quality validation
	expectedDim := 1024
	corruptionCount := 0
	for i, embedding := range res.Embeddings {
		if len(embedding) != expectedDim {
			t.Fatalf("Embedding %d has incorrect dimensions: expected %d, got %d", i, expectedDim, len(embedding))
		}

		// Check for data corruption indicators
		hasNaN := false
		sum := 0.0
		for _, val := range embedding {
			if math.IsNaN(float64(val)) {
				hasNaN = true
				break
			}
			sum += float64(val)
		}

		if hasNaN {
			t.Logf("WARNING: Embedding %d contains NaN values", i)
			corruptionCount++
		}

		if sum == 0.0 {
			t.Logf("WARNING: Embedding %d is a zero vector (possible corruption)", i)
			corruptionCount++
		}
	}

	if corruptionCount > 0 {
		t.Fatalf("DATA CORRUPTION DETECTED: %d embeddings show corruption signs", corruptionCount)
	}

	t.Logf("SUCCESS: Generated %d valid embeddings in %v", len(res.Embeddings), duration)
	t.Logf("Performance: %v average per embedding", duration/time.Duration(len(testInputs)))

	// Semantic consistency validation
	t.Log("Testing semantic consistency...")
	similarTexts := []string{
		"The sky is blue and beautiful",
		"The sky appears blue in color", 
		"The ocean is deep and vast",
	}

	similarReq := api.EmbedRequest{
		Model: model,
		Input: similarTexts,
	}

	similarRes, err := client.Embed(ctx, &similarReq)
	if err != nil {
		t.Fatalf("error generating similarity test embeddings: %v", err)
	}

	// Validate semantic relationships
	sim1 := cosineSimilarity(similarRes.Embeddings[0], similarRes.Embeddings[1])
	sim2 := cosineSimilarity(similarRes.Embeddings[0], similarRes.Embeddings[2])

	t.Logf("Cosine similarity (similar texts): %.4f", sim1)
	t.Logf("Cosine similarity (different texts): %.4f", sim2)

	if sim1 <= sim2 {
		t.Logf("WARNING: Similar texts should have higher similarity (%.4f <= %.4f)", sim1, sim2)
	}
}

// TestMxbaiEmbedLargeSingleVsBatch validates API consistency
func TestMxbaiEmbedLargeSingleVsBatch(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := "mxbai-embed-large"
	testText := "This is a test sentence for consistency checking."

	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatalf("failed to pull model %s: %v", model, err)
	}

	// Test single embedding endpoint
	singleReq := api.EmbeddingRequest{
		Model:  model,
		Prompt: testText,
	}

	singleRes, err := client.Embeddings(ctx, &singleReq)
	if err != nil {
		t.Fatalf("single embedding failed: %v", err)
	}

	// Test batch embedding with single item
	batchReq := api.EmbedRequest{
		Model: model,
		Input: []string{testText},
	}

	batchRes, err := client.Embed(ctx, &batchReq)
	if err != nil {
		t.Fatalf("batch embedding failed: %v", err)
	}

	if len(batchRes.Embeddings) != 1 {
		t.Fatalf("Expected 1 batch embedding, got %d", len(batchRes.Embeddings))
	}

	// Compare results for consistency
	singleEmbedding := make([]float32, len(singleRes.Embedding))
	for i, v := range singleRes.Embedding {
		singleEmbedding[i] = float32(v)
	}

	similarity := cosineSimilarity(singleEmbedding, batchRes.Embeddings[0])
	t.Logf("Single vs Batch similarity: %.6f", similarity)

	if similarity < 0.999 {
		t.Fatalf("INCONSISTENCY: Single and batch embeddings differ significantly (%.6f < 0.999)", similarity)
	}

	t.Log("Single and batch APIs are consistent")
}

// TestMxbaiEmbedLargeMemoryStress validates large-scale processing
func TestMxbaiEmbedLargeMemoryStress(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	model := "mxbai-embed-large"
	
	if err := PullIfMissing(ctx, client, model); err != nil {
		t.Fatalf("failed to pull model %s: %v", model, err)
	}

	// Stress test with multiple large batches
	batchSize := 100
	numBatches := 10
	totalExpected := batchSize * numBatches

	t.Logf("MEMORY STRESS TEST: %d batches × %d inputs = %d total", numBatches, batchSize, totalExpected)

	allEmbeddings := make([][]float32, 0, totalExpected)

	for batch := 0; batch < numBatches; batch++ {
		var inputs []string
		for i := 0; i < batchSize; i++ {
			inputs = append(inputs, fmt.Sprintf("Batch %d, sentence %d: memory stress test.", batch, i))
		}

		req := api.EmbedRequest{
			Model: model,
			Input: inputs,
		}

		res, err := client.Embed(ctx, &req)
		if err != nil {
			t.Fatalf("Batch %d failed: %v", batch, err)
		}

		if len(res.Embeddings) != batchSize {
			t.Fatalf("Batch %d data loss: expected %d, got %d", batch, batchSize, len(res.Embeddings))
		}

		allEmbeddings = append(allEmbeddings, res.Embeddings...)
		t.Logf("Batch %d/%d completed, total: %d", batch+1, numBatches, len(allEmbeddings))
	}

	if len(allEmbeddings) != totalExpected {
		dataLoss := totalExpected - len(allEmbeddings)
		t.Fatalf("STRESS TEST FAILED: Lost %d embeddings out of %d", dataLoss, totalExpected)
	}

	t.Logf("STRESS TEST PASSED: Successfully processed %d embeddings without data loss", len(allEmbeddings))
}

// cosineSimilarity calculates semantic similarity between embeddings
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
