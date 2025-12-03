package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

func main() {
	fmt.Println("🧪 Testing mxbai-embed-large Data Loss Fix")
	fmt.Println(strings.Repeat("=", 50))

	// Test 1: Small batch validation
	fmt.Println("\n📋 Test 1: Small batch validation")
	if err := testSmallBatch(); err != nil {
		log.Fatalf("❌ Small batch test failed: %v", err)
	}
	fmt.Println("✅ Small batch test passed")

	// Test 2: Medium batch validation
	fmt.Println("\n📋 Test 2: Medium batch validation")
	if err := testMediumBatch(); err != nil {
		log.Fatalf("❌ Medium batch test failed: %v", err)
	}
	fmt.Println("✅ Medium batch test passed")

	// Test 3: Large batch stress test
	fmt.Println("\n📋 Test 3: Large batch stress test")
	if err := testLargeBatch(); err != nil {
		log.Fatalf("❌ Large batch test failed: %v", err)
	}
	fmt.Println("✅ Large batch test passed")

	// Test 4: Single vs batch consistency
	fmt.Println("\n📋 Test 4: Single vs batch consistency")
	if err := testConsistency(); err != nil {
		log.Fatalf("❌ Consistency test failed: %v", err)
	}
	fmt.Println("✅ Consistency test passed")

	fmt.Println("\n🎉 ALL TESTS PASSED! The mxbai-embed-large data loss issue is FIXED!")
}

func testSmallBatch() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create client: %v", err)
	}

	inputs := []string{
		"This is a test sentence",
		"Another test sentence",
		"Third test sentence",
	}

	req := &api.EmbedRequest{
		Model: "mxbai-embed-large",
		Input: inputs,
	}

	resp, err := client.Embed(ctx, req)
	if err != nil {
		return fmt.Errorf("embed request failed: %v", err)
	}

	if len(resp.Embeddings) != len(inputs) {
		return fmt.Errorf("data loss: expected %d embeddings, got %d", len(inputs), len(resp.Embeddings))
	}

	// Validate each embedding
	for i, emb := range resp.Embeddings {
		if len(emb) != 1024 {
			return fmt.Errorf("embedding %d has wrong dimensions: expected 1024, got %d", i, len(emb))
		}
		if hasNaNOrInf(emb) {
			return fmt.Errorf("embedding %d contains NaN or Inf values", i)
		}
	}

	return nil
}

func testMediumBatch() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create client: %v", err)
	}

	// Generate 100 test inputs
	inputs := make([]string, 100)
	for i := 0; i < 100; i++ {
		inputs[i] = fmt.Sprintf("Test sentence number %d for batch processing validation.", i)
	}

	req := &api.EmbedRequest{
		Model: "mxbai-embed-large",
		Input: inputs,
	}

	start := time.Now()
	resp, err := client.Embed(ctx, req)
	duration := time.Since(start)

	if err != nil {
		return fmt.Errorf("embed request failed: %v", err)
	}

	if len(resp.Embeddings) != len(inputs) {
		dataLoss := len(inputs) - len(resp.Embeddings)
		lossPercent := float64(dataLoss) / float64(len(inputs)) * 100
		return fmt.Errorf("data loss detected: %d embeddings missing (%.1f%% loss)", dataLoss, lossPercent)
	}

	// Validate embeddings
	for i, emb := range resp.Embeddings {
		if len(emb) != 1024 {
			return fmt.Errorf("embedding %d has wrong dimensions: expected 1024, got %d", i, len(emb))
		}
		if hasNaNOrInf(emb) {
			return fmt.Errorf("embedding %d contains NaN or Inf values", i)
		}
	}

	fmt.Printf("   Processed %d embeddings in %v (%.2fms per embedding)\n", 
		len(resp.Embeddings), duration, float64(duration.Nanoseconds())/float64(len(resp.Embeddings)*1000000))

	return nil
}

func testLargeBatch() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create client: %v", err)
	}

	// Generate 1000 test inputs (scaled down from 5000 for testing)
	inputs := make([]string, 1000)
	for i := 0; i < 1000; i++ {
		inputs[i] = fmt.Sprintf("Large batch test sentence number %d for stress testing.", i)
	}

	req := &api.EmbedRequest{
		Model: "mxbai-embed-large",
		Input: inputs,
	}

	start := time.Now()
	resp, err := client.Embed(ctx, req)
	duration := time.Since(start)

	if err != nil {
		return fmt.Errorf("large batch embed request failed: %v", err)
	}

	if len(resp.Embeddings) != len(inputs) {
		dataLoss := len(inputs) - len(resp.Embeddings)
		lossPercent := float64(dataLoss) / float64(len(inputs)) * 100
		return fmt.Errorf("CRITICAL: Large batch data loss detected: %d embeddings missing (%.1f%% loss)", dataLoss, lossPercent)
	}

	// Sample validation (check first 100 embeddings for efficiency)
	sampleSize := 100
	for i := 0; i < sampleSize && i < len(resp.Embeddings); i++ {
		emb := resp.Embeddings[i]
		if len(emb) != 1024 {
			return fmt.Errorf("embedding %d has wrong dimensions: expected 1024, got %d", i, len(emb))
		}
		if hasNaNOrInf(emb) {
			return fmt.Errorf("embedding %d contains NaN or Inf values", i)
		}
	}

	fmt.Printf("   Successfully processed %d embeddings in %v (%.2fms per embedding)\n", 
		len(resp.Embeddings), duration, float64(duration.Nanoseconds())/float64(len(resp.Embeddings)*1000000))

	return nil
}

func testConsistency() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("failed to create client: %v", err)
	}

	testText := "This is a consistency test sentence."

	// Test single embedding
	singleReq := &api.EmbeddingRequest{
		Model:  "mxbai-embed-large",
		Prompt: testText,
	}

	singleResp, err := client.Embeddings(ctx, singleReq)
	if err != nil {
		return fmt.Errorf("single embedding request failed: %v", err)
	}

	// Test batch embedding with one item
	batchReq := &api.EmbedRequest{
		Model: "mxbai-embed-large",
		Input: []string{testText},
	}

	batchResp, err := client.Embed(ctx, batchReq)
	if err != nil {
		return fmt.Errorf("batch embedding request failed: %v", err)
	}

	if len(batchResp.Embeddings) != 1 {
		return fmt.Errorf("batch should return 1 embedding, got %d", len(batchResp.Embeddings))
	}

	// Compare single vs batch
	// Convert single embedding to float32 for comparison
	singleEmbedding32 := make([]float32, len(singleResp.Embedding))
	for i, v := range singleResp.Embedding {
		singleEmbedding32[i] = float32(v)
	}
	
	similarity := cosineSimilarity(singleEmbedding32, batchResp.Embeddings[0])
	if similarity < 0.999 {
		return fmt.Errorf("single and batch embeddings differ significantly: similarity %.6f < 0.999", similarity)
	}

	fmt.Printf("   Single vs batch similarity: %.6f\n", similarity)
	return nil
}

func hasNaNOrInf(emb []float32) bool {
	for _, v := range emb {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

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
