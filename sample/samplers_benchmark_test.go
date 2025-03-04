package sample

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkWeightedSampler tests the performance of the weighted sampler
// with various input sizes and configurations
func BenchmarkWeightedSampler(b *testing.B) {
	// Different sizes of logits to test
	sizes := []int{10, 100, 1000, 10000}

	// Different seed values to test
	seedValue := uint64(42)

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			// Create logits with random values
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5) // values between -5 and 5
			}

			// Initialize sampler with seed
			sampler := Weighted(&seedValue)

			// Reset timer before the actual benchmark
			b.ResetTimer()

			// Run the benchmark
			for i := 0; i < b.N; i++ {
				_, err := sampler.Sample(logits)
				if err != nil {
					b.Fatalf("Sampling failed: %v", err)
				}
			}
		})
	}

	// Test with different transforms
	transforms := []struct {
		name      string
		transform []Transform
	}{
		{"NoTransform", []Transform{}},
		{"Temperature", []Transform{Temperature(0.8)}},
		{"TopK", []Transform{TopK(50)}},
		{"TopP", []Transform{TopP(0.9)}},
		{"MinP", []Transform{MinP(0.05)}},
		{"Combined", []Transform{Temperature(0.8), TopK(50), TopP(0.9), MinP(0.05)}},
	}

	// Fixed size for transform tests
	size := 1000
	logits := make([]float32, size)
	for i := range logits {
		logits[i] = float32(rand.Float64()*10 - 5)
	}

	for _, tc := range transforms {
		b.Run("Transform"+tc.name, func(b *testing.B) {
			sampler := Weighted(&seedValue, tc.transform...)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, err := sampler.Sample(logits)
				if err != nil {
					b.Fatalf("Sampling failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkGreedySampler tests the performance of the greedy sampler
// with various input sizes
func BenchmarkGreedySampler(b *testing.B) {
	// Different sizes of logits to test
	sizes := []int{10, 100, 1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			// Create logits with random values
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5) // values between -5 and 5
			}

			// Initialize sampler
			sampler := Greedy()

			// Reset timer before the actual benchmark
			b.ResetTimer()

			// Run the benchmark
			for i := 0; i < b.N; i++ {
				_, err := sampler.Sample(logits)
				if err != nil {
					b.Fatalf("Sampling failed: %v", err)
				}
			}
		})
	}
}
