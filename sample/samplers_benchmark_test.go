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

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			// Create logits with random values
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5) // values between -5 and 5
			}

			// Initialize sampler with seed
			// sampler := Weighted(seedValue)
			sampler, err := NewSampler(0.8, 0, 0, 0, 42)
			if err != nil {
				b.Error(err)
				return
			}

			// Reset timer before the actual benchmark
			b.ResetTimer()

			// Run the benchmark
			for b.Loop() {
				_, err := sampler.Sample(logits)
				if err != nil {
					b.Fatalf("Sampling failed: %v", err)
				}
			}
		})
	}

	// Test with different sampling configurations
	configs := []struct {
		name        string
		temperature float32
		topK        int
		topP        float32
		minP        float32
		seed        int
	}{
		{"Greedy", 0, -1, 0, 0, -1},
		{"Temperature", 0.8, -1, 0, 0, -1},
		{"TopK", 0.8, 50, 0, 0, -1},
		{"TopP", 0.8, -1, 0.9, 0, -1},
		{"MinP", 0.8, -1, 0, 0.05, -1},
		{"WithSeed", 0.8, 50, 0, 0, 42},
	}

	// Fixed size for configuration tests
	size := 128000
	logits := make([]float32, size)
	for i := range logits {
		logits[i] = float32(rand.Float64()*10 - 5)
	}

	for _, tc := range configs {
		b.Run("Config"+tc.name, func(b *testing.B) {
			sampler, err := NewSampler(tc.temperature, tc.topK, tc.topP, tc.minP, tc.seed)
			if err != nil {
				b.Error(err)
				return
			}
			sampler.Sample(logits)

			b.ResetTimer()

			for b.Loop() {
				_, err := sampler.Sample(logits)
				if err != nil {
					b.Fatalf("Sampling failed: %v", err)
				}
			}
		})
	}

	// Test with combined transforms separately
	b.Run("TransformCombined", func(b *testing.B) {
		sampler, err := NewSampler(0.8, 50, 0.9, 0.05, 42)
		if err != nil {
			b.Error(err)
			return
		}

		b.ResetTimer()

		for b.Loop() {
			_, err := sampler.Sample(logits)
			if err != nil {
				b.Fatalf("Sampling failed: %v", err)
			}
		}
	})
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
			for b.Loop() {
				_, err := sampler.Sample(logits)
				if err != nil {
					b.Fatalf("Sampling failed: %v", err)
				}
			}
		})
	}
}
