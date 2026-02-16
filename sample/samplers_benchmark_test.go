package sample

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkWeightedSampler(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5)
			}

			sampler := NewSampler(0.8, 0, 0, 0, 42, nil)
			b.ResetTimer()
			for b.Loop() {
				sampler.Sample(logits)
			}
		})
	}

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

	// Fixed size for common vocab size
	size := 128000
	logits := make([]float32, size)
	for i := range logits {
		logits[i] = float32(rand.Float64()*10 - 5)
	}

	for _, tc := range configs {
		b.Run("Config"+tc.name, func(b *testing.B) {
			sampler := NewSampler(tc.temperature, tc.topK, tc.topP, tc.minP, tc.seed, nil)
			sampler.Sample(logits)

			b.ResetTimer()

			for b.Loop() {
				sampler.Sample(logits)
			}
		})
	}

	// Test with combined transforms separately - topK influences performance greatly
	b.Run("TransformCombined", func(b *testing.B) {
		sampler := NewSampler(0.8, 50, 0.9, 0.05, 42, nil)
		b.ResetTimer()

		for b.Loop() {
			sampler.Sample(logits)
		}
	})
}

func BenchmarkGreedySampler(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000, 100000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size %d", size), func(b *testing.B) {
			logits := make([]float32, size)
			for i := range logits {
				logits[i] = float32(rand.Float64()*10 - 5)
			}

			sampler := NewSampler(0, -1, 0, 0, -1, nil)
			b.ResetTimer()

			for b.Loop() {
				sampler.Sample(logits)
			}
		})
	}
}
