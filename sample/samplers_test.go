package sample

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestWeighted(t *testing.T) {
	logits := []float32{-10, 3, -10, -10}
	sampler := NewSampler(0, 0, 0, 0, 0, nil)
	got, err := sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	want := int32(1)
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	logits = []float32{-100, -10, 0, 10}
	sampler = NewSampler(0, 0, 0, 0, 0, nil)
	got, err = sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	want = int32(3) // Should pick highest probability with this r value
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	// Test very high p
	logits = []float32{1.0, 0.9999999999999999, 0.5, 0.1}
	// Use extremely small topP to filter out all tokens
	sampler = NewSampler(1.0, 0, 1e-10, 0, 0, nil)
	got, err = sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	// Should get the token with the highest logit
	want = int32(0)
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	logits = []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN())}
	sampler = NewSampler(1, 0, 0.95, 0.05, 0, nil)
	got, err = sampler.Sample(logits)
	if err == nil {
		t.Errorf("expected error, got %d", got)
		return
	}
}

func BenchmarkSample(b *testing.B) {
	samplers := map[string]Sampler{
		"Greedy":   NewSampler(0, 0, 0, 0, 0, nil), // Use NewSampler with temp=0 for greedy
		"Weighted": NewSampler(0.5, 10, 0.9, 0.2, -1, nil),
	}

	// Generate random logits for benchmarking
	logits := make([]float32, 1<<16)
	for i := range logits {
		logits[i] = rand.Float32()
	}

	for name, s := range samplers {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for b.Loop() {
				if _, err := s.Sample(logits); err != nil {
					b.Fatalf("error sampling: %v", err)
				}
			}
		})
	}
}
