package sample

import (
	"math/rand/v2"
	"testing"
)

func TestWeighted(t *testing.T) {
	logits := []float32{-10, 3, -10, -10}
	sampler, err := NewSampler(0, 0, 0, 0, 0)
	if err != nil {
		t.Error(err)
		return
	}
	// sampler := Weighted(0, softmax{}, sortTokens{})
	got, err := sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	want := int32(1)
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	// Test with seed for deterministic sampling
	// r := float32(0.5) // Use a fixed random value for deterministic sampling
	logits = []float32{-100, -10, 0, 10}
	sampler, err = NewSampler(0, 0, 0, 0, 0)
	if err != nil {
		t.Error(err)
		return
	}
	got, err = sampler.Sample(logits)
	if err != nil {
		t.Error(err)
		return
	}
	want = int32(3) // Should pick highest probability with this r value
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}
}

func TestNewSampler(t *testing.T) {
	tests := []struct {
		name        string
		temperature float32
		topK        int
		topP        float32
		minP        float32
		seed        int
		wantErr     bool
	}{
		{
			name:        "temperature",
			temperature: 0.5,
			wantErr:     false,
		},
		{
			name:        "invalid temperature negative",
			temperature: -1,
			wantErr:     true,
		},
		{
			name:        "invalid temperature too high",
			temperature: 2.1,
			wantErr:     true,
		},
		{
			name:        "top k",
			temperature: 0.1,
			topK:        10,
			wantErr:     false,
		},
		{
			name:        "top p",
			temperature: 0.1,
			topP:        0.9,
			wantErr:     false,
		},
		{
			name:        "invalid top p negative",
			temperature: 0.1,
			topP:        -0.1,
			wantErr:     true,
		},
		{
			name:        "invalid top p one",
			temperature: 0.1,
			topP:        1.0,
			wantErr:     true,
		},
		{
			name:        "min p",
			temperature: 0.1,
			minP:        0.2,
			wantErr:     false,
		},
		{
			name:        "invalid min p negative",
			temperature: 0.1,
			minP:        -0.1,
			wantErr:     true,
		},
		{
			name:        "invalid min p one",
			temperature: 0.1,
			minP:        1.0,
			wantErr:     true,
		},
		{
			name:        "seed - greedy",
			temperature: 0.1,
			seed:        42,
			wantErr:     false,
		},
		{
			name:        "default values",
			temperature: 0.8,
			topK:        40,
			topP:        0.9,
			minP:        0.0,
			seed:        0,
			wantErr:     false,
		},
		{
			name:        "all zeroes - greedy",
			temperature: 0.0,
			topK:        0,
			topP:        0.0,
			minP:        0.0,
			seed:        0,
			wantErr:     false,
		},
		{
			name:        "all transforms",
			temperature: 0.8,
			topK:        50,
			topP:        0.95,
			minP:        0.1,
			seed:        42,
			wantErr:     false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSampler(tt.temperature, tt.topK, tt.topP, tt.minP, tt.seed)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewSampler() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func BenchmarkSample(b *testing.B) {
	weighted, err := NewSampler(0.5, 10, 0.9, 0.2, -1)
	if err != nil {
		b.Error(err)
		return
	}
	samplers := map[string]Sampler{
		"Greedy":   Greedy(),
		"Weighted": weighted,
	}

	// Generate random logits for benchmarking
	logits := make([]float32, 1<<16)
	for i := range logits {
		logits[i] = rand.Float32()
	}

	for name, s := range samplers {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := s.Sample(logits); err != nil {
					b.Error(err)
				}
			}
		})
	}
}
