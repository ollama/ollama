package sample

import (
	"math/rand/v2"
	"testing"
)

func TestWeighted(t *testing.T) {
	logits := []float32{-10, 3, -10, -10}
	sampler := NewSampler(0, 0, 0, 0, 0)
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
	sampler = NewSampler(0, 0, 0, 0, 0)
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
		wantGreedy  bool // Instead of wantErr, check if we get greedy sampler
	}{
		{
			name:        "temperature",
			temperature: 0.5,
			wantGreedy:  false,
		},
		{
			name:        "zero temperature - greedy",
			temperature: 0,
			wantGreedy:  true,
		},
		{
			name:        "top k",
			temperature: 0.1,
			topK:        10,
			wantGreedy:  false,
		},
		{
			name:        "top p",
			temperature: 0.1,
			topP:        0.9,
			wantGreedy:  false,
		},
		{
			name:        "min p",
			temperature: 0.1,
			minP:        0.2,
			wantGreedy:  false,
		},
		{
			name:        "seed - weighted",
			temperature: 0.1,
			seed:        42,
			wantGreedy:  false,
		},
		{
			name:        "default values",
			temperature: 0.8,
			topK:        40,
			topP:        0.9,
			minP:        0.0,
			seed:        0,
			wantGreedy:  false,
		},
		{
			name:        "all zeroes - greedy",
			temperature: 0.0,
			topK:        0,
			topP:        0.0,
			minP:        0.0,
			seed:        0,
			wantGreedy:  true,
		},
		{
			name:        "all transforms",
			temperature: 0.8,
			topK:        50,
			topP:        0.95,
			minP:        0.1,
			seed:        42,
			wantGreedy:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sampler := NewSampler(tt.temperature, tt.topK, tt.topP, tt.minP, tt.seed)
			_, isGreedy := sampler.(*greedy)
			if isGreedy != tt.wantGreedy {
				t.Errorf("NewSampler() got greedy = %v, want %v", isGreedy, tt.wantGreedy)
			}
		})
	}
}

func BenchmarkSample(b *testing.B) {
	weighted := NewSampler(0.5, 10, 0.9, 0.2, -1)
	samplers := map[string]Sampler{
		"Greedy":   NewSampler(0, 0, 0, 0, 0), // Use NewSampler with temp=0 for greedy
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
			for b.Loop() {
				if _, err := s.Sample(logits); err != nil {
					b.Error(err)
				}
			}
		})
	}
}
