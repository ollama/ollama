package sample

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWeighted(t *testing.T) {
	got, err := Weighted(nil).Sample([]float32{float32(math.Inf(-1)), 2, float32(math.Inf(-1)), float32(math.Inf(-1))})
	if err != nil {
		t.Error(err)
		return
	}
	want := int32(1)
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}

	got, err = Weighted(nil).Sample([]float32{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))})
	if err == nil {
		t.Error("expected error for no valid tokens, got index", got)
	}

	seed := uint64(42)
	got, err = Weighted(&seed).Sample([]float32{1, 2, 3, 4})
	if err != nil {
		t.Error(err)
		return
	}
	// With seed 42, we expect a consistent sample
	want = int32(3) // This will be deterministic due to the seed
	if want != got {
		t.Errorf("index mismatch: want %d, got %d", want, got)
	}
}

type testTransform struct {
	id        int
	callOrder *[]int
}

func (ts *testTransform) Apply(logits []float64) []float64 {
	if ts.callOrder != nil {
		*ts.callOrder = append(*ts.callOrder, ts.id)
	}
	return logits
}

func TestSample(t *testing.T) {
	input := []float32{1, 2, 3, 4}

	var callOrder []int
	mock1 := &testTransform{
		id:        1,
		callOrder: &callOrder,
	}
	mock2 := &testTransform{
		id:        2,
		callOrder: &callOrder,
	}
	mock3 := &testTransform{
		id:        3,
		callOrder: &callOrder,
	}

	_, err := Weighted(nil, mock1, mock2, mock3).Sample(input)
	if err != nil {
		t.Error(err)
		return
	}
	wantOrder := []int{1, 2, 3}
	if diff := cmp.Diff(wantOrder, callOrder); diff != "" {
		t.Errorf("call order mismatch (-want +got):\n%s", diff)
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
			name: "no transforms",
			// temperature is 0, so greedy should be used
			wantErr: false,
		},
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
			topK:        10,
			temperature: 0.8,
			wantErr:     false,
		},
		{
			name:        "invalid top k negative",
			topK:        -1,
			temperature: 0.8,
			wantErr:     true,
		},
		{
			name:        "top p",
			topP:        0.9,
			temperature: 0.8,
			wantErr:     false,
		},
		{
			name:        "invalid top p negative",
			topP:        -0.1,
			temperature: 0.8,
			wantErr:     true,
		},
		{
			name:        "invalid top p one",
			topP:        1.0,
			temperature: 0.8,
			wantErr:     true,
		},
		{
			name:        "min p",
			minP:        0.2,
			temperature: 0.8,
			wantErr:     false,
		},
		{
			name:        "invalid min p negative",
			minP:        -0.1,
			temperature: 0.8,
			wantErr:     true,
		},
		{
			name:        "invalid min p one",
			minP:        1.0,
			temperature: 0.8,
			wantErr:     true,
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
			name:        "all zeroes",
			temperature: 0.0,
			topK:        0,
			topP:        0.0,
			minP:        0.0,
			seed:        0,
			wantErr:     false, // all zeroes means no transforms
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
	transforms := []Transform{
		Temperature(0.5),
		TopK(10),
		TopP(0.9),
		MinP(0.2),
	}

	samplers := map[string]Sampler{
		"Greedy":   Greedy(),
		"Weighted": Weighted(nil, transforms...),
	}

	logits := make([]float32, 1<<16)
	for i := range logits {
		logits[i] = rand.Float32()
	}

	for name, s := range samplers {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for range b.N {
				if _, err := s.Sample(logits); err != nil {
					b.Error(err)
				}
			}
		})
	}
}
