package sample

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWeighted(t *testing.T) {
	idx, err := Weighted(nil).Sample([]float32{float32(math.Inf(-1)), 2, float32(math.Inf(-1)), float32(math.Inf(-1))})
	if err != nil {
		t.Error(err)
		return
	}
	want := int32(1)
	if diff := cmp.Diff(want, idx); diff != "" {
		t.Errorf("index mismatch (-want +got):\n%s", diff)
	}

	idx, err = Weighted(nil).Sample([]float32{float32(math.Inf(-1)), float32(math.Inf(-1)), float32(math.Inf(-1))})
	if err == nil {
		t.Error("expected error for no valid tokens, got index", idx)
	}

	seed := int64(42)
	idx, err = Weighted(&seed).Sample([]float32{1, 2, 3, 4})
	if err != nil {
		t.Error(err)
		return
	}
	// With seed 42, we expect a consistent sample
	want = int32(3) // This will be deterministic due to the seed
	if diff := cmp.Diff(want, idx); diff != "" {
		t.Errorf("seeded sample index mismatch (-want +got):\n%s", diff)
	}
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

	got, err := Greedy(mock1, mock2, mock3).Sample(input)
	if err != nil {
		t.Error(err)
		return
	}

	want := int32(3) // Greedy sampler should pick highest logit
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("sampled index mismatch (-want +got):\n%s", diff)
	}
	wantOrder := []int{1, 2, 3}
	if diff := cmp.Diff(wantOrder, callOrder); diff != "" {
		t.Errorf("call order mismatch (-want +got):\n%s", diff)
	}
	callOrder = nil

	_, err = Weighted(nil, mock1, mock2, mock3).Sample(input)
	if err != nil {
		t.Error(err)
		return
	}
	wantOrder = []int{1, 2, 3}
	if diff := cmp.Diff(wantOrder, callOrder); diff != "" {
		t.Errorf("call order mismatch (-want +got):\n%s", diff)
	}

	errMock := &testTransform{
		returnErr: fmt.Errorf("mock error"),
	}
	_, err = Weighted(nil, mock1, errMock, mock2).Sample(input)
	if err == nil {
		t.Error("Expected error from sampler")
	}
}

type testTransform struct {
	id        int
	callOrder *[]int
	returnErr error
}

func (ts *testTransform) Apply(logits []float64) ([]float64, error) {
	if ts.callOrder != nil {
		*ts.callOrder = append(*ts.callOrder, ts.id)
	}
	if ts.returnErr != nil {
		return nil, ts.returnErr
	}
	return logits, nil
}

func BenchmarkSample(b *testing.B) {
	transforms := []Transform{
		Temperature(0.5),
		TopK(10),
		TopP(0.9),
		MinP(0.2),
	}

	samplers := map[string]Sampler{
		"Greedy":   Greedy(transforms...),
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
