package sample

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestTemperature(t *testing.T) {
	got, err := Temperature(0.5).Apply([]float64{2, -1, 4, -3, 1, -2, 0})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{-4, -10, 0, -14, -6, -12, -8}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	got, err = Temperature(-1).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for temperature=-1, got %v", got)
	}
	got, err = Temperature(0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for temperature=0, got %v", got)
	}
	got, err = Temperature(2.1).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for temperature=2.1, got %v", got)
	}
}

func TestSoftmax(t *testing.T) {
	got := softmax([]float64{-3, -2, -1, 0, 1, 2, 4})

	want := []float64{0.000751406628089903, 0.0020425349829204676, 0.005552185728064613, 0.015092405572827691, 0.04102541181635154, 0.11151863144543739, 0.8240174238263085}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("probs mismatch (-want +got):\n%s", diff)
	}
}

func TestTopK(t *testing.T) {
	got, err := TopK(3).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 1, 2, 4}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	_, err = TopK(0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Errorf("expected error for k=0, got %v", err)
	}

	got, err = TopK(10).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Error(err)
		return
	}
	want = []float64{-3, -2, -1, 0, 1, 2, 4}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}
}

func TestTopP(t *testing.T) {
	got, err := TopP(0.9).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 2, 4}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	_, err = TopP(1.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Error("expected error for p=1.0")
	}
	_, err = TopP(0.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Error("expected error for p=0.0")
	}
}

func TestMinP(t *testing.T) {
	got, err := MinP(0.2).Apply([]float64{-3, -2, -1, 0, 1, 2, 4, 3})
	if err != nil {
		t.Error(err)
		return
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 4, 3}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("logits mismatch (-want +got):\n%s", diff)
	}

	_, err = MinP(1.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Error("expected error for p=1.0")
	}
	_, err = MinP(0.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Error("expected error for p=0.0")
	}
}

func BenchmarkTransform(b *testing.B) {
	transforms := map[string]Transform{
		"Temperature": Temperature(0.5),
		"TopK":        TopK(10),
		"TopP":        TopP(0.9),
		"MinP":        MinP(0.2),
	}

	logits := make([]float64, 1<<16)
	for i := range logits {
		logits[i] = rand.Float64()
	}

	for name, transform := range transforms {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for range b.N {
				_, err := transform.Apply(logits)
				if err != nil {
					b.Error(err)
				}
			}
		})
	}
}
