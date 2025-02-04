package sample

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestTemperature(t *testing.T) {
	logits, err := Temperature(0.5).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{-14, -12, -10, -8, -6, -4, 0}
	if !floats.Equal(logits, want) {
		t.Fatalf("got: %v, want: %v", logits, want)
	}

	if _, err := Temperature(-1).Apply([]float64{-3, -2, -1, 0, 1, 2, 4}); err == nil {
		t.Fatalf("expected error for temperature=-1, got %v", logits)
	}
	if _, err := Temperature(2.1).Apply([]float64{-3, -2, -1, 0, 1, 2, 4}); err == nil {
		t.Fatalf("expected error for temperature=2.1, got %v", logits)
	}
}

func TestSoftmax(t *testing.T) {
	probs, err := Softmax().Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}

	expectedProbs := []float64{0.000751406628089903, 0.0020425349829204676, 0.005552185728064613, 0.015092405572827691, 0.04102541181635154, 0.11151863144543739, 0.8240174238263085}
	if !floats.Equal(probs, expectedProbs) {
		t.Fatalf("logits: %v, expectedlogits: %v", probs, expectedProbs)
	}
}

func TestTopK(t *testing.T) {
	logits, err := TopK(3).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedlogits := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 1, 2, 4}
	if !floats.Same(logits, expectedlogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedlogits)
	}
	logits, err = TopK(0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Fatalf("expected error for k=0, got %v", logits)
	}

	logits, err = TopK(10).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedlogits = []float64{-3, -2, -1, 0, 1, 2, 4}
	if !floats.Same(logits, expectedlogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedlogits)
	}
}

func TestTopP(t *testing.T) {
	logits, err := TopP(0.9).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 2, 4}
	if !floats.Same(logits, want) {
		t.Fatalf("got: %v, want: %v", logits, want)
	}
	logits, err = TopP(1.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Fatalf("expected error for p=1.0, got %v", logits)
	}
	logits, err = TopP(0.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Fatalf("expected error for p=0.0, got %v", logits)
	}
}

func TestMinP(t *testing.T) {
	logits, err := MinP(0.2).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), math.Inf(-1), 3, 4}
	if !floats.Same(logits, want) {
		t.Fatalf("got: %v, want: %v", logits, want)
	}
	logits, err = MinP(1.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Fatalf("expected error for p=1.0, got %v", logits)
	}
	logits, err = MinP(0.0).Apply([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Fatalf("expected error for p=0.0, got %v", logits)
	}
}

func TestWeighed(t *testing.T) {
	idx, err := Weighed().Sample([]float64{math.Inf(-1), 2, math.Inf(-1), math.Inf(-1)})
	if err != nil {
		t.Fatal(err)
	}
	want := 1
	if idx != want {
		t.Fatalf("got: %v, want: %v", idx, want)
	}
	idx, err = Weighed().Sample([]float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)})
	if err == nil {
		t.Fatalf("expected error for no valid tokens, got %v", idx)
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
	sampler := NewSampler([]Transform{mock1, mock2, mock3}, Greedy())

	got, err := sampler.Sample(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(callOrder, []int{1, 2, 3}) {
		t.Errorf("got %v, want %v", callOrder, []int{1, 2, 3})
	}

	want := 3 // Greedy sampler should pick highest logit
	if got != want {
		t.Errorf("got %v, want %v", got, want)
	}

	errMock := &testTransform{
		returnErr: fmt.Errorf("mock error"),
	}
	sampler = NewSampler([]Transform{mock1, errMock, mock2}, Greedy())
	_, err = sampler.Sample(input)
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

func TestSampleTemperatureZero(t *testing.T) {
	sampler := NewSampler([]Transform{Temperature(0)}, Greedy())
	got, err := sampler.Sample([]float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	want := 3 // Greedy sampler should pick highest logit index
	if got != want {
		t.Fatalf("got: %v, want: %v", got, want)
	}
}
