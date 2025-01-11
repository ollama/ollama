package sample

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestTemperature(t *testing.T) {
	logits, err := Temperature(0.5).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedlogits := []float64{-6, -4, -2, 0, 2, 4, 8}
	if !floats.Equal(logits, expectedlogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedlogits)
	}

	// Only expect the max value returned
	logits, err = Temperature(0).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedlogits = []float64{4}
	if !floats.Equal(logits, expectedlogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedlogits)
	}

	if _, err := Temperature(-1).Sample([]float64{-3, -2, -1, 0, 1, 2, 4}); err == nil {
		t.Fatalf("expected error for temperature=-1, got %v", logits)
	}
}

func TestSoftmax(t *testing.T) {
	probs, err := computeSoftmax([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}

	expectedProbs := []float64{0.000751406628089903, 0.0020425349829204676, 0.005552185728064613, 0.015092405572827691, 0.04102541181635154, 0.11151863144543739, 0.8240174238263085}
	if !floats.Equal(probs, expectedProbs) {
		t.Fatalf("logits: %v, expectedlogits: %v", probs, expectedProbs)
	}
}

func TestTopK(t *testing.T) {
	logits, err := TopK(3).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedlogits := []float64{math.NaN(), math.NaN(), math.NaN(), math.NaN(), 1, 2, 4}
	if !floats.Same(logits, expectedlogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedlogits)
	}
	logits, err = TopK(0).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Fatalf("expected error for k=0, got %v", logits)
	}
}

func TestTopP(t *testing.T) {
	logits, err := TopP(0.9).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedLogits := []float64{math.NaN(), math.NaN(), math.NaN(), math.NaN(), math.NaN(), 2, 4}
	if !floats.Same(logits, expectedLogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedLogits)
	}
	logits, err = TopP(1.0).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Fatalf("expected error for p=1.0, got %v", logits)
	}
	logits, err = TopP(0.0).Sample([]float64{-3, -2, -1, 0, 1, 2, 4})
	if err == nil {
		t.Fatalf("expected error for p=0.0, got %v", logits)
	}
}

func TestMinP(t *testing.T) {
	logits, err := MinP(0.2).Sample([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	expectedLogits := []float64{math.NaN(), math.NaN(), math.NaN(), math.NaN(), math.NaN(), math.NaN(), 3, 4}
	if !floats.Same(logits, expectedLogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedLogits)
	}
	logits, err = MinP(1.0).Sample([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Fatalf("expected error for p=1.0, got %v", logits)
	}
	logits, err = MinP(0.0).Sample([]float64{-3, -2, -1, 0, 1, 2, 3, 4})
	if err == nil {
		t.Fatalf("expected error for p=0.0, got %v", logits)
	}
}

func TestWeighed(t *testing.T) {
	logits, err := Weighed().Sample([]float64{math.NaN(), 2, math.NaN(), math.NaN()})
	if err != nil {
		t.Fatal(err)
	}
	expectedLogits := []float64{1}
	if !floats.Equal(logits, expectedLogits) {
		t.Fatalf("logits: %v, expectedlogits: %v", logits, expectedLogits)
	}
	logits, err = Weighed().Sample([]float64{math.NaN(), math.NaN(), math.NaN()})
	if err == nil {
		t.Fatalf("expected error for no valid tokens, got %v", logits)
	}
}

func TestSample(t *testing.T) {
	input := []float64{1, 2, 3, 4}
	expectedOutput := []float64{1, 2, 3, 4}

	var callOrder []int
	mock1 := &mockSampler{
		id:         1,
		callOrder:  &callOrder,
		returnVals: expectedOutput,
	}
	mock2 := &mockSampler{
		id:         2,
		callOrder:  &callOrder,
		returnVals: expectedOutput,
	}
	mock3 := &mockSampler{
		id:         3,
		callOrder:  &callOrder,
		returnVals: expectedOutput,
	}

	result, err := Sample(input, mock1, mock2, mock3)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(callOrder, []int{1, 2, 3}) {
		t.Errorf("Expected call order [1,2,3], got %v", callOrder)
	}

	if !floats.Equal(result, expectedOutput) {
		t.Errorf("Expected output %v, got %v", expectedOutput, result)
	}

	errMock := &mockSampler{
		returnErr: fmt.Errorf("mock error"),
	}
	_, err = Sample(input, mock1, errMock, mock2)
	if err == nil {
		t.Error("Expected error from sampler")
	}
}

type mockSampler struct {
	id         int
	callOrder  *[]int
	returnVals []float64
	returnErr  error
}

func (m *mockSampler) Sample(logits []float64) ([]float64, error) {
	if m.callOrder != nil {
		*m.callOrder = append(*m.callOrder, m.id)
	}
	if m.returnErr != nil {
		return nil, m.returnErr
	}
	return m.returnVals, nil
}
