package qwen3next

import (
	"slices"
	"strings"
	"testing"
)

func TestInferRecurrentLayersMixedKVArray(t *testing.T) {
	got, err := inferRecurrentLayers([]uint64{0, 2, 0, 2}, 4, 0)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	want := []bool{true, false, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestInferRecurrentLayersScalarKVDefaultInterval(t *testing.T) {
	got, err := inferRecurrentLayers([]uint64{2, 2, 2, 2, 2, 2, 2, 2}, 8, 0)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	want := []bool{true, true, true, false, true, true, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestInferRecurrentLayersScalarKVConfiguredInterval(t *testing.T) {
	got, err := inferRecurrentLayers([]uint64{2, 2, 2, 2, 2, 2}, 6, 3)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	want := []bool{true, true, false, true, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestInferRecurrentLayersAllZeroRejects(t *testing.T) {
	_, err := inferRecurrentLayers([]uint64{0, 0, 0, 0}, 4, 0)
	if err == nil {
		t.Fatal("inferRecurrentLayers() expected error, got nil")
	}
	if !strings.Contains(err.Error(), "must include at least one non-zero value") {
		t.Fatalf("unexpected error = %v", err)
	}
}

