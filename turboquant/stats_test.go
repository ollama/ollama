package turboquant

import (
	"math"
	"testing"
)

func TestCompareIdenticalVectors(t *testing.T) {
	values := []float32{1, -2, 3, -4}
	stats := Compare(values, values)
	if stats.MSE != 0 || stats.RMSE != 0 || stats.MeanAbsErr != 0 || stats.MaxAbsErr != 0 {
		t.Fatalf("unexpected non-zero stats: %+v", stats)
	}
}

func TestCompareKnownExample(t *testing.T) {
	reference := []float32{1, 2, 3}
	approx := []float32{2, 0, 3}
	stats := Compare(reference, approx)

	if abs32(stats.MSE-float32(5.0/3.0)) > 1e-6 {
		t.Fatalf("MSE = %v, want %v", stats.MSE, float32(5.0/3.0))
	}
	if abs32(stats.MeanAbsErr-1) > 1e-6 {
		t.Fatalf("MeanAbsErr = %v, want 1", stats.MeanAbsErr)
	}
	if abs32(stats.MaxAbsErr-2) > 1e-6 {
		t.Fatalf("MaxAbsErr = %v, want 2", stats.MaxAbsErr)
	}
}

func TestCompareRMSEMatchesSqrtMSE(t *testing.T) {
	reference := []float32{1, 2, 3}
	approx := []float32{2, 0, 3}
	stats := Compare(reference, approx)

	want := float32(math.Sqrt(float64(stats.MSE)))
	if abs32(stats.RMSE-want) > 1e-6 {
		t.Fatalf("RMSE = %v, want %v", stats.RMSE, want)
	}
}
