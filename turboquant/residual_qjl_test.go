package turboquant

import "testing"

func TestResidualSketchDeterministic(t *testing.T) {
	a := encodeResidual([]float32{1, 2, 3, 4}, []float32{0, 0, 0, 0}, PresetTQ2, 99)
	b := encodeResidual([]float32{1, 2, 3, 4}, []float32{0, 0, 0, 0}, PresetTQ2, 99)
	if a.Scale != b.Scale || a.Seed != b.Seed || string(a.Signs) != string(b.Signs) {
		t.Fatal("residual sketch is not deterministic")
	}
}

func TestReconstructResidualLength(t *testing.T) {
	sketch := encodeResidual(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
		[]float32{0, 0, 0, 0, 0, 0, 0, 0},
		PresetTQ3,
		123,
	)
	reconstructed := reconstructResidual(8, sketch)
	if len(reconstructed) != 8 {
		t.Fatalf("reconstructed length = %d, want 8", len(reconstructed))
	}
	if len(sketch.Signs) != expectedPackedBytes(int(sketch.SketchDim), 1) {
		t.Fatalf("sign sketch bytes = %d, want %d", len(sketch.Signs), expectedPackedBytes(int(sketch.SketchDim), 1))
	}
}

func TestZeroResidualProducesZeroReconstruction(t *testing.T) {
	sketch := encodeResidual(
		[]float32{1, 1, 1, 1},
		[]float32{1, 1, 1, 1},
		PresetTQ3,
		456,
	)
	reconstructed := reconstructResidual(4, sketch)
	for i, value := range reconstructed {
		if abs32(value) > 1e-6 {
			t.Fatalf("reconstructed[%d] = %v, want 0", i, value)
		}
	}
}

func TestResidualDotCorrectionDeterministicAndFinite(t *testing.T) {
	rotated := []float32{1.5, -2.5, 0.5, 3.0, -1.25, 2.25, 0.75, -0.5}
	approx := []float32{1.0, -2.0, 0.25, 2.5, -1.0, 2.0, 0.5, -0.25}
	sketch := encodeResidual(rotated, approx, PresetTQ3, 789)
	query := []float32{0.2, -0.1, 0.3, 0.4, -0.2, 0.5, -0.6, 0.7}

	a := residualDotCorrection(query, sketch)
	b := residualDotCorrection(query, sketch)
	if a != b {
		t.Fatalf("dot correction = %v and %v, want deterministic output", a, b)
	}
	if a != a {
		t.Fatal("dot correction returned NaN")
	}
}
