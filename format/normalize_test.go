package format

import (
	"math"
	"testing"
)

func TestNormalize(t *testing.T) {
	type testCase struct {
		input []float32
	}

	testCases := []testCase{
		{input: []float32{1}},
		{input: []float32{0, 1, 2, 3}},
		{input: []float32{0.1, 0.2, 0.3}},
		{input: []float32{-0.1, 0.2, 0.3, -0.4}},
		{input: []float32{0, 0, 0}},
	}

	assertNorm := func(vec []float32) (res bool) {
		sum := 0.0
		for _, v := range vec {
			sum += float64(v * v)
		}
		if math.Abs(sum-1) > 1e-6 {
			return sum == 0
		} else {
			return true
		}
	}

	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			normalized := Normalize(tc.input)
			if !assertNorm(normalized) {
				t.Errorf("Vector %v is not normalized", tc.input)
			}
		})
	}
}
