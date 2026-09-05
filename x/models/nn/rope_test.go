package nn

import (
	"math"
	"testing"
)

func TestYarnRopeFreqValues(t *testing.T) {
	freqs, scale := YarnRopeFreqValues(8, 10000, &RopeParameters{
		Factor:                        2,
		OriginalMaxPositionEmbeddings: 4096,
		BetaFast:                      32,
		BetaSlow:                      1,
	})
	want := []float32{1, 10, 133.33333, 2000}
	if len(freqs) != len(want) {
		t.Fatalf("len(freqs) = %d, want %d", len(freqs), len(want))
	}
	for i := range want {
		if math.Abs(float64(freqs[i]-want[i])) > 1e-4 {
			t.Fatalf("freqs[%d] = %v, want %v", i, freqs[i], want[i])
		}
	}
	wantScale := float32(0.1*math.Log(2) + 1)
	if math.Abs(float64(scale-wantScale)) > 1e-6 {
		t.Fatalf("scale = %v, want %v", scale, wantScale)
	}
}

func TestYarnRopeFreqValuesHonorsMScaleAndTruncate(t *testing.T) {
	truncate := false
	freqs, scale := YarnRopeFreqValues(8, 10000, &RopeParameters{
		Factor:                        2,
		OriginalMaxPositionEmbeddings: 4096,
		BetaFast:                      32,
		BetaSlow:                      1,
		MScale:                        2,
		MScaleAllDim:                  1,
		Truncate:                      &truncate,
	})
	want := []float32{1, 10, 129.79179, 2000}
	for i := range want {
		if math.Abs(float64(freqs[i]-want[i])) > 1e-4 {
			t.Fatalf("freqs[%d] = %v, want %v", i, freqs[i], want[i])
		}
	}
	if math.Abs(float64(scale-1.0648216)) > 1e-6 {
		t.Fatalf("scale = %v, want 1.0648216", scale)
	}
}
