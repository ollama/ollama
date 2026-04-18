package turboquant

import (
	"math"
	"testing"
)

func TestQJLProjectionIsDeterministic(t *testing.T) {
	dim := 32
	seed := uint64(99999)

	s1 := GenerateQJLProjection(dim, seed)
	s2 := GenerateQJLProjection(dim, seed)

	for i := range s1 {
		if s1[i] != s2[i] {
			t.Fatalf("index %d: %f != %f", i, s1[i], s2[i])
		}
	}
}

func TestQJLProjectionIsApproxGaussian(t *testing.T) {
	dim := 64
	s := GenerateQJLProjection(dim, TurboQuantSeed)

	var sum, sumSq float64
	n := float64(dim * dim)

	for _, v := range s {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}

	mean := sum / n
	variance := sumSq/n - mean*mean

	if math.Abs(mean) > 3.0/math.Sqrt(n) {
		t.Errorf("mean = %f, expected near 0 (within 3 sigma)", mean)
	}

	if math.Abs(variance-1.0) > 0.3 {
		t.Errorf("variance = %f, expected near 1.0", variance)
	}
}

func TestQJLProjectionUsesDistinctSeedFromRotation(t *testing.T) {
	dim := 16
	seed := TurboQuantSeed

	rot := GenerateRotation(dim, seed)
	qjl := GenerateQJLProjection(dim, seed)

	// The QJL projection must be independent from the rotation matrix.
	// They use different seed derivations, so their entries should differ.
	same := 0
	for i := range rot {
		if rot[i] == qjl[i] {
			same++
		}
	}

	// With distinct random matrices, very few entries should match
	if float64(same)/float64(len(rot)) > 0.1 {
		t.Errorf("%d/%d entries identical between rotation and QJL (expected independent)", same, len(rot))
	}
}

func TestQJLDequantScaleMatchesPaper(t *testing.T) {
	// Paper Definition 1: dequant uses factor sqrt(pi/2) / d
	scale := QJLDequantScale(128)
	expected := math.Sqrt(math.Pi/2.0) / 128.0

	if math.Abs(scale-expected) > 1e-15 {
		t.Errorf("scale = %e, want %e", scale, expected)
	}
}
