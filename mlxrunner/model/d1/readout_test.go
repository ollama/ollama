package d1

import (
	"math"
	"slices"
	"testing"
)

func TestLoadCalibration(t *testing.T) {
	cal, err := loadCalibration([]byte(`{"default": 1.337398453966105, "temperatures": {}, "tables": {"found:text": [[0.1, 0.2]]}}`))
	if err != nil || cal.Default != 1.337398453966105 {
		t.Errorf("calibration = %+v, %v; want default 1.337398453966105", cal, err)
	}
	if cal, err := loadCalibration([]byte(`{"temperatures": {}}`)); err != nil || cal.Default != 1 {
		t.Errorf("calibration without a default = %+v, %v; want 1", cal, err)
	}
	if _, err := loadCalibration([]byte(`{"default": "hot"}`)); err == nil {
		t.Error("expected an error for a non-numeric default")
	}
}

func TestSoftmax(t *testing.T) {
	near := func(a, b float64) bool { return math.Abs(a-b) <= 1e-15 }
	for _, tc := range []struct {
		scores []float64
		t      float64
		want   []float64
	}{
		{[]float64{2, 1, 0}, 1, []float64{0.6652409557748218, 0.24472847105479764, 0.09003057317038046}},
		{[]float64{2, 1, 0}, 1.337398453966105, []float64{0.5890688475848814, 0.27889148004226244, 0.13203967237285616}},
		{[]float64{-1000, -1000}, 2, []float64{0.5, 0.5}},
	} {
		if got := softmax(tc.scores, tc.t); !slices.EqualFunc(got, tc.want, near) {
			t.Errorf("softmax(%v, %v) = %v, want %v", tc.scores, tc.t, got, tc.want)
		}
	}
}
