package turboquant

import (
	"testing"
)

func TestSplitOutlierChannelsBasic(t *testing.T) {
	values := []float32{0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6}
	split := SplitOutlierChannels(values, 3)

	// Top-3 by abs magnitude: indices 1(0.9), 3(0.8), 5(0.7)
	wantOutlierIdx := []uint16{1, 3, 5}
	wantRegularIdx := []uint16{0, 2, 4, 6, 7}

	if len(split.OutlierIndices) != len(wantOutlierIdx) {
		t.Fatalf("outlier count: got %d want %d", len(split.OutlierIndices), len(wantOutlierIdx))
	}
	for i, idx := range wantOutlierIdx {
		if split.OutlierIndices[i] != idx {
			t.Errorf("outlier[%d]: got %d want %d", i, split.OutlierIndices[i], idx)
		}
		if split.OutlierValues[i] != values[idx] {
			t.Errorf("outlierVal[%d]: got %v want %v", i, split.OutlierValues[i], values[idx])
		}
	}

	if len(split.RegularIndices) != len(wantRegularIdx) {
		t.Fatalf("regular count: got %d want %d", len(split.RegularIndices), len(wantRegularIdx))
	}
	for i, idx := range wantRegularIdx {
		if split.RegularIndices[i] != idx {
			t.Errorf("regular[%d]: got %d want %d", i, split.RegularIndices[i], idx)
		}
		if split.RegularValues[i] != values[idx] {
			t.Errorf("regularVal[%d]: got %v want %v", i, split.RegularValues[i], values[idx])
		}
	}
}

func TestSplitOutlierChannelsCoversAll(t *testing.T) {
	// Every channel index must appear exactly once across outlier + regular.
	values := []float32{3, 1, 4, 1, 5, 9, 2, 6}
	for outlierCount := 0; outlierCount <= len(values); outlierCount++ {
		split := SplitOutlierChannels(values, outlierCount)
		seen := make(map[uint16]int)
		for _, idx := range split.OutlierIndices {
			seen[idx]++
		}
		for _, idx := range split.RegularIndices {
			seen[idx]++
		}
		for i := range values {
			if seen[uint16(i)] != 1 {
				t.Errorf("outlierCount=%d: channel %d appears %d times", outlierCount, i, seen[uint16(i)])
			}
		}
		if len(split.OutlierIndices)+len(split.RegularIndices) != len(values) {
			t.Errorf("outlierCount=%d: total channels %d != %d", outlierCount, len(split.OutlierIndices)+len(split.RegularIndices), len(values))
		}
	}
}

func TestSplitOutlierChannelsZeroOutliers(t *testing.T) {
	values := []float32{1, 2, 3, 4}
	split := SplitOutlierChannels(values, 0)
	if len(split.OutlierIndices) != 0 {
		t.Errorf("expected 0 outliers, got %d", len(split.OutlierIndices))
	}
	if len(split.RegularIndices) != 4 {
		t.Errorf("expected 4 regular, got %d", len(split.RegularIndices))
	}
}

func TestSplitOutlierChannelsAllOutliers(t *testing.T) {
	values := []float32{1, 2, 3, 4}
	split := SplitOutlierChannels(values, len(values))
	if len(split.OutlierIndices) != 4 {
		t.Errorf("expected 4 outliers, got %d", len(split.OutlierIndices))
	}
	if len(split.RegularIndices) != 0 {
		t.Errorf("expected 0 regular, got %d", len(split.RegularIndices))
	}
}

func TestSplitOutlierChannelsNegativeValues(t *testing.T) {
	// Negative values: magnitude matters, not sign.
	values := []float32{-5, 1, -3, 2}
	split := SplitOutlierChannels(values, 2)
	// Top-2 by abs: index 0 (5.0), index 2 (3.0)
	if split.OutlierIndices[0] != 0 || split.OutlierIndices[1] != 2 {
		t.Errorf("expected outlier indices [0,2], got %v", split.OutlierIndices)
	}
}

func TestSplitOutlierChannelsTieBreakerByIndex(t *testing.T) {
	// Equal magnitudes: lower index wins as outlier (tie-break index ascending).
	values := []float32{1, 1, 1, 1}
	split := SplitOutlierChannels(values, 2)
	// Should pick indices 0 and 1 as outliers.
	if split.OutlierIndices[0] != 0 || split.OutlierIndices[1] != 1 {
		t.Errorf("expected outlier indices [0,1], got %v", split.OutlierIndices)
	}
}

func TestSplitOutlierChannelsIndicesSorted(t *testing.T) {
	// Output indices must be sorted ascending in both slices.
	values := pseudoRandomVector(16, 42)
	split := SplitOutlierChannels(values, 5)
	for i := 1; i < len(split.OutlierIndices); i++ {
		if split.OutlierIndices[i] <= split.OutlierIndices[i-1] {
			t.Errorf("outlier indices not sorted at %d: %d <= %d", i, split.OutlierIndices[i], split.OutlierIndices[i-1])
		}
	}
	for i := 1; i < len(split.RegularIndices); i++ {
		if split.RegularIndices[i] <= split.RegularIndices[i-1] {
			t.Errorf("regular indices not sorted at %d: %d <= %d", i, split.RegularIndices[i], split.RegularIndices[i-1])
		}
	}
}

func TestSplitOutlierChannelsDeterministic(t *testing.T) {
	values := pseudoRandomVector(32, 99)
	a := SplitOutlierChannels(values, 8)
	b := SplitOutlierChannels(values, 8)
	if len(a.OutlierIndices) != len(b.OutlierIndices) {
		t.Fatal("non-deterministic outlier count")
	}
	for i := range a.OutlierIndices {
		if a.OutlierIndices[i] != b.OutlierIndices[i] {
			t.Fatalf("non-deterministic at outlier index %d", i)
		}
	}
}
