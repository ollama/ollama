package discover

import "testing"

func TestMemoryMonitorFirstReadEstablishesBaseline(t *testing.T) {
	m, err := NewMemoryMonitor()
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.observed {
		t.Fatal("constructor consumed the first pressure sample")
	}
	got, err := m.Read()
	if err != nil {
		t.Fatal(err)
	}
	if !m.observed || got.ReclaimBytes != 0 || got.FreeMemoryAboveTarget == nil || got.RAMLimit <= 0 {
		t.Fatalf("first observation did not establish the Mac baseline: %+v", got)
	}
}

func TestCompressionObservationTracksChanges(t *testing.T) {
	s := &MemoryMonitor{}
	for i, tc := range []struct{ current, want int64 }{{10, 0}, {20, 10}, {5, 0}, {12, 7}, {12, 0}} {
		if got := s.observePressure(tc.current, 0); got != tc.want {
			t.Fatalf("observation %d: new compressed bytes = %d, want %d", i, got, tc.want)
		}
	}
}

func TestMemoryMonitorsHaveIndependentBaselines(t *testing.T) {
	a, _ := NewMemoryMonitor()
	b, _ := NewMemoryMonitor()
	defer a.Close()
	defer b.Close()
	if a.observePressure(100, 0) != 0 || b.observePressure(150, 0) != 0 {
		t.Fatal("existing compression was reported as new pressure")
	}
	if got := a.observePressure(160, 0); got != 60 {
		t.Fatalf("first consumer lost its baseline: got %d, want 60", got)
	}
	if got := b.observePressure(160, 0); got != 10 {
		t.Fatalf("second consumer lost its baseline: got %d, want 10", got)
	}
	if a.observePressure(160, 0) != 0 || b.observePressure(160, 0) != 0 {
		t.Fatal("repeated observation replayed pressure")
	}
}

func TestSwapOutPressureIsReportedOnce(t *testing.T) {
	m := &MemoryMonitor{}
	for i, tc := range []struct{ compressed, swappedOut, want int64 }{
		{100, 200, 0},  // Existing pressure is baseline, not a new event.
		{110, 200, 10}, // Compression increased without swapping.
		{105, 230, 30}, // Swapping can rise even as compression falls.
		{115, 240, 20}, // Both signals can rise between samples.
		{115, 240, 0},  // An unchanged sample must not replay pressure.
		{100, 10, 0},   // A reset counter starts a new baseline.
		{100, 17, 7},
	} {
		if got := m.observePressure(tc.compressed, tc.swappedOut); got != tc.want {
			t.Fatalf("observation %d: got %d bytes, want %d", i, got, tc.want)
		}
	}
}
