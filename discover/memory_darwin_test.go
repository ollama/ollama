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
		if got := s.observeCompression(tc.current); got != tc.want {
			t.Fatalf("observation %d: new compressed bytes = %d, want %d", i, got, tc.want)
		}
	}
}

func TestMemoryMonitorsHaveIndependentBaselines(t *testing.T) {
	a, _ := NewMemoryMonitor()
	b, _ := NewMemoryMonitor()
	defer a.Close()
	defer b.Close()
	if a.observeCompression(100) != 0 || b.observeCompression(150) != 0 {
		t.Fatal("existing compression was reported as new pressure")
	}
	if got := a.observeCompression(160); got != 60 {
		t.Fatalf("first consumer lost its baseline: got %d, want 60", got)
	}
	if got := b.observeCompression(160); got != 10 {
		t.Fatalf("second consumer lost its baseline: got %d, want 10", got)
	}
	if a.observeCompression(160) != 0 || b.observeCompression(160) != 0 {
		t.Fatal("repeated observation replayed pressure")
	}
}
