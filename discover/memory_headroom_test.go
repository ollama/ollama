//go:build !darwin && !linux && !windows

package discover

import "testing"

func TestMemoryMonitorPreservesCPUFallback(t *testing.T) {
	m, err := NewMemoryMonitor()
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Changes() != nil {
		t.Fatal("CPU fallback unexpectedly owns notifications")
	}
	for range 2 {
		got, err := m.Read()
		if err != nil {
			t.Fatal(err)
		}
		if got.RAMLimit <= 0 || got.AvailableRAM < 0 || got.SampledAt.IsZero() {
			t.Fatalf("invalid CPU memory observation: %+v", got)
		}
		if got.ReclaimBytes != 0 || got.Pressure != PressureUnknown || got.AvailableCommit != nil || got.FreeMemoryAboveTarget != nil {
			t.Fatalf("CPU fallback gained a pressure policy: %+v", got)
		}
	}
	m.Close()
}
