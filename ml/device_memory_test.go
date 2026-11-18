package ml

import "testing"

func TestAvailableMemory(t *testing.T) {
	const gib = 1 << 30

	for _, tt := range []struct {
		name       string
		device     DeviceInfo
		systemFree uint64
		want       uint64
	}{
		{"discrete GPU ignores host memory", DeviceInfo{FreeMemory: 20 * gib}, 8 * gib, 20 * gib},
		{"integrated GPU capped at host memory", DeviceInfo{Integrated: true, FreeMemory: 87 * gib}, 20 * gib, 20 * gib},
		{"integrated GPU below host memory", DeviceInfo{Integrated: true, FreeMemory: 10 * gib}, 20 * gib, 10 * gib},
		{"unknown host memory leaves the figure", DeviceInfo{Integrated: true, FreeMemory: 87 * gib}, 0, 87 * gib},
		{"memory budget ignores host memory", DeviceInfo{Integrated: true, MemoryBudget: true, FreeMemory: 60 * gib}, 20 * gib, 60 * gib},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.device.AvailableMemory(tt.systemFree); got != tt.want {
				t.Fatalf("AvailableMemory = %d GiB, want %d GiB", got/gib, tt.want/gib)
			}
		})
	}
}
