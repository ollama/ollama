package discover

import "testing"

func TestKernelAvailableMemory(t *testing.T) {
	const physical = 48 << 30
	for _, tc := range []struct {
		name         string
		pageEstimate uint64
		level        uint32
		want         uint64
	}{
		// Free + file-backed pages can be far below what macOS reports as
		// available while inactive memory is reclaimable without pressure.
		{"kernel estimate is larger", 5 << 30, 31, physical * 31 / 100},
		{"page estimate is larger", 20 << 30, 10, 20 << 30},
		{"unknown level", 5 << 30, 0, 5 << 30},
		{"invalid level", 5 << 30, 101, 5 << 30},
		{"all memory available", 5 << 30, 100, physical},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := kernelAvailableMemory(tc.pageEstimate, physical, tc.level); got != tc.want {
				t.Fatalf("available = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestMemoryHeadroom(t *testing.T) {
	for _, tc := range []struct {
		name              string
		available, target uint64
		want              int64
	}{
		{"above OS target", 128 << 20, 64 << 20, 64 << 20},
		{"at OS target", 64 << 20, 64 << 20, 0},
		{"below OS target", 32 << 20, 64 << 20, -32 << 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := memoryHeadroom(tc.available, tc.target); got != tc.want {
				t.Fatalf("headroom = %d, want %d", got, tc.want)
			}
		})
	}
}
