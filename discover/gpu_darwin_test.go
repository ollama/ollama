package discover

import "testing"

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
