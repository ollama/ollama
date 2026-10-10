//go:build !windows

package compatmigrate

import "testing"

func TestSaturatingCount(t *testing.T) {
	t.Parallel()
	if got := saturatingCount(uint64(3)); got != 3 {
		t.Fatalf("uint64 count = %d, want 3", got)
	}
	if got := saturatingCount(int64(4)); got != 4 {
		t.Fatalf("positive int64 count = %d, want 4", got)
	}
	if got := saturatingCount(int64(-1)); got != 0 {
		t.Fatalf("negative int64 count = %d, want 0", got)
	}
}
