package server

import (
	"testing"
)

func TestAvailableBytes(t *testing.T) {
	dir := t.TempDir()
	avail, err := availableBytes(dir)
	if err != nil {
		t.Fatalf("availableBytes(%q) returned error: %v", dir, err)
	}
	if avail <= 0 {
		t.Fatalf("availableBytes(%q) = %d, want > 0", dir, avail)
	}
}
