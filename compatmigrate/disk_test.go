package compatmigrate

import (
	"testing"
)

func TestAvailableSpace(t *testing.T) {
	space, err := availableSpace(t.TempDir())
	if err != nil {
		t.Fatalf("availableSpace failed: %v", err)
	}
	if space == 0 {
		t.Fatalf("expected non-zero available space")
	}
}
