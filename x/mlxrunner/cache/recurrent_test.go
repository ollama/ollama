package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// TestRecurrentCacheRestoreDirectionality verifies that RecurrentCache only
// allows restoring forward (target >= snapshot offset), never backward.
func TestRecurrentCacheRestoreDirectionality(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	_ = c.ConvState(1, mlx.DTypeFloat16)
	_ = c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(10)

	snap := c.Snapshot(0)

	c.Advance(5) // now at 15

	// Restore backward should fail.
	if c.Restore(snap, 5) {
		t.Fatal("Restore(snap, 5) should fail — target < snap.offset")
	}

	// Restore to exact snap offset should succeed.
	if !c.Restore(snap, 10) {
		t.Fatal("Restore(snap, 10) should succeed")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}

	// Restore forward (target > snap offset) should succeed, offset = snap.offset.
	snap2 := c.Snapshot(0)
	if !c.Restore(snap2, 15) {
		t.Fatal("Restore(snap, 15) should succeed")
	}
	// Recurrent state is at snap.offset (10), not target (15).
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10 (snap offset)", c.Offset())
	}
}
