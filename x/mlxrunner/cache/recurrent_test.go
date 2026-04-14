package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// TestRecurrentCacheRestoreExactOffset verifies that RecurrentCache restore
// only succeeds when target exactly matches the snapshot's offset. Recurrent
// state is cumulative, so it can't be rewound or fast-forwarded.
func TestRecurrentCacheRestoreExactOffset(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	_ = c.ConvState(1, mlx.DTypeFloat16)
	_ = c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(10)

	snap := c.Snapshot(0) // snap.offset == 10

	c.Advance(5) // cache now at 15

	// target < snap.offset: fails (can't rewind past snapshot)
	if c.Restore(snap, 5) {
		t.Fatal("Restore(snap, 5) should fail — target != snap.offset")
	}

	// target > snap.offset: fails (can't advance without feeding tokens)
	if c.Restore(snap, 15) {
		t.Fatal("Restore(snap, 15) should fail — target != snap.offset")
	}

	// target == snap.offset: succeeds
	if !c.Restore(snap, 10) {
		t.Fatal("Restore(snap, 10) should succeed — target == snap.offset")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}
}
