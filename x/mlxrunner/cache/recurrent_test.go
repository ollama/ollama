package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// TestRecurrentCacheRestoreExactOffset verifies that RecurrentCache restore
// only succeeds when target exactly matches the snapshot's offset. Recurrent
// state is cumulative, so it can't be rewound or fast-forwarded.
func TestRecurrentCacheRestoreExactOffset(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	c.SetSeqs([]int{0})

	b := &batch.ForwardBatch{SeqIDs: []int{0}, SeqLens: []int{1}}
	_, _ = c.ConvState(b, mlx.DTypeFloat16)
	_, _ = c.DeltaState(b, mlx.DTypeFloat16)
	c.Advance(&batch.ForwardBatch{SeqIDs: []int{0}, SeqLens: []int{10}})

	snap := c.Snapshot(0, 0) // snap.offset == 10

	c.Advance(&batch.ForwardBatch{SeqIDs: []int{0}, SeqLens: []int{5}}) // cache now at 15

	// target < snap.offset: fails (can't rewind past snapshot)
	if c.Restore(0, snap, 5) {
		t.Fatal("Restore(snap, 5) should fail — target != snap.offset")
	}

	// target > snap.offset: fails (can't advance without feeding tokens)
	if c.Restore(0, snap, 15) {
		t.Fatal("Restore(snap, 15) should fail — target != snap.offset")
	}

	// target == snap.offset: succeeds
	if !c.Restore(0, snap, 10) {
		t.Fatal("Restore(snap, 10) should succeed — target == snap.offset")
	}
	if int(c.Offsets(0)[0]) != 10 {
		t.Fatalf("offset = %d, want 10", int(c.Offsets(0)[0]))
	}
}
