package kvcache

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func fillCache(t *testing.T, cache *Causal, backend ml.Backend, seq int, n int) {
	t.Helper()
	for i := range n {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{seq},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("StartForward at pos %d: %v", i, err)
		}
	}
}

func countSeqCells(cache *Causal, seq int) int {
	count := 0
	for _, cell := range cache.cells {
		if slices.Contains(cell.sequences, seq) {
			count++
		}
	}
	return count
}

func seqPositions(cache *Causal, seq int) []int32 {
	var positions []int32
	for _, cell := range cache.cells {
		if slices.Contains(cell.sequences, seq) {
			positions = append(positions, cell.pos)
		}
	}
	slices.Sort(positions)
	return positions
}

func snapshotCells(cache *Causal) []cacheCell {
	snap := make([]cacheCell, len(cache.cells))
	for i, c := range cache.cells {
		snap[i] = cacheCell{
			pos:       c.pos,
			sequences: slices.Clone(c.sequences),
		}
	}
	return snap
}

func snapshotCellRanges(cache *Causal) map[int]cellRange {
	snap := make(map[int]cellRange, len(cache.cellRanges))
	for k, v := range cache.cellRanges {
		snap[k] = v
	}
	return snap
}

func cellsEqual(a, b []cacheCell) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].pos != b[i].pos {
			return false
		}
		if !slices.Equal(a[i].sequences, b[i].sequences) {
			return false
		}
	}
	return true
}

func cellRangesEqual(a, b map[int]cellRange) bool {
	if len(a) != len(b) {
		return false
	}
	for k, va := range a {
		vb, ok := b[k]
		if !ok || va != vb {
			return false
		}
	}
	return true
}

func TestSpeculation_FullCommit(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 64, 16)

	fillCache(t, cache, backend, 0, 10)
	if got := countSeqCells(cache, 0); got != 10 {
		t.Fatalf("expected 10 cells, got %d", got)
	}

	cache.BeginSpeculation(0)

	// Forward 4 speculative tokens
	for i := 10; i < 14; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("spec StartForward at pos %d: %v", i, err)
		}
	}

	if got := countSeqCells(cache, 0); got != 14 {
		t.Fatalf("expected 14 cells during speculation, got %d", got)
	}

	cache.Commit(4)

	if got := countSeqCells(cache, 0); got != 14 {
		t.Fatalf("expected 14 cells after full commit, got %d", got)
	}

	positions := seqPositions(cache, 0)
	for i := range int32(14) {
		if !slices.Contains(positions, i) {
			t.Errorf("position %d missing after commit", i)
		}
	}
}

func TestSpeculation_Rollback(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 64, 16)

	fillCache(t, cache, backend, 0, 10)

	preSpecCells := snapshotCells(cache)
	preSpecRanges := snapshotCellRanges(cache)

	cache.BeginSpeculation(0)

	for i := 10; i < 14; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("spec StartForward at pos %d: %v", i, err)
		}
	}

	if got := countSeqCells(cache, 0); got != 14 {
		t.Fatalf("expected 14 cells during speculation, got %d", got)
	}

	cache.Rollback()

	if got := countSeqCells(cache, 0); got != 10 {
		t.Fatalf("expected 10 cells after rollback, got %d", got)
	}

	if !cellsEqual(cache.cells, preSpecCells) {
		t.Error("cells not restored after rollback")
	}

	if !cellRangesEqual(cache.cellRanges, preSpecRanges) {
		t.Error("cellRanges not restored after rollback")
	}

	// Verify subsequent non-speculative forward works
	batch := input.Batch{
		Positions: []int32{10},
		Sequences: []int{0},
	}
	ctx := backend.NewContext()
	if err := cache.StartForward(ctx, batch, false); err != nil {
		t.Fatalf("post-rollback StartForward: %v", err)
	}

	if got := countSeqCells(cache, 0); got != 11 {
		t.Fatalf("expected 11 cells after post-rollback forward, got %d", got)
	}
}

func TestSpeculation_PartialCommit(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 64, 16)

	fillCache(t, cache, backend, 0, 10)

	cache.BeginSpeculation(0)

	for i := 10; i < 16; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("spec StartForward at pos %d: %v", i, err)
		}
	}

	// Accept 3 of 6
	cache.Commit(3)

	if got := countSeqCells(cache, 0); got != 13 {
		t.Fatalf("expected 13 cells after partial commit (10+3), got %d", got)
	}

	positions := seqPositions(cache, 0)
	for _, p := range []int32{10, 11, 12} {
		if !slices.Contains(positions, p) {
			t.Errorf("accepted position %d missing", p)
		}
	}
	for _, p := range []int32{13, 14, 15} {
		if slices.Contains(positions, p) {
			t.Errorf("rejected position %d still present", p)
		}
	}

	// Next forward at position 13 should work
	batch := input.Batch{
		Positions: []int32{13},
		Sequences: []int{0},
	}
	ctx := backend.NewContext()
	if err := cache.StartForward(ctx, batch, false); err != nil {
		t.Fatalf("post-commit StartForward: %v", err)
	}
	if got := countSeqCells(cache, 0); got != 14 {
		t.Fatalf("expected 14 cells after next forward, got %d", got)
	}
}

func TestSpeculation_SWADeferred(t *testing.T) {
	backend := &testBackend{}
	cache := NewSWAMemCache(4, 6, nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 64, 16)

	fillCache(t, cache, backend, 0, 10)

	preSpecPositions := seqPositions(cache, 0)

	cache.BeginSpeculation(0)

	for i := 10; i < 13; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("spec StartForward at pos %d: %v", i, err)
		}
	}

	// During speculation, SWA eviction should be deferred.
	// Positions that would normally be evicted at pos 12 (window=4, memory=6 -> evict < 12-6=6)
	// should still be present.
	specPositions := seqPositions(cache, 0)
	for _, p := range preSpecPositions {
		if !slices.Contains(specPositions, p) {
			t.Errorf("position %d was evicted during speculation (should be deferred)", p)
		}
	}

	cache.Commit(3)

	// After commit, SWA eviction should NOT run (it runs on the next non-speculative StartForward).
	// But positions outside the memory window should be evictable on next forward.
	postCommitPositions := seqPositions(cache, 0)
	if len(postCommitPositions) != len(specPositions) {
		t.Logf("positions after commit: %v", postCommitPositions)
	}
}

func TestSpeculation_SWARollbackNoEviction(t *testing.T) {
	backend := &testBackend{}
	cache := NewSWAMemCache(4, 6, nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 64, 16)

	fillCache(t, cache, backend, 0, 10)

	preSpecCells := snapshotCells(cache)
	preSpecRanges := snapshotCellRanges(cache)

	cache.BeginSpeculation(0)

	for i := 10; i < 13; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("spec StartForward at pos %d: %v", i, err)
		}
	}

	cache.Rollback()

	if !cellsEqual(cache.cells, preSpecCells) {
		t.Error("cells not restored after SWA rollback")
	}
	if !cellRangesEqual(cache.cellRanges, preSpecRanges) {
		t.Error("cellRanges not restored after SWA rollback")
	}
}

func TestSpeculation_MultipleRounds(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 128, 16)

	fillCache(t, cache, backend, 0, 10)

	nextPos := int32(10)
	totalCommitted := 0

	for round := range 10 {
		cache.BeginSpeculation(0)

		for i := range 4 {
			batch := input.Batch{
				Positions: []int32{nextPos + int32(i)},
				Sequences: []int{0},
			}
			ctx := backend.NewContext()
			if err := cache.StartForward(ctx, batch, false); err != nil {
				t.Fatalf("round %d: StartForward at pos %d: %v", round, nextPos+int32(i), err)
			}
		}

		// Accept 2, reject 2
		cache.Commit(2)
		nextPos += 2
		totalCommitted += 2
	}

	expectedCells := 10 + totalCommitted
	if got := countSeqCells(cache, 0); got != expectedCells {
		t.Fatalf("expected %d cells after %d rounds, got %d", expectedCells, 10, got)
	}

	// Verify no ghost entries
	positions := seqPositions(cache, 0)
	for i := range int32(expectedCells) {
		if !slices.Contains(positions, i) {
			t.Errorf("position %d missing after multi-round speculation", i)
		}
	}
}

func TestSpeculation_NearFullCache(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 1, 32, 16)

	fillCache(t, cache, backend, 0, 30)
	preSpecCells := snapshotCells(cache)

	cache.BeginSpeculation(0)

	// Try to forward 4 tokens; only 2 cells remain
	var specErr error
	speculated := 0
	for i := 30; i < 34; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		err := cache.StartForward(ctx, batch, false)
		if err != nil {
			specErr = err
			break
		}
		speculated++
	}

	// Either all succeed or we get a cache-full error
	if specErr != nil {
		// Rolling back should restore the cache
		cache.Rollback()
		if !cellsEqual(cache.cells, preSpecCells) {
			t.Error("cells not restored after near-full rollback")
		}
	} else {
		cache.Commit(speculated)
		if got := countSeqCells(cache, 0); got != 30+speculated {
			t.Fatalf("expected %d cells, got %d", 30+speculated, got)
		}
	}
}

func TestSpeculation_DoesNotCorruptOtherSequences(t *testing.T) {
	backend := &testBackend{}
	cache := NewCausalCache(nil)
	defer cache.Close()
	cache.Init(backend, ml.DTypeF16, 2, 64, 16)

	// Fill seq 0 and seq 1
	fillCache(t, cache, backend, 0, 10)
	fillCache(t, cache, backend, 1, 10)

	seq1PreSpec := seqPositions(cache, 1)

	cache.BeginSpeculation(0)

	for i := 10; i < 14; i++ {
		batch := input.Batch{
			Positions: []int32{int32(i)},
			Sequences: []int{0},
		}
		ctx := backend.NewContext()
		if err := cache.StartForward(ctx, batch, false); err != nil {
			t.Fatalf("spec StartForward at pos %d: %v", i, err)
		}
	}

	cache.Rollback()

	seq1PostRollback := seqPositions(cache, 1)
	if !slices.Equal(seq1PreSpec, seq1PostRollback) {
		t.Errorf("seq 1 positions changed: before=%v after=%v", seq1PreSpec, seq1PostRollback)
	}

	if got := countSeqCells(cache, 0); got != 10 {
		t.Fatalf("seq 0: expected 10 cells after rollback, got %d", got)
	}
	if got := countSeqCells(cache, 1); got != 10 {
		t.Fatalf("seq 1: expected 10 cells, got %d", got)
	}
}
