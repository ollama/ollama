package mlx

// KVHistory carries sequence metadata alongside K/V buffers for SDPA.
// Page table and seq lens travel together — SDPA always needs both.
type KVHistory struct {
	// PageTable maps (seqIdx, position) → slot index in the K/V buffer.
	// Shape: [numSeqs, maxSeqLen], int32. Unused entries are 0.
	PageTable *Array

	// SeqLens is the history length per sequence (number of valid
	// entries in each row of PageTable).
	SeqLens []int
}
