package ollamarunner

// continuousBatchingScheduler implements a scheduling policy that dynamically
// composes batches to maximize GPU utilization while maintaining low latency
// for all in-flight sequences.
//
// Policy:
//   - Decode sequences (generating tokens) get 1 slot each, prioritized first
//   - Prefill sequences (processing prompts) share remaining capacity fairly
//   - A single sequence gets the full batch (no overhead)
type continuousBatchingScheduler struct {
	maxBatchSize int
}

// batchPlan describes how many tokens to take from each sequence slot.
type batchPlan struct {
	tokensPerSeq []int // indexed by seqs slot index; 0 = skip, -1 = nil slot
	prefillSeqs  []int // slot indices of prefill sequences
	decodeSeqs   []int // slot indices of decode sequences
	totalTokens  int
}

// plan analyzes the current sequences and returns a token allocation plan.
func (cbs *continuousBatchingScheduler) plan(seqs []*Sequence) *batchPlan {
	bp := &batchPlan{
		tokensPerSeq: make([]int, len(seqs)),
	}

	// Categorize sequences
	for i, seq := range seqs {
		if seq == nil {
			bp.tokensPerSeq[i] = -1
			continue
		}
		if len(seq.inputs) == 0 {
			bp.decodeSeqs = append(bp.decodeSeqs, i)
		} else {
			bp.prefillSeqs = append(bp.prefillSeqs, i)
		}
	}

	numDecode := len(bp.decodeSeqs)
	numPrefill := len(bp.prefillSeqs)
	active := numDecode + numPrefill

	if active == 0 {
		return bp
	}

	// Single sequence: give it the full batch (no overhead)
	if active == 1 {
		if numDecode == 1 {
			bp.tokensPerSeq[bp.decodeSeqs[0]] = 1
			bp.totalTokens = 1
		} else {
			seq := seqs[bp.prefillSeqs[0]]
			bp.tokensPerSeq[bp.prefillSeqs[0]] = min(cbs.maxBatchSize, len(seq.inputs))
			bp.totalTokens = bp.tokensPerSeq[bp.prefillSeqs[0]]
		}
		return bp
	}

	// Multiple active sequences: continuous batching policy
	// Step 1: Reserve 1 token per decode sequence
	reservedDecode := numDecode

	// Step 2: Allocate remaining capacity to prefill sequences
	remainingCapacity := cbs.maxBatchSize - reservedDecode
	if remainingCapacity < 0 {
		// Not enough batch capacity for all decode sequences.
		// This shouldn't happen with reasonable batch sizes, but handle gracefully.
		// Prioritize decode sequences by only taking what fits.
		for _, idx := range bp.decodeSeqs {
			if bp.totalTokens < cbs.maxBatchSize {
				bp.tokensPerSeq[idx] = 1
				bp.totalTokens++
			}
		}
		return bp
	}

	// Step 3: Fair-share remaining capacity among prefill sequences
	var prefillCapacity int
	if numPrefill > 0 {
		prefillCapacity = remainingCapacity
		capPerSeq := prefillCapacity / numPrefill
		if capPerSeq < 1 {
			capPerSeq = 1
		}

		// First pass: allocate up to capPerSeq or available inputs
		allocated := 0
		remainingPrefill := make([]int, 0, numPrefill)
		for _, idx := range bp.prefillSeqs {
			seq := seqs[idx]
			take := min(capPerSeq, len(seq.inputs))
			bp.tokensPerSeq[idx] = take
			allocated += take
			if take < len(seq.inputs) {
				remainingPrefill = append(remainingPrefill, idx)
			}
		}

		// Second pass: distribute any leftover capacity to sequences that still have inputs
		leftover := prefillCapacity - allocated
		for _, idx := range remainingPrefill {
			if leftover <= 0 {
				break
			}
			seq := seqs[idx]
			canTake := len(seq.inputs) - bp.tokensPerSeq[idx]
			take := min(leftover, canTake)
			bp.tokensPerSeq[idx] += take
			leftover -= take
		}

		// If we still have capacity and no prefill seqs need more, it's OK
		bp.totalTokens = prefillCapacity - leftover
	}

	// Step 4: Fill decode slots
	for _, idx := range bp.decodeSeqs {
		bp.tokensPerSeq[idx] = 1
		bp.totalTokens++
	}

	return bp
}

// isDecodeSeq returns true if the sequence at the given index is in decode mode.
func (bp *batchPlan) isDecodeSeq(idx int) bool {
	for _, d := range bp.decodeSeqs {
		if d == idx {
			return true
		}
	}
	return false
}

// isPrefillSeq returns true if the sequence at the given index is in prefill mode.
func (bp *batchPlan) isPrefillSeq(idx int) bool {
	for _, p := range bp.prefillSeqs {
		if p == idx {
			return true
		}
	}
	return false
}
