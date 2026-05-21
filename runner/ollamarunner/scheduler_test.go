package ollamarunner

import (
	"testing"

	"github.com/ollama/ollama/model/input"
)

func TestSchedulerSingleSequence(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 512}

	// Single decode sequence
	seqs := []*Sequence{
		{inputs: []*input.Input{}}, // empty inputs = decode mode
	}
	plan := cbs.plan(seqs)
	if plan.tokensPerSeq[0] != 1 {
		t.Errorf("expected 1 token for single decode seq, got %d", plan.tokensPerSeq[0])
	}
	if plan.totalTokens != 1 {
		t.Errorf("expected 1 total token, got %d", plan.totalTokens)
	}

	// Single prefill sequence with many inputs
	inputs := make([]*input.Input, 1000)
	for i := range inputs {
		inputs[i] = &input.Input{Token: int32(i)}
	}
	seqs = []*Sequence{
		{inputs: inputs},
	}
	plan = cbs.plan(seqs)
	if plan.tokensPerSeq[0] != 512 {
		t.Errorf("expected 512 tokens for single prefill seq, got %d", plan.tokensPerSeq[0])
	}
	if plan.totalTokens != 512 {
		t.Errorf("expected 512 total tokens, got %d", plan.totalTokens)
	}

	// Single prefill with few inputs
	seqs = []*Sequence{
		{inputs: inputs[:10]},
	}
	plan = cbs.plan(seqs)
	if plan.tokensPerSeq[0] != 10 {
		t.Errorf("expected 10 tokens for single small prefill seq, got %d", plan.tokensPerSeq[0])
	}
}

func TestSchedulerDecodePriority(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 512}

	// 2 decode + 1 prefill
	seqs := []*Sequence{
		{inputs: []*input.Input{}}, // decode
		nil,
		{inputs: makeInputs(500)}, // prefill with 500 inputs
		{inputs: []*input.Input{}}, // decode
	}
	plan := cbs.plan(seqs)

	// Decode sequences should each get 1 token
	if plan.tokensPerSeq[0] != 1 {
		t.Errorf("decode seq 0: expected 1 token, got %d", plan.tokensPerSeq[0])
	}
	if plan.tokensPerSeq[3] != 1 {
		t.Errorf("decode seq 3: expected 1 token, got %d", plan.tokensPerSeq[3])
	}

	// Nil slot should be -1
	if plan.tokensPerSeq[1] != -1 {
		t.Errorf("nil slot: expected -1, got %d", plan.tokensPerSeq[1])
	}

	// Prefill should get remaining capacity (512 - 2 = 510)
	if plan.tokensPerSeq[2] != 500 {
		t.Errorf("prefill seq 2: expected 500 tokens, got %d", plan.tokensPerSeq[2])
	}

	// Total should match
	if plan.totalTokens != 502 {
		t.Errorf("expected 502 total tokens, got %d", plan.totalTokens)
	}

	// Verify process order: decode first, then prefill
	if len(plan.decodeSeqs) != 2 {
		t.Errorf("expected 2 decode seqs, got %d", len(plan.decodeSeqs))
	}
	if len(plan.prefillSeqs) != 1 {
		t.Errorf("expected 1 prefill seq, got %d", len(plan.prefillSeqs))
	}
	if plan.decodeSeqs[0] != 0 || plan.decodeSeqs[1] != 3 {
		t.Errorf("decode seqs in wrong order: %v", plan.decodeSeqs)
	}
}

func TestSchedulerFairSharing(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 512}

	// 3 prefill sequences, each with 300 inputs
	seqs := []*Sequence{
		{inputs: makeInputs(300)},
		{inputs: makeInputs(300)},
		{inputs: makeInputs(300)},
	}
	plan := cbs.plan(seqs)

	// Each should get ~170 (512/3), not full batch
	for i := 0; i < 3; i++ {
		got := plan.tokensPerSeq[i]
		if got > 200 || got < 150 {
			t.Errorf("prefill seq %d: expected ~170 tokens, got %d", i, got)
		}
	}

	// Total should be close to 512
	if plan.totalTokens > 512 {
		t.Errorf("total tokens %d exceeds batch size 512", plan.totalTokens)
	}
}

func TestSchedulerDecodeOnly(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 512}

	// All decode sequences
	seqs := []*Sequence{
		{inputs: []*input.Input{}},
		{inputs: []*input.Input{}},
		{inputs: []*input.Input{}},
	}
	plan := cbs.plan(seqs)

	for i := 0; i < 3; i++ {
		if plan.tokensPerSeq[i] != 1 {
			t.Errorf("decode seq %d: expected 1 token, got %d", i, plan.tokensPerSeq[i])
		}
	}
	if plan.totalTokens != 3 {
		t.Errorf("expected 3 total tokens, got %d", plan.totalTokens)
	}
}

func TestSchedulerEmpty(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 512}

	// All nil
	seqs := []*Sequence{nil, nil, nil}
	plan := cbs.plan(seqs)

	if plan.totalTokens != 0 {
		t.Errorf("expected 0 total tokens for empty seqs, got %d", plan.totalTokens)
	}
	for i := 0; i < 3; i++ {
		if plan.tokensPerSeq[i] != -1 {
			t.Errorf("nil seq %d: expected -1, got %d", i, plan.tokensPerSeq[i])
		}
	}
}

func TestSchedulerDecodeOverflow(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 4}

	// 6 decode sequences (more than batch size)
	seqs := make([]*Sequence, 6)
	for i := range seqs {
		seqs[i] = &Sequence{inputs: []*input.Input{}}
	}
	plan := cbs.plan(seqs)

	// Should only allocate up to batchSize
	if plan.totalTokens > 4 {
		t.Errorf("total tokens %d exceeds batch size 4", plan.totalTokens)
	}

	// First 4 should get 1 token, last 2 should get 0
	for i := 0; i < 4; i++ {
		if plan.tokensPerSeq[i] != 1 {
			t.Errorf("decode seq %d: expected 1 token, got %d", i, plan.tokensPerSeq[i])
		}
	}
	for i := 4; i < 6; i++ {
		if plan.tokensPerSeq[i] != 0 {
			t.Errorf("overflow decode seq %d: expected 0 tokens, got %d", i, plan.tokensPerSeq[i])
		}
	}
}

func TestSchedulerMixedOverflow(t *testing.T) {
	cbs := &continuousBatchingScheduler{maxBatchSize: 8}

	// 5 decode + 2 prefill (decode takes 5 slots, 3 left for prefill)
	seqs := []*Sequence{
		{inputs: []*input.Input{}},   // decode
		{inputs: makeInputs(100)},    // prefill
		{inputs: []*input.Input{}},   // decode
		{inputs: makeInputs(100)},    // prefill
		{inputs: []*input.Input{}},   // decode
		{inputs: []*input.Input{}},   // decode
		{inputs: []*input.Input{}},   // decode
	}
	plan := cbs.plan(seqs)

	// All 5 decode should get 1 token
	decodeCount := 0
	for _, d := range plan.decodeSeqs {
		if plan.tokensPerSeq[d] != 1 {
			t.Errorf("decode seq %d: expected 1 token, got %d", d, plan.tokensPerSeq[d])
		}
		decodeCount++
	}
	if decodeCount != 5 {
		t.Errorf("expected 5 decode seqs, got %d", decodeCount)
	}

	// Remaining 3 slots for 2 prefill = 1 each
	for _, p := range plan.prefillSeqs {
		if plan.tokensPerSeq[p] < 1 {
			t.Errorf("prefill seq %d: expected >=1 token, got %d", p, plan.tokensPerSeq[p])
		}
	}

	if plan.totalTokens > 8 {
		t.Errorf("total tokens %d exceeds batch size 8", plan.totalTokens)
	}
}

func makeInputs(n int) []*input.Input {
	inputs := make([]*input.Input, n)
	for i := range inputs {
		inputs[i] = &input.Input{Token: int32(i)}
	}
	return inputs
}
