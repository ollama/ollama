package mlxrunner

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// dflashPendingFlushTokens bounds the pinned feature rows between flushes.
const dflashPendingFlushTokens = 256

// dflashDrafter drafts with a block-diffusion draft model (DFlash): one
// forward proposes a whole block. A context-cache entry depends only on its
// own position, so the trie keys need no look-ahead.
type dflashDrafter struct {
	spec      *speculation
	blockSize int
	maskToken int32
}

func newDFlashDrafter(s *speculation, draft base.BlockDraft) *dflashDrafter {
	blockSize, maskToken := draft.BlockParams()
	return &dflashDrafter{spec: s, blockSize: blockSize, maskToken: maskToken}
}

func (d *dflashDrafter) draftLimit() int { return d.blockSize - 1 }

// open returns a session synced to the draft caches' restored offset.
func (d *dflashDrafter) open(layout []any) draftSession {
	s := &dflashDraftSession{drafter: d, layout: layout}
	if kv := d.spec.draftKV; len(kv) > 0 {
		s.ctxOffset = kv[0].Offset()
	}
	return s
}

// dflashDraftSession runs one request's drafting. A context entry at slot S
// derives only from the features at S; ctxOffset+pendingCount is the slot
// after the last reported token.
type dflashDraftSession struct {
	drafter *dflashDrafter
	layout  []any

	// ctxOffset is the slot after the last feature row written; pendingCount
	// rows are buffered past it.
	ctxOffset       int
	pendingFeatures []*mlx.Array
	pendingCount    int

	// blockOutstanding tracks the proposal's scheduled rollback point, which
	// commitBlock has to drain even when it needs no rewind.
	blockOutstanding bool
}

// committed ignores the media manifest: a context row derives from the
// target hidden at its slot, which already carries any image content.
func (d *dflashDraftSession) committed(tokens, features *mlx.Array, position int, _ []batch.MediaItem) {
	n := tokens.Dim(1)
	// Skip leading rows the session already has (a restored prefix). A run
	// that starts past the frontier would leave a gap, which is a bug.
	start := d.ctxOffset + d.pendingCount - position
	if start < 0 {
		panic(fmt.Sprintf("dflash: committed run at %d leaves a context gap at %d", position, d.ctxOffset+d.pendingCount))
	}
	if start < n {
		f := features.Slice(mlx.Slice(), mlx.Slice(start, n), mlx.Slice())
		mlx.Pin(f)
		d.pendingFeatures = append(d.pendingFeatures, f)
		d.pendingCount += n - start
		if d.pendingCount >= dflashPendingFlushTokens {
			d.flush()
		}
	}
}

// settle writes buffered rows through, leveling the draft caches with the
// target's; next is unused.
func (d *dflashDraftSession) settle(_ *mlx.Array) {
	d.flush()
}

func (d *dflashDraftSession) close() {
	d.flush()
}

// takePending returns the buffered rows, advancing ctxOffset past them.
func (d *dflashDraftSession) takePending() *mlx.Array {
	if len(d.pendingFeatures) == 0 {
		return nil
	}
	features := mlx.Concatenate(d.pendingFeatures, 1)
	mlx.Unpin(d.pendingFeatures...)
	d.pendingFeatures = nil
	d.ctxOffset += d.pendingCount
	d.pendingCount = 0
	return features
}

// commitBlock rewinds the round's block out of the draft caches; the caches
// keep only context rows, and accepted tokens arrive as context later. Every
// write path must run this first, otherwise the new rows land after the block
// and the cache contents no longer match their positions.
func (d *dflashDraftSession) commitBlock() {
	if !d.blockOutstanding {
		return
	}
	commitSpeculation(d.drafter.spec.draftKV, 0, 1, d.ctxOffset)
	d.blockOutstanding = false
}

// flush rewinds the round's block, then writes the pending rows in one
// context-only forward. The block is rewound even when nothing is pending.
func (d *dflashDraftSession) flush() {
	spec := d.drafter.spec
	d.commitBlock()

	offset := d.ctxOffset
	features := d.takePending()
	if features == nil {
		return
	}
	spec.draft.Forward(&batch.Batch{
		SeqOffsets: []int32{int32(offset)},
		Hidden:     features,
		Layout:     d.layout,
	}, spec.targets, spec.draftKV)

	// Force the cache writes: a session that never drafts would otherwise
	// leave the flush chain unevaluated, pinning every feature until close.
	state := make([]*mlx.Array, 0, 2*len(spec.draftKV))
	for _, c := range spec.draftKV {
		state = append(state, c.State()...)
	}
	mlx.AsyncEval(state...)
}

// propose drafts a block after the not-yet-validated current token, one
// forward filling every mask position.
func (d *dflashDraftSession) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	spec := d.drafter.spec
	r := spec.r
	blockSize := d.drafter.blockSize

	n := min(maxTokens, blockSize-1)
	if n <= 0 || d.ctxOffset+d.pendingCount == 0 {
		return nil
	}
	d.commitBlock()

	// Send only the anchor and the rows being sampled, not the full trained
	// block. Exact for causal layers, and measured as free for bidirectional
	// ones.
	masks := make([]int32, n)
	for i := range masks {
		masks[i] = d.drafter.maskToken
	}
	block := current.ExpandDims(-1).Concatenate(1, mlx.FromValues(masks, 1, len(masks)))

	offset := d.ctxOffset
	features := d.takePending()

	scheduleSpeculation(spec.draftKV, d.ctxOffset, 1)
	d.blockOutstanding = true

	hidden, _ := spec.draft.Forward(&batch.Batch{
		InputIDs:     block,
		SeqOffsets:   []int32{int32(offset)},
		SeqQueryLens: []int32{int32(n + 1)},
		Hidden:       features,
		Layout:       d.layout,
	}, spec.targets, spec.draftKV)

	// Row i predicts the token at its own position, so the anchor row is
	// unused. Rows 1..n are sampled from one batched distribution; penalties
	// see only the committed history, not the other rows of the block.
	logits := spec.draft.Unembed(hidden.Slice(mlx.Slice(), mlx.Slice(1, n+1), mlx.Slice()))
	dist := r.Sampler.Distribution(pipelineSlot, logits, nil)
	tokens := r.Sampler.SampleDistribution(pipelineSlot, dist)
	return &draftCandidates{
		tokens: tokens.ExpandDims(0),
		dist:   dist,
	}
}
