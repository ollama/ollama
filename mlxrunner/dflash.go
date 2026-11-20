package mlxrunner

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/sample"
)

// dflashPendingFlushTokens bounds the held feature rows between flushes.
const dflashPendingFlushTokens = 256

// dflash2Depth is a DFlash2 session's fixed draft length. The depth controller's
// shallower picks lose more accepted tokens than they save in verify time.
const dflash2Depth = 7

type dflashBatchModel interface {
	ForwardBatch([]*batch.Batch, [][]cache.Cache) *mlx.Array
}

// dflashDrafter drafts with a block-diffusion draft model (DFlash): one
// forward proposes a whole block. A context-cache entry depends only on its
// own position, so the trie keys need no look-ahead.
type dflashDrafter struct {
	spec      *speculation
	blockSize int
	maskToken int32
}

func newDFlashDrafter(s *speculation, draft model.BlockDraft) *dflashDrafter {
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
	pending         *mlx.Scope // holds the rows until the flush

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
		if d.pending == nil {
			d.pending = mlx.NewScope()
		}
		d.pending.Attach(f)
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
	d.pending.Close()
	d.pending = nil
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

	if len(d.pendingFeatures) == 0 {
		return
	}
	offset := d.ctxOffset
	// Evaluating the cache state forces the writes: a session that never
	// drafts would otherwise leave the flush chain unevaluated, holding
	// every feature until close.
	mlx.ScopedAsyncEval(func() []*mlx.Array {
		spec.draft.Forward(&batch.Batch{
			SeqOffsets: []int32{int32(offset)},
			Hidden:     d.takePending(),
			Layout:     d.layout,
		}, cache.Layers(spec.targets), cache.Layers(spec.draftKV))
		state := make([]*mlx.Array, 0, 2*len(spec.draftKV))
		for _, c := range spec.draftKV {
			state = append(state, c.State()...)
		}
		return state
	})
}

// propose drafts a block after the not-yet-validated current token, one
// forward filling every mask position.
func (d *dflashDraftSession) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	spec := d.drafter.spec
	blockSize := d.drafter.blockSize

	n := min(maxTokens, blockSize-1)
	if n <= 0 || d.ctxOffset+d.pendingCount == 0 {
		return nil
	}
	// Draft only the requested rows, not the full trained block: exact for
	// causal layers, while a bidirectional draft's proposal distribution
	// changes; verification uses the actual proposal probabilities.
	var candidates *draftCandidates
	mlx.ScopedArrays(func() []*mlx.Array {
		hidden, _ := spec.draft.Forward(d.prepareBlock(current, n), cache.Layers(spec.targets), cache.Layers(spec.draftKV))
		candidates = d.selectBlock(hidden, current, n)
		return candidates.Arrays()
	})
	return candidates
}

func (d *dflashDraftSession) prepareBlock(current *mlx.Array, n int) *batch.Batch {
	d.commitBlock()
	masks := make([]int32, n)
	for i := range masks {
		masks[i] = d.drafter.maskToken
	}
	offset := d.ctxOffset
	features := d.takePending()
	b := &batch.Batch{
		InputIDs:     current.ExpandDims(-1).Concatenate(1, mlx.FromValues(masks, 1, n)),
		SeqOffsets:   []int32{int32(offset)},
		SeqQueryLens: []int32{int32(n + 1)},
		Hidden:       features,
		Layout:       d.layout,
	}
	scheduleSpeculation(d.drafter.spec.draftKV, d.ctxOffset, 1)
	d.blockOutstanding = true
	return b
}

// proposeDFlashBatch shares the forward and vocabulary projection, while each
// row keeps its own selector sampling stream and rollback boundary.
func proposeDFlashBatch(sessions []*dflashDraftSession, current []*mlx.Array, n int) []*draftCandidates {
	spec := sessions[0].drafter.spec
	draft := spec.draft.(dflashBatchModel)
	candidates := make([]*draftCandidates, len(sessions))
	mlx.ScopedAsyncEval(func() []*mlx.Array {
		rows := make([]*batch.Batch, len(sessions))
		caches := make([][]cache.Cache, len(sessions))
		for i, session := range sessions {
			rows[i] = session.prepareBlock(current[i], n)
			caches[i] = session.drafter.spec.draftKV
		}
		hidden := draft.ForwardBatch(rows, caches).Slice(mlx.Slice(), mlx.Slice(1, n+1), mlx.Slice())
		logits := spec.draft.Unembed(hidden)
		var arrays []*mlx.Array
		for i, session := range sessions {
			h := hidden.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice())
			l := logits.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice())
			candidates[i] = session.selectCandidates(h, l, current[i])
			arrays = append(arrays, candidates[i].Arrays()...)
		}
		return arrays
	})
	return candidates
}

// selectBlock samples the proposal from a draft forward's block rows.
func (d *dflashDraftSession) selectBlock(hidden, current *mlx.Array, n int) *draftCandidates {
	// Row i predicts the token at its own position; the anchor row is unused.
	hidden = hidden.Slice(mlx.Slice(), mlx.Slice(1, n+1), mlx.Slice())
	return d.selectCandidates(hidden, d.drafter.spec.draft.Unembed(hidden), current)
}

func (d *dflashDraftSession) selectCandidates(hidden, logits, current *mlx.Array) *draftCandidates {
	spec := d.drafter.spec
	r := spec.r
	if selector, ok := spec.draft.(model.BlockSelector); ok {
		tokens, ids, probs := selector.Select(hidden, logits, current, func(ids, scores *mlx.Array) (*mlx.Array, *mlx.Array) {
			tokens, dist := r.Sampler.SampleCandidates(spec.slotID, ids, scores)
			return tokens, dist.Probs
		})
		return &draftCandidates{tokens: tokens, dist: sample.Distribution{IDs: ids.Squeeze(0), Probs: probs.Squeeze(0)}}
	}
	// Independent DFlash rows see the committed history, without the other
	// rows of the block contributing to penalties.
	dist := r.Sampler.Distribution(spec.slotID, logits, nil)
	tokens := r.Sampler.SampleDistribution(spec.slotID, dist)
	return &draftCandidates{
		tokens: tokens.ExpandDims(0),
		dist:   dist,
	}
}
