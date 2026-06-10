package mlxrunner

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

const (
	mtpDefaultInitialDraftTokens = 4
	mtpDefaultMaxDraftTokens     = 16
)

// mtpPendingFlushTokens caps how many committed look-ahead tokens wait in the
// pending buffer before a batched flush, bounding the pinned hidden states
// regardless of what else triggers a flush.
const mtpPendingFlushTokens = 32

func (r *Runner) mtpDefaults(sample bool) base.MTPDefaults {
	defaults := base.MTPDefaults{
		InitialDraftTokens: mtpDefaultInitialDraftTokens,
		MaxDraftTokens:     mtpDefaultMaxDraftTokens,
		Enabled:            true,
	}
	if p, ok := r.Model.(base.MTPDefaultsProvider); ok {
		defaults = p.MTPDraftDefaults(sample)
	}
	if defaults.InitialDraftTokens <= 0 {
		defaults.InitialDraftTokens = mtpDefaultInitialDraftTokens
	}
	if defaults.MaxDraftTokens <= 0 {
		defaults.MaxDraftTokens = mtpDefaultMaxDraftTokens
	}
	return defaults
}

// mtpDrafter drafts with a model's multi-token-prediction head, fed through
// the committed-stream reports. The draft KV pairs each slot S with the
// look-ahead token at S+1 fused with the target hidden at S, so a pair
// completes only when the next token arrives.
type mtpDrafter struct {
	spec *speculation

	// frontier is the slot after the last reported token; frontierHidden is
	// the pinned target hidden at frontier-1, fused into the next pair.
	frontier       int
	frontierHidden *mlx.Array

	// committedDraftOffset is the slot after the last pair written to the
	// draft caches; later pairs wait pinned in the pending lists until
	// flushed. pendingCount is the look-ahead tokens those lists hold, summed
	// across the buffered runs.
	committedDraftOffset int
	pendingTokens        []*mlx.Array
	pendingHiddens       []*mlx.Array
	pendingCount         int

	// heldHidden is the frontier row's pre-unembed hidden and heldProjected
	// its fusion hidden, carried from the last flush so the first proposal
	// reuses them without a head call.
	heldHidden    *mlx.Array
	heldProjected *mlx.Array
}

// newMTPDrafter returns the MTP drafter cursor for this request, syncing its
// pairing frontier to the draft caches' restored offset.
func newMTPDrafter(s *speculation) *mtpDrafter {
	d := &mtpDrafter{spec: s}
	if len(s.draftKV) > 0 {
		// A restored prefix arrives with the draft caches already written;
		// pairing resumes from their absolute offset.
		d.committedDraftOffset = s.draftKV[0].Offset()
		d.frontier = d.committedDraftOffset
	}
	return d
}

func (d *mtpDrafter) committed(tokens, hiddens *mlx.Array, position int) {
	n := tokens.Dim(1)
	if len(d.spec.draftKV) > 0 {
		// The pair at slot S fuses token[S+1] with hidden[S], so a run pairs its
		// tokens with its own hiddens shifted one slot back: the first writable
		// token takes the carried frontier hidden, each later token the row
		// before it. Leading tokens whose slot is already buffered or written
		// through (a proposal consumed the run's first token, or a restored
		// prefix sits at the run start) are skipped; a slot below the frontier
		// is a gap bug.
		start := d.committedDraftOffset + d.pendingCount - position + 1
		if start < 0 {
			panic(fmt.Sprintf("mtp: committed run at %d leaves a pair gap at %d", position, d.committedDraftOffset+d.pendingCount))
		}
		if start < n {
			ids := tokens.Slice(mlx.Slice(), mlx.Slice(start, n))
			var h *mlx.Array
			if start == 0 {
				h = mlx.Concatenate([]*mlx.Array{d.frontierHidden, hiddens.Slice(mlx.Slice(), mlx.Slice(0, n-1), mlx.Slice())}, 1)
			} else {
				h = hiddens.Slice(mlx.Slice(), mlx.Slice(start-1, n-1), mlx.Slice())
			}
			d.queueCacheWrites(ids, h)
		}
	}

	d.frontier = position + n
	d.setFrontierHidden(lastHiddenRow(hiddens))
}

// finish settles the drafter when generation ends: current completes the
// frontier pair, leveling the draft caches with the target's resting offset.
//
// TODO: leveling the draft to the target writes a boundary entry whose
// look-ahead token is outside the stored prefix (here, the never-committed
// current). When a later request restores this prefix and diverges at the
// boundary, that entry is stale and lowers draft acceptance. EAGLE keeps the
// draft one slot behind the target so the unconfirmed boundary entry is never
// written (its bigram partner token[S+1] does not exist yet); we should do the
// same rather than level here. Regenerating hidden[S] to re-pair the boundary
// on the next request is a separate, re-prefill-bound concern for recurrent
// targets.
func (d *mtpDrafter) finish(current *mlx.Array) {
	if len(d.spec.draftKV) == 0 {
		return
	}
	d.settle(current)
}

// settle completes any open frontier pair with current, then flushes.
func (d *mtpDrafter) settle(current *mlx.Array) {
	if d.frontierHidden != nil && d.frontier-1 == d.committedDraftOffset+d.pendingCount {
		d.queueCacheWrites(current.ExpandDims(-1), d.frontierHidden)
	}
	d.flush()
}

func (d *mtpDrafter) close() {
	d.flush()
	d.setFrontierHidden(nil)
	d.setHeld(nil, nil)
}

// queueCacheWrites buffers completed draft-cache writes — look-ahead tokens
// fused with their target hiddens — flushing once the buffer reaches the token
// cap so the pinned hiddens stay bounded. flush coalesces the buffered writes
// into one head forward, so a contiguous run lands in a single draft-cache extend.
func (d *mtpDrafter) queueCacheWrites(tokens, hiddens *mlx.Array) {
	mlx.Pin(tokens, hiddens)
	d.pendingTokens = append(d.pendingTokens, tokens)
	d.pendingHiddens = append(d.pendingHiddens, hiddens)
	d.pendingCount += tokens.Dim(1)
	if d.pendingCount >= mtpPendingFlushTokens {
		d.flush()
	}
}

// flush writes the pending pairs to the draft caches in one head forward,
// dropping speculative entries past the committed range first and holding
// the last row's logits and projected hidden for the next proposal chain.
func (d *mtpDrafter) flush() {
	if len(d.pendingTokens) == 0 {
		return
	}
	for _, c := range d.spec.draftKV {
		if c.Offset() > d.committedDraftOffset {
			if !c.Restore(nil, d.committedDraftOffset) {
				panic(fmt.Sprintf("mtp: draft cache rewind to %d failed", d.committedDraftOffset))
			}
		}
	}

	ids := mlx.Concatenate(d.pendingTokens, 1)
	hiddens := mlx.Concatenate(d.pendingHiddens, 1)
	hidden, projected := d.spec.draft.Draft(&batch.Batch{
		InputIDs:     ids,
		SeqOffsets:   []int32{int32(d.committedDraftOffset)},
		SeqQueryLens: []int32{int32(ids.Dim(1))},
		Hidden:       hiddens,
	}, d.spec.caches)
	d.setHeld(lastHiddenRow(hidden), lastHiddenRow(projected))
	d.committedDraftOffset += ids.Dim(1)

	// Force the draft writes: a session that never drafts would otherwise
	// leave the flush chain unevaluated, pinning every hidden until close.
	state := make([]*mlx.Array, 0, 2*len(d.spec.draftKV))
	for _, c := range d.spec.draftKV {
		state = append(state, c.State()...)
	}
	mlx.AsyncEval(state...)

	mlx.Unpin(d.pendingTokens...)
	mlx.Unpin(d.pendingHiddens...)
	d.pendingTokens, d.pendingHiddens = nil, nil
	d.pendingCount = 0
}

func (d *mtpDrafter) setFrontierHidden(h *mlx.Array) {
	mlx.Pin(h)
	mlx.Unpin(d.frontierHidden)
	d.frontierHidden = h
}

// setHeld replaces the held flush outputs, pinned until the next flush or close.
func (d *mtpDrafter) setHeld(hidden, projected *mlx.Array) {
	mlx.Pin(hidden, projected)
	mlx.Unpin(d.heldHidden, d.heldProjected)
	d.heldHidden, d.heldProjected = hidden, projected
}

// propose drafts a token chain after the not-yet-validated current token.
// A head with draft caches settles the frontier pair first, so its first step
// reuses the held frontier row with no head call; a cacheless head re-attends
// the target caches read-only, anchored at the last committed slot.
func (d *mtpDrafter) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	if maxTokens <= 0 || d.frontierHidden == nil {
		return nil
	}
	r := d.spec.r

	if len(d.spec.draftKV) > 0 {
		d.settle(current)
		if d.heldHidden == nil {
			return nil
		}
	}

	lastToken := current.ExpandDims(-1)
	lastHidden := d.frontierHidden
	draftDists := make([]sampler.Distribution, 0, maxTokens)
	var prefix *mlx.Array

	for i := range maxTokens {
		var hidden, projected *mlx.Array
		if i == 0 && len(d.spec.draftKV) > 0 {
			// The settle flush already produced the frontier row; reuse it
			// instead of re-running the head.
			hidden, projected = d.heldHidden, d.heldProjected
		} else {
			// A head with draft caches writes each draft token to the next
			// draft-cache slot, advancing one per step from the last committed
			// slot (the held i==0 step stands in for that slot). A cacheless
			// head stays at the last committed slot every step, re-attending
			// the committed prefix read-only ("single-position").
			pos := d.frontier - 1
			if len(d.spec.draftKV) > 0 {
				pos = d.frontier - 1 + i
			}
			hidden, projected = d.spec.draft.Draft(&batch.Batch{
				InputIDs:     lastToken,
				SeqOffsets:   []int32{int32(pos)},
				SeqQueryLens: []int32{1},
				Hidden:       lastHidden,
			}, d.spec.caches)
		}
		// Unembed only the row being sampled, never the batch.
		stepLogits := d.spec.draft.Unembed(hidden).Squeeze(1)
		lastHidden = projected
		// The chain's earlier drafts ride along as the row's history, so
		// penalties shape proposals the same way they shape validation.
		dist := r.Sampler.Distribution(pipelineSlot, stepLogits, prefix)
		nextToken := r.Sampler.SampleDistribution(pipelineSlot, dist)

		lastToken = nextToken.ExpandDims(-1)
		draftDists = append(draftDists, dist)
		if prefix == nil {
			prefix = lastToken
		} else {
			prefix = prefix.Concatenate(1, lastToken)
		}
	}
	return &draftCandidates{
		tokens: prefix,
		dist:   sampler.ConcatenateDistributions(draftDists),
	}
}

func lastHiddenRow(hidden *mlx.Array) *mlx.Array {
	return hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1), mlx.Slice())
}
