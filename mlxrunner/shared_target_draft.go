package mlxrunner

import (
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/sample"
)

// sharedTargetBlockDraft reads the target KV through Forward and retains no
// private cache. CorrectHidden takes a single [1,1,H] feature row and its
// preceding chosen token [1], then returns the row for normal Unembed.
type sharedTargetBlockDraft interface {
	model.DraftModel
	SharedTargetBlockParams() (blockSize int, maskToken int32)
	CorrectHidden(hidden, previous *mlx.Array) *mlx.Array
}

type sharedTargetDrafter struct {
	spec      *speculation
	draft     sharedTargetBlockDraft
	blockSize int
	maskToken int32
}

func (d *sharedTargetDrafter) draftLimit() int { return d.blockSize - 1 }

func (d *sharedTargetDrafter) open(layout []any) draftSession {
	return &sharedTargetDraftSession{drafter: d, layout: layout}
}

// sharedTargetDraftSession retains the last committed feature row. The
// proposal uses the target's cache, so only the verifier writes and rewinds KV.
type sharedTargetDraftSession struct {
	drafter  *sharedTargetDrafter
	layout   []any
	features *mlx.Array
	scope    *mlx.Scope
	offset   int
}

func (d *sharedTargetDraftSession) committed(tokens, features *mlx.Array, position int, _ []batch.MediaItem) {
	n := tokens.Dim(1)
	last := features.TakeAxis(mlx.FromValues([]int32{int32(n - 1)}, 1), 1)
	if d.features == nil {
		d.scope = mlx.NewScope()
		d.scope.Attach(last)
		d.features = last
	} else {
		d.features.Set(last)
	}
	d.offset = position + n
}

func (*sharedTargetDraftSession) settle(*mlx.Array) {}

func (d *sharedTargetDraftSession) close() {
	d.scope.Close()
}

func (d *sharedTargetDraftSession) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	n := min(maxTokens, d.drafter.draftLimit())
	if n <= 0 || d.features == nil {
		return nil
	}

	spec := d.drafter.spec
	var candidates *draftCandidates
	mlx.ScopedArrays(func() []*mlx.Array {
		draft := d.drafter.draft
		masks := make([]int32, n)
		for i := range masks {
			masks[i] = d.drafter.maskToken
		}
		hidden, _ := draft.Forward(&batch.Batch{
			InputIDs:     current.ExpandDims(-1).Concatenate(1, mlx.FromValues(masks, 1, n)),
			SeqOffsets:   []int32{int32(d.offset)},
			SeqQueryLens: []int32{int32(n + 1)},
			Hidden:       d.features,
			Layout:       d.layout,
		}, cache.Layers(spec.targets), nil)
		dists := make([]sample.Distribution, n)
		previous := current
		var prefix *mlx.Array
		for i := range n {
			row := hidden.Slice(mlx.Slice(), mlx.Slice(i+1), mlx.Slice())
			logits := draft.Unembed(draft.CorrectHidden(row, previous))
			dists[i] = spec.r.Sampler.Distribution(spec.slotID, logits, prefix)
			previous = spec.r.Sampler.SampleDistribution(spec.slotID, dists[i])
			if prefix == nil {
				prefix = previous
			} else {
				prefix = prefix.Concatenate(0, previous)
			}
		}
		candidates = &draftCandidates{tokens: prefix.ExpandDims(0), dist: sample.ConcatenateDistributions(dists)}
		return candidates.Arrays()
	})
	return candidates
}
