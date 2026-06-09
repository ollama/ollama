package mlxrunner

import (
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

const (
	mtpDefaultInitialDraftTokens = 4
	mtpDefaultMaxDraftTokens     = 16
)

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

// mtpDrafter drafts with a model's multi-token-prediction head: a small
// head trained to continue the target's hidden states, fed through the
// committed-stream reports. It retains the hidden at the last committed
// slot and tracks the committed frontier, which anchors its drafting.
type mtpDrafter struct {
	spec   *speculation
	draft  base.MTPDraftModel
	target base.MTPEmbeddingModel
	caches []cache.Cache

	// frontier is the slot after the last committed token; the hidden
	// reported for slot frontier-1 is retained (pinned) as the fusion
	// input for the next proposal chain.
	frontier       int
	frontierHidden *mlx.Array
}

// newMTPDrafter returns the MTP drafter for this request's caches, or nil
// when the model carries no MTP head.
func newMTPDrafter(s *speculation, caches []cache.Cache) *mtpDrafter {
	draft, ok := s.draft.(base.MTPDraftModel)
	if !ok {
		return nil
	}
	target, ok := s.r.Model.(base.MTPEmbeddingModel)
	if !ok {
		return nil
	}
	return &mtpDrafter{spec: s, draft: draft, target: target, caches: caches}
}

func (d *mtpDrafter) committed(tokens, hiddens *mlx.Array, position int) {
	d.frontier = position + tokens.Dim(1)
	h := lastHiddenRow(hiddens)
	mlx.Pin(h)
	if d.frontierHidden != nil {
		mlx.Unpin(d.frontierHidden)
	}
	d.frontierHidden = h
}

func (d *mtpDrafter) close() {
	if d.frontierHidden != nil {
		mlx.Unpin(d.frontierHidden)
		d.frontierHidden = nil
	}
}

// propose drafts a token chain Gemma-style ("single-position"): the head is
// trained to draft every speculative token as if it sat at the last
// committed slot, re-attending the target caches read-only. Each step fuses
// the previous token's target embedding with the hidden chain — the
// reported hidden first, then the head's own projections — and the
// RoPE/cache anchor stays at the last committed slot while the proposed
// tokens advance.
func (d *mtpDrafter) propose(current *mlx.Array, maxTokens int) *draftCandidates {
	if maxTokens <= 0 || d.frontierHidden == nil {
		return nil
	}
	r := d.spec.r

	anchor := int32(d.frontier - 1)
	lastToken := current.ExpandDims(-1)
	lastHidden := d.frontierHidden
	draftTokens := make([]*mlx.Array, 0, maxTokens)
	draftDists := make([]sampler.Distribution, 0, maxTokens)
	var prefix *mlx.Array

	for range maxTokens {
		tokenEmbedding := d.target.TokenEmbeddings(lastToken)
		inputs := tokenEmbedding.Concatenate(-1, lastHidden)
		logits, projected := d.draft.Draft(inputs, anchor, d.caches)
		stepLogits := lastLogits(logits)
		dist := r.Sampler.Distribution(pipelineSlot, stepLogits, prefix)
		nextToken := r.Sampler.SampleDistribution(pipelineSlot, dist)

		lastToken = nextToken.ExpandDims(-1)
		lastHidden = projected
		draftTokens = append(draftTokens, lastToken)
		draftDists = append(draftDists, dist)
		if prefix == nil {
			prefix = lastToken
		} else {
			prefix = prefix.Concatenate(1, lastToken)
		}
	}
	return &draftCandidates{
		tokens: mlx.Concatenate(draftTokens, 1),
		dist:   sampler.ConcatenateDistributions(draftDists),
	}
}

func lastHiddenRow(hidden *mlx.Array) *mlx.Array {
	return hidden.Slice(mlx.Slice(), mlx.Slice(hidden.Dim(1)-1), mlx.Slice())
}
