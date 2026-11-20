package hy4_flash

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type dflyDraft struct {
	Embed                             nn.EmbeddingLayer
	Head                              nn.LinearLayer
	Layers                            []dflyLayer
	FusionEmbed, FusionHidden, Norm   *nn.RMSNorm
	Fusion                            nn.LinearLayer
	FusionGate                        *mlx.Array
	CorrectionEmbed, CorrectionHidden *nn.RMSNorm
	Correction                        denseMLP
	kvRank, blockSize, indexWidth     int
	maskToken                         int32
}

type dflyLayer struct {
	Attention                      attention
	AttentionNorm, FeedForwardNorm *nn.RMSNorm
	FeedForward                    denseMLP
	sharedSlot                     int
}

const dflyPrefix = "model.dfly_layers"

func (c *config) hasDFly() bool {
	return c.DraftLayers > 0 && c.Speculative.Method == "dfly"
}

func (m *Model) loadDFly(w weights) (*dflyDraft, error) {
	c := &m.cfg
	if len(c.DraftKVLayers) != c.DraftLayers || !slices.Equal(c.DraftKVLayers, c.DraftIndexLayers) ||
		c.DraftFusion != "slot0" || !c.DraftCorrection || c.Speculative.Tokens <= 0 || c.Speculative.Tokens >= c.Text.Context ||
		c.DraftIndexTopK != c.Text.IndexRatio*c.Text.IndexTopK || c.DraftMaskToken < 0 || int(c.DraftMaskToken) >= m.tok.VocabSize() ||
		c.DraftLocalWindow != 0 || (c.DraftCausal != nil && !*c.DraftCausal) || c.Text.Context > 1<<24 {
		return nil, fmt.Errorf("unsupported Hy4 DFly draft configuration")
	}
	const prefix = dflyPrefix
	draft := &dflyDraft{
		Embed: m.Embed, Head: m.Head, Layers: make([]dflyLayer, c.DraftLayers),
		FusionEmbed: w.norm(prefix + ".mtp_fusion.enorm"), FusionHidden: w.norm(prefix + ".mtp_fusion.hnorm"),
		Fusion: w.linear(prefix + ".mtp_fusion.eh_proj"), FusionGate: w.tensor(prefix + ".mtp_fusion.gate"),
		CorrectionEmbed: w.norm(prefix + ".hidden_correction.embed_norm"), CorrectionHidden: w.norm(prefix + ".hidden_correction.hidden_norm"),
		Correction: loadMLP(w, prefix+".hidden_correction"),
		Norm:       w.norm(prefix + ".final_norm"), kvRank: c.Text.KVRank,
		blockSize: c.Speculative.Tokens + 1, maskToken: c.DraftMaskToken, indexWidth: c.DraftIndexTopK,
	}
	for j, slot := range c.DraftKVLayers {
		if slot < 0 || slot >= len(m.Layers) || m.Layers[slot].Attention.Indexer == nil || slices.Contains(c.DraftKVLayers[:j], slot) {
			return nil, fmt.Errorf("Hy4 DFly draft layer %d has invalid target layer %d", j, slot)
		}
		m.Layers[slot].Attention.draftSelection = true
		path := fmt.Sprintf("%s.%d", prefix, j)
		draft.Layers[j] = dflyLayer{
			Attention:     loadAttention(w, path+".self_attn", c.Text.Indexed, c.Text.KVRank),
			AttentionNorm: w.norm(path + ".input_layernorm"), FeedForwardNorm: w.norm(path + ".post_attention_layernorm"),
			FeedForward: loadMLP(w, path+".mlp"), sharedSlot: slot,
		}
	}
	if err := *w.err; err != nil {
		return nil, err
	}
	if head, ok := draft.Head.(*outputHead); ok {
		// Verification uses a batched FP32 projection. Keep the fast BF16
		// view for the predecessor-dependent single-token drafts.
		head.Batched = nn.NewLinear(head.Weight.Clone(), nil)
	}
	return draft, nil
}

func (m *Model) SelfDraft() model.DraftModel {
	if m.Draft == nil {
		return nil
	}
	return m.Draft
}

func (*dflyDraft) LoadWeights(map[string]*mlx.Array) error { return nil }
func (*dflyDraft) NewCaches() []cache.Cache                { return nil }

func (d *dflyDraft) SharedTargetBlockParams() (int, int32) { return d.blockSize, d.maskToken }

func (d *dflyDraft) conditioning(b *batch.Batch, hidden *mlx.Array) *mlx.Array {
	B, L := hidden.Dim(0), hidden.Dim(1)
	sparse := slices.ContainsFunc(b.SeqOffsets, func(offset int32) bool {
		return int(offset)+L > d.indexWidth
	})
	if !sparse {
		return hidden
	}
	// Integer positions are lossless in FP32 throughout Hy4's context.
	// Carry the selection with the hidden rows so target verification can
	// retain the last accepted row after a speculative rollback.
	features := []*mlx.Array{hidden.AsType(mlx.DTypeFloat32)}
	for _, layer := range d.Layers {
		rows := make([]*mlx.Array, B)
		for row := range rows {
			if selected, ok := b.Memo.Get(draftSelectionKey{layer.sharedSlot, row}); ok {
				rows[row] = selected.(*mlx.Array).AsType(mlx.DTypeFloat32)
			} else {
				rows[row] = mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L, d.indexWidth), -1)
			}
		}
		values := rows[0]
		if B > 1 {
			values = mlx.Concatenate(rows, 0)
		}
		features = append(features, values)
	}
	return mlx.Concatenate(features, -1)
}

func (d *dflyDraft) Forward(b *batch.Batch, targetCaches, _ []cache.Layer) (hidden, auxHidden *mlx.Array) {
	hidden = d.Embed.Forward(b.InputIDs)
	anchor := hidden.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
	width := hidden.Dim(-1)
	targetHidden := b.Hidden.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, width)).AsType(hidden.DType())
	input := mlx.Concatenate([]*mlx.Array{d.FusionEmbed.Forward(anchor, 0), d.FusionHidden.Forward(targetHidden, 0)}, -1)
	anchor = mlx.Add(anchor, mlx.Mul(d.FusionGate, d.Fusion.Forward(input)))
	hidden = hidden.SliceUpdate(anchor, mlx.Slice(), mlx.Slice(0, 1), mlx.Slice())
	for j, l := range d.Layers {
		x := l.AttentionNorm.Forward(hidden, 0)
		a := l.Attention
		_, q, k, v := a.project(x, rotaryPositions{batch: b}, d.kvRank)
		history := targetCaches[l.sharedSlot].(cache.Attention).View(b)
		oldK, oldV := history.K(), history.V()
		if oldV.Dim(-1) == 0 {
			oldV = oldK.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, d.kvRank))
		}
		mask := nn.DeviceCausalMask()
		if oldK.Dim(2) > d.indexWidth {
			start := width + j*d.indexWidth
			selected := b.Hidden.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, start+d.indexWidth)).Reshape(-1).AsType(mlx.DTypeInt32)
			valid := mlx.Where(selected.Less(mlx.FromValue(0)), mlx.FromValue(false), selected.Less(mlx.FromValue(oldK.Dim(2))))
			prefixBias := mlx.Where(valid, mlx.FromValue(float32(0)), mlx.FromValue(float32(math.Inf(-1))))
			positions := mlx.Where(valid, selected, mlx.FromValue(0))
			oldK, oldV = oldK.TakeAxis(positions, 2), oldV.TakeAxis(positions, 2)
			bias := prefixBias.Concatenate(0, mlx.Zeros(mlx.DTypeFloat32, k.Dim(2))).Reshape(1, 1, 1, -1)
			mask = mask.Intersect(nn.ArrayMask(bias))
		}
		k = oldK.Concatenate(2, k)
		v = oldV.Concatenate(2, v)
		attended := nn.ScaledDotProductAttention(b, q, a.scale(),
			nn.WithKV(k, v, []int32{int32(k.Dim(2))}), nn.WithMask(mask), nn.WithSinks(a.Sinks))
		hidden = mlx.Add(hidden, a.finish(x, attended))
		hidden = mlx.Add(hidden, l.FeedForward.forward(l.FeedForwardNorm.Forward(hidden, 0)))
	}
	return d.Norm.Forward(hidden, 0), nil
}

func (d *dflyDraft) CorrectHidden(hidden, previous *mlx.Array) *mlx.Array {
	embed := d.Embed.Forward(previous.Reshape(1, 1))
	input := mlx.Concatenate([]*mlx.Array{d.CorrectionHidden.Forward(hidden, 0), d.CorrectionEmbed.Forward(embed, 0)}, -1)
	return mlx.Add(hidden, d.Correction.forward(input))
}

func (d *dflyDraft) Unembed(x *mlx.Array) *mlx.Array {
	return d.Head.Forward(x.AsType(mlx.DTypeFloat32))
}
