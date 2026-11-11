package gemma4embedding

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/gemma4"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// preparedImage / preparedAudio are gemma4embedding's model-private media
// state: PreparedItem.Opaque values round-tripped untouched from
// PrepareMedia into Forward's scatter.
type preparedImage struct {
	positions []int32
	geom      gemma4.ImageGeometry
}

type preparedAudio struct {
	numTokens int32
}

// Forward runs the bidirectional encoder and returns (projected hidden,
// unused aux). The returned hidden is [B, L, embedding_dim]: final RMSNorm
// followed by the per-token embedding_projection. Pooling and L2
// normalization are the runner pipeline's job.
func (m *Model) Forward(b *batch.Batch, caches []cache.Layer) (hidden, auxHidden *mlx.Array) {
	dims := b.InputIDs.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))

	h := m.EmbedTokens.Forward(b.InputIDs)
	h = mlx.MulScalar(h, m.Cfg.EmbedScale)

	if len(b.Media) > 0 {
		if scattered, err := m.scatterMedia(h, b); err != nil {
			// Forward has no error return on model.Model; misconfigured
			// media means the request itself is bad, and runEmbed
			// validates before reaching here. A leftover nil-Features
			// entry is the only realistic path and is a no-op.
			slog.Error("gemma4embedding media scatter", "error", err)
		} else {
			h = scattered
		}
	}

	var perLayerInputs *mlx.Array
	if m.Cfg.HiddenSizePerLayer > 0 {
		perLayerInputs = m.computePLEInputs(h, B, L)
	}

	slidingMask, fullMask := m.buildMasks(b)

	for i, layer := range m.Layers {
		var pleInput *mlx.Array
		if perLayerInputs != nil {
			pleInput = sliceLayerDim(perLayerInputs, int32(i), B, L, m.Cfg.HiddenSizePerLayer)
		}

		mask := fullMask
		if layer.IsSliding {
			mask = slidingMask
		}

		h = layer.Forward(h, b, positions, B, L, m.Cfg, pleInput, mask)
	}

	out := mlx.RMSNormFn(h, m.NormScaled, m.Cfg.RMSNormEps)
	out = m.EmbeddingProjection.Forward(out)
	return out, nil
}

// Unembed is unused for embedding models; the runner taps Forward output
// directly. Implemented to satisfy model.Model.
func (m *Model) Unembed(x *mlx.Array) *mlx.Array { return x }

// buildMasks returns the per-layer-type masks. Embeddings are bidirectional:
// full layers return the zero mask (SDPA composes padding itself), sliding
// layers attend |q-kv| <= sliding_window+1 inclusive — the reference does
// config.sliding_window + 1 (modeling_embedding_gemma2) over an inclusive
// abs() overlay (masking_utils).
func (m *Model) buildMasks(b *batch.Batch) (sliding, full nn.AttentionMask) {
	window := int(m.Cfg.SlidingWindow) + 1
	return BidirectionalSlidingWindowMask(b, window, mlx.DTypeFloat32), nn.AttentionMask{}
}

// BidirectionalSlidingWindowMask returns an additive [B, 1, L, L] mask that
// blocks keys k for query q when |q-k| > window; window is the inclusive
// max attended distance. Padding columns are left to SDPA's kLens handling.
func BidirectionalSlidingWindowMask(b *batch.Batch, window int, dtype mlx.DType) nn.AttentionMask {
	if window <= 0 {
		return nn.AttentionMask{}
	}
	B := len(b.SeqOffsets)
	L := b.InputIDs.Dim(1)

	// If every row is short enough that no pair exceeds the window, the
	// mask is all-attending: skip materializing it.
	needed := false
	for i := range B {
		if int(b.SeqQueryLens[i]) > window {
			needed = true
			break
		}
	}
	if !needed {
		return nn.AttentionMask{}
	}

	negInf := float32(math.Inf(-1))
	vals := make([]float32, B*L*L)
	for i := range B {
		qlen := int(b.SeqQueryLens[i])
		base := i * L * L
		for q := range qlen {
			row := base + q*L
			for k := range qlen {
				d := q - k
				if d < 0 {
					d = -d
				}
				if d > window {
					vals[row+k] = negInf
				}
			}
		}
	}
	out := mlx.FromValues(vals, B, 1, L, L)
	if dtype != mlx.DTypeFloat32 {
		out = out.AsType(dtype)
	}
	return nn.ArrayMask(out)
}

// Forward runs one decoder layer: attn residual, MLP residual, PLE
// injection (projection-only), then the trained per-layer scalar.
func (l *DecoderLayer) Forward(x *mlx.Array, b *batch.Batch, positions *mlx.Array, B, L int32, cfg *Config, pleInput *mlx.Array, mask nn.AttentionMask) *mlx.Array {
	tc := &cfg.TextConfig
	normed := mlx.RMSNormFn(x, l.InputNormScaled, tc.RMSNormEps)
	attnOut := l.Attention.Forward(normed, b, positions, B, L, l, cfg, mask)
	attnOut = mlx.RMSNormFn(attnOut, l.PostAttnNormScaled, tc.RMSNormEps)
	h := mlx.Add(x, attnOut)

	normed = mlx.RMSNormFn(h, l.PreFFNormScaled, tc.RMSNormEps)
	mlpOut := l.MLP.Forward(normed)
	mlpOut = mlx.RMSNormFn(mlpOut, l.PostFFNormScaled, tc.RMSNormEps)
	h = mlx.Add(h, mlpOut)

	if l.PLE != nil && pleInput != nil {
		residual := h
		gated := mlx.GELUApprox(l.PLE.InputGate.Forward(h))
		gated = mlx.Mul(gated, pleInput)
		projected := l.PLE.Projection.Forward(gated)
		projected = mlx.RMSNormFn(projected, l.PLE.PostNormScaled, tc.RMSNormEps)
		h = mlx.Add(residual, projected)
	}

	h = mlx.Mul(h, l.LayerScalar)

	return h
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, positions *mlx.Array, B, L int32, layer *DecoderLayer, cfg *Config, mask nn.AttentionMask) *mlx.Array {
	tc := &cfg.TextConfig
	headDim := tc.layerHeadDim(layer.LayerIdx)
	kvHeads := tc.layerKVHeads(layer.LayerIdx)

	q := a.QProj.Forward(x)
	q = mlx.Reshape(q, B, L, tc.NumAttentionHeads, headDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	q = mlx.RMSNormFn(q, a.QNormScaled, tc.RMSNormEps)

	k := a.KProj.Forward(x)
	k = mlx.Reshape(k, B, L, kvHeads, headDim)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	v := a.VProj.Forward(x)
	v = mlx.Reshape(v, B, L, kvHeads, headDim)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	k = mlx.RMSNormFn(k, a.KNormScaled, tc.RMSNormEps)

	ropeBase := cfg.FullRopeBase
	ropeDims := cfg.FullRopeDims
	if layer.IsSliding {
		ropeBase = cfg.SlidingRopeBase
		ropeDims = int(headDim)
	}
	q = mlx.RoPEWithFreqs(q, ropeDims, false, ropeBase, 1.0, positions, nil)
	k = mlx.RoPEWithFreqs(k, ropeDims, false, ropeBase, 1.0, positions, nil)

	// Weightless V RMS norm (scale-free in the checkpoint).
	v = mlx.RMSNormFn(v, nil, tc.RMSNormEps)

	out := nn.ScaledDotProductAttention(b, q, 1.0, nn.WithKV(k, v, b.SeqQueryLens), nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, tc.NumAttentionHeads*headDim)
	return a.OProj.Forward(out)
}

// Forward runs the GELU-tanh MLP.
func (m *MLP) Forward(x *mlx.Array) *mlx.Array {
	gate := m.GateProj.Forward(x)
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(mlx.GeGLU(gate, up))
}

// computePLEInputs builds per-layer inputs for the projection-only PLE:
// per_layer_model_projection of the input hidden, scaled, normed, then
// reshaped to [B, L, NumLayers, HiddenSizePerLayer].
func (m *Model) computePLEInputs(h *mlx.Array, B, L int32) *mlx.Array {
	tc := &m.Cfg.TextConfig
	pleProj := m.PerLayerModelProj.Forward(h)
	pleProj = mlx.MulScalar(pleProj, float32(1.0/math.Sqrt(float64(tc.HiddenSize))))
	pleProj = mlx.Reshape(pleProj, B, L, tc.NumHiddenLayers, tc.HiddenSizePerLayer)
	return mlx.RMSNormFn(pleProj, m.PerLayerProjNormWeight, tc.RMSNormEps)
}

// sliceLayerDim extracts a single layer's PLE input from the combined
// [B, L, NumLayers, HiddenSizePerLayer] tensor.
func sliceLayerDim(combined *mlx.Array, layerIdx, B, L, pleDim int32) *mlx.Array {
	sliced := mlx.SliceStartStop(combined,
		[]int32{0, 0, layerIdx, 0},
		[]int32{B, L, layerIdx + 1, pleDim},
	)
	return mlx.Squeeze(sliced, 2)
}

// scatterMedia writes the tower-encoded features into the placeholder rows
// the runner spliced into InputIDs. Embedding forward is a single bidirectional
// pass over the whole request, so gemma4's chunk-window bookkeeping reduces
// to direct row replacement; nil Features means the item arrived without
// preprocessing (a caller bug) and is skipped.
func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) (*mlx.Array, error) {
	for i, item := range b.Media {
		if item.Features == nil {
			continue
		}
		if item.Seq < 0 || item.Seq >= len(b.SeqQueryLens) {
			return h, fmt.Errorf("media item %d: seq %d outside the input batch", i, item.Seq)
		}
		var rows int
		switch p := item.Opaque.(type) {
		case preparedAudio:
			rows = int(p.numTokens)
		case preparedImage:
			rows = int(p.geom.NumSoftTokens)
		default:
			return h, fmt.Errorf("media item %d: unknown opaque %T", i, item.Opaque)
		}
		if got := item.Features.Dim(0); got != rows {
			return h, fmt.Errorf("media item %d: features rows %d, want %d", i, got, rows)
		}
		if item.Pos < 0 || item.Pos+rows > int(b.SeqQueryLens[item.Seq]) {
			return h, fmt.Errorf("media item %d: [%d,%d) runs past the query tail %d", i, item.Pos, item.Pos+rows, b.SeqQueryLens[item.Seq])
		}
		feat := mlx.Reshape(item.Features.AsType(h.DType()), 1, int32(rows), m.Cfg.HiddenSize)
		h = h.SliceUpdate(feat, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(item.Pos, item.Pos+rows), mlx.Slice())
	}
	return h, nil
}

// EncodeMedia builds one tower's lazy graph. PrepareMedia has already checked
// the media kind and tower availability.
func (m *Model) EncodeMedia(item *model.PreparedItem, data *mlx.Array) *mlx.Array {
	switch p := item.Opaque.(type) {
	case preparedImage:
		return m.VisionTower.Encode(data, p.positions, p.geom, m.Cfg.VisionConfig, m.EmbedVision)
	case preparedAudio:
		return m.AudioTower.Encode(data, m.Cfg.AudioConfig, m.EmbedAudio)
	default:
		panic(fmt.Sprintf("unknown prepared media opaque %T", item.Opaque))
	}
}
