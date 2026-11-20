package hy4_flash

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type attention struct {
	shape                         attentionShape
	QDown, QUp, KVDown, Gate, Out nn.LinearLayer
	QNorm, KVNorm, OutNorm        *nn.RMSNorm
	QueryToLatent, LatentToValue  *nn.MultiLinear
	Sinks                         *mlx.Array
	Indexer                       *indexer
	indexSlot                     int
	draftSelection                bool
}

type draftSelectionKey struct{ slot, row int }

func loadAttention(w weights, prefix string, shape attentionShape, kvRank int) attention {
	a := attention{
		shape: shape,
		QDown: w.linear(prefix + ".q_a_proj"), QUp: w.linear(prefix + ".q_b_proj"),
		KVDown: w.linear(prefix + ".kv_a_proj_with_mqa"), Gate: w.linear(prefix + ".linear_gate"), Out: w.linear(prefix + ".o_proj"),
		QNorm: w.norm(prefix + ".q_a_layernorm"), KVNorm: w.norm(prefix + ".kv_a_layernorm"), OutNorm: w.norm(prefix + ".post_proj_norm"),
		Sinks: w.tensor(prefix + ".learnable_sink_param"),
	}
	// Absorb the key projection into Q and apply the value projection after
	// attention. The cache only needs one latent KV and one rotary key per token.
	kv := w.tensor(prefix + ".kv_b_proj.weight")
	if *w.err != nil {
		return a
	}
	kv = kv.Reshape(shape.heads, shape.plain+shape.value, kvRank)
	key := kv.Slice(mlx.Slice(), mlx.Slice(0, shape.plain), mlx.Slice())
	value := kv.Slice(mlx.Slice(), mlx.Slice(shape.plain, shape.plain+shape.value), mlx.Slice())
	// Matmul promotes this weight to the absorbed query dtype. Do the
	// promotion once, before pinning the model weights.
	if a.Sinks.DType() == mlx.DTypeFloat32 {
		key = key.AsType(mlx.DTypeFloat32)
		// Acquiring a separate value buffer lets the original packed
		// projection be released when weights are materialized.
		value = mlx.Contiguous(value, false)
	}
	a.QueryToLatent = nn.NewMultiLinear(key.Transpose(0, 2, 1))
	a.LatentToValue = nn.NewMultiLinear(value)
	return a
}

func (a attention) project(x *mlx.Array, positions rotaryPositions, kvRank int) (q, queries, keys, latent *mlx.Array) {
	B, L := x.Dim(0), x.Dim(1)
	shape := a.shape
	q = a.QNorm.Forward(a.QDown.Forward(x), 0)
	heads := a.QUp.Forward(q).Reshape(B, L, shape.heads, shape.plain+shape.rotary).Transpose(0, 2, 1, 3)
	plainQ := heads.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, shape.plain))
	ropeQ := heads.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(shape.plain, shape.plain+shape.rotary))
	ropeQ = positions.apply(ropeQ, shape)
	// Carry sink precision through the absorbed query projection. Rounding
	// its extra contraction to BF16 changes the resulting attention scores.
	if a.Sinks.DType() == mlx.DTypeFloat32 {
		plainQ = plainQ.AsType(mlx.DTypeFloat32)
		ropeQ = ropeQ.AsType(mlx.DTypeFloat32)
	}
	queries = mlx.Concatenate([]*mlx.Array{a.QueryToLatent.Forward(plainQ), ropeQ}, -1)

	kv := a.KVDown.Forward(x)
	latent = a.KVNorm.Forward(kv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, kvRank)), 0).ExpandDims(1)
	ropeK := kv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(kvRank, kvRank+shape.rotary)).ExpandDims(1)
	ropeK = positions.apply(ropeK, shape)
	keys = mlx.Concatenate([]*mlx.Array{latent, ropeK}, -1)
	return q, queries, keys, latent
}

func (a attention) scale() float32 {
	return float32(1 / math.Sqrt(float64(a.shape.plain+a.shape.rotary)))
}

func (a attention) finish(x, out *mlx.Array) *mlx.Array {
	out = a.LatentToValue.Forward(out.AsType(x.DType())).Transpose(0, 2, 1, 3).
		Reshape(x.Dim(0), x.Dim(1), a.shape.heads*a.shape.value)
	gate := a.Gate.Forward(x).AsType(mlx.DTypeFloat32).Sigmoid().AsType(x.DType())
	return a.OutNorm.Forward(a.Out.Forward(mlx.Mul(out, gate)), 0)
}

func (a attention) forward(x *mlx.Array, b *batch.Batch, caches []cache.Layer, slot int, positions rotaryPositions, c *textConfig) *mlx.Array {
	B, L := x.Dim(0), x.Dim(1)
	q, queries, keys, latent := a.project(x, positions, c.KVRank)
	var history *nn.KVHistory
	if len(caches) != 0 {
		placeholder := mlx.Zeros(x.DType(), B, 1, L, 0)
		history = caches[slot].(cache.Attention).Update(b, keys, placeholder)
	}
	mask := nn.DeviceCausalMask()
	scale := a.scale()
	var out *mlx.Array
	if a.Indexer == nil {
		kv := nn.WithKV(keys, latent, b.SeqQueryLens)
		if history != nil {
			kv = nn.WithMLAHistory(history, c.KVRank)
		} else {
			mask = mask.IntersectSlidingWindow(b, L, c.Window, x.DType())
		}
		out = nn.ScaledDotProductAttention(b, queries, scale, kv, nn.WithMask(mask), nn.WithSinks(a.Sinks))
	} else {
		var indexCache cache.Attention
		if len(caches) != 0 {
			indexCache = caches[a.indexSlot].(cache.Attention)
		}
		indexKeys := a.Indexer.histories(x, b, indexCache, a.shape, c)
		rows := make([]*mlx.Array, B)
		for row := range rows {
			bRow := b.Row(row)
			r := mlx.Slice(row, row+1)
			query := queries.Slice(r, mlx.Slice(), mlx.Slice(), mlx.Slice())
			index := indexKeys[row]
			selected := a.Indexer.selectTokens(x.Slice(r, mlx.Slice(), mlx.Slice()), q.Slice(r, mlx.Slice(), mlx.Slice()), index, bRow, a.shape, c)
			if a.draftSelection && selected != nil {
				b.Memo.Put(draftSelectionKey{slot, row}, selected)
			}
			kv := nn.WithKV(keys.Slice(r, mlx.Slice(), mlx.Slice(), mlx.Slice()), latent.Slice(r, mlx.Slice(), mlx.Slice(), mlx.Slice()), bRow.SeqQueryLens)
			if history != nil {
				h := history
				if h.Rows() != nil {
					h = h.Rows()[row]
				}
				kv = nn.WithMLAHistory(h, c.KVRank)
			}
			rows[row] = nn.ScaledDotProductAttention(bRow, query, scale, kv,
				nn.WithMask(mask.Intersect(selectionMask(selected, index.tokens))), nn.WithSinks(a.Sinks))
		}
		out = mlx.Concatenate(rows, 0)
	}
	return a.finish(x, out)
}
