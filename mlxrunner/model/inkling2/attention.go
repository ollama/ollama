package inkling2

import (
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type selfAttention struct {
	Query, Key, Value, Output *nn.Linear
	Relative, Gate, Sink      *nn.Linear
	PositionProjection        *mlx.Array
	QueryScale, RelativeScale *mlx.Array
	KeyConv, ValueConv        shortConv
	headDim, heads, kvHeads   int
	window                    int
	rmsEpsilon                float32
	qkClip, relativeClip      float32
	logFloor, logAlpha        float32
}

func (a selfAttention) forward(qInput, kInput, vInput *mlx.Array, b *batch.Batch, kvCache, keyCache, valueCache cache.Layer) *mlx.Array {
	rows, tokens := qInput.Dim(0), qInput.Dim(1)
	q := projectTarget(a.Query, qInput).Reshape(rows, tokens, a.heads, a.headDim)
	k := a.KeyConv.forward(projectTarget(a.Key, kInput), kInput, b, keyCache).Reshape(rows, tokens, a.kvHeads, a.headDim)
	v := a.ValueConv.forward(projectTarget(a.Value, vInput), vInput, b, valueCache).Reshape(rows, tokens, a.kvHeads, a.headDim)
	r := projectTarget(a.Relative, qInput).Reshape(rows, tokens, a.heads, a.PositionProjection.Dim(0))
	var tau *mlx.Array
	if a.window == 0 && a.logFloor > 0 && a.logAlpha != 0 {
		offsets := mlx.FromValues(b.SeqOffsets, rows, 1)
		indices := mlx.Arange(1, float64(tokens+1), 1, mlx.DTypeInt32)
		length := offsets.Add(indices).AsType(mlx.DTypeFloat32)
		ratio := mlx.Maximum(mlx.DivScalar(length, a.logFloor), mlx.FromValue(float32(1)))
		tau = mlx.AddScalar(mlx.MulScalar(mlx.Log(ratio), a.logAlpha), 1).Reshape(rows, tokens, 1, 1)
	}
	q = clipL2(q, a.QueryScale, tau, a.qkClip).Transpose(0, 2, 1, 3)
	k = clipL2(k, nil, nil, a.qkClip).Transpose(0, 2, 1, 3)
	v = mlx.RMSNormFn(v, nil, a.rmsEpsilon).Transpose(0, 2, 1, 3)
	r = clipL2(r, a.RelativeScale, tau, a.relativeClip)
	table := r.Matmul(a.PositionProjection).Transpose(0, 2, 1, 3)
	sink := projectTarget(a.Sink, qInput).Transpose(0, 2, 1).ExpandDims(-1)

	var history *nn.KVHistory
	if kvCache != nil {
		history = kvCache.(cache.Attention).Update(b, k, v)
	}
	scale := float32(1 / math.Sqrt(float64(a.headDim)))
	attended := relativeSDPA(b, q, k, v, table, sink, history, a.window, scale)
	attended = attended.Transpose(0, 2, 1, 3).Reshape(rows, tokens, a.heads*a.headDim)
	gate := projectTarget(a.Gate, qInput)
	if mlx.MetalIsAvailable() && gate.DType() == qInput.DType() {
		return projectTarget(a.Output, compiledAttentionGate(attended, gate))
	}
	return projectTarget(a.Output, gateAttention(attended, gate, qInput.DType()))
}

func gateAttention(value, gate *mlx.Array, dtype mlx.DType) *mlx.Array {
	return mlx.Mul(value.AsType(mlx.DTypeFloat32), gate.AsType(mlx.DTypeFloat32).Sigmoid()).AsType(dtype)
}

var compiledAttentionGate = mlx.Compile2("InklingAttentionGate", func(value, gate *mlx.Array) *mlx.Array {
	return gateAttention(value, gate, gate.DType())
}, mlx.Shapeless())

func clipL2(x, weight, temperature *mlx.Array, threshold float32) *mlx.Array {
	inputs := []*mlx.Array{x, mlx.FromValue(threshold)}
	if weight != nil {
		inputs = append(inputs, weight)
	}
	if temperature != nil {
		if weight == nil {
			inputs = append(inputs, mlx.FromValue(float32(1)))
		}
		inputs = append(inputs, temperature)
	}
	if mlx.MetalIsAvailable() {
		return compiledClipL2(inputs...)[0]
	}
	return clipL2Tensors(inputs...)[0]
}

var compiledClipL2 = mlx.Compile("InklingClipL2", clipL2Tensors, mlx.Shapeless())

func clipL2Tensors(inputs ...*mlx.Array) []*mlx.Array {
	x, threshold := inputs[0], inputs[1]
	values := x.AsType(mlx.DTypeFloat32)
	if len(inputs) > 2 {
		values = mlx.Mul(values, inputs[2].AsType(mlx.DTypeFloat32))
	}
	norm := mlx.Sum(mlx.Mul(values, values), -1, true).Sqrt()
	factor := mlx.Div(threshold, mlx.AddScalar(norm, residualEpsilon))
	values = mlx.Mul(values, mlx.Minimum(factor, mlx.FromValue(float32(1))))
	if len(inputs) > 3 {
		values = mlx.Mul(values, inputs[3])
	}
	return []*mlx.Array{values.AsType(x.DType())}
}

func relativeSDPA(b *batch.Batch, q, k, v, table, sink *mlx.Array, history *nn.KVHistory, window int, scale float32) *mlx.Array {
	if history != nil && history.Rows() != nil {
		output := make([]*mlx.Array, q.Dim(0))
		for i, row := range history.Rows() {
			selectRow := func(x *mlx.Array) *mlx.Array {
				return x.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice(), mlx.Slice())
			}
			output[i] = relativeSDPA(b.Row(i), selectRow(q), nil, nil, selectRow(table), selectRow(sink), row, window, scale)
		}
		return mlx.Concatenate(output, 0)
	}
	// Attention stays in FP32 through its output gate. Preserve the stored
	// BF16 keys and values, and widen only the views used by this forward.
	q = q.AsType(mlx.DTypeFloat32)
	if history == nil {
		k, v = k.AsType(mlx.DTypeFloat32), v.AsType(mlx.DTypeFloat32)
		return relativeWindowSDPA(b, q, table, sink, k.Dim(2), window, scale, nn.WithKV(k, v, b.SeqQueryLens))
	}
	queries, keys := q.Dim(2), history.K().Dim(2)
	if len(b.SeqQueryLens) == 1 && window > 0 && int(b.SeqOffsets[0])+queries >= keys {
		// Ordered cache writes include the physical query width. Remove matching
		// Q and K tails so live rows use the same positions and kernel as serial.
		if live := int(b.SeqQueryLens[0]); live > 0 && live < queries {
			pad := queries - live
			trim := func(x *mlx.Array) *mlx.Array {
				return x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, x.Dim(2)-pad), mlx.Slice())
			}
			row := &batch.Batch{InputIDs: b.InputIDs.Slice(mlx.Slice(), mlx.Slice(0, live)), SeqOffsets: b.SeqOffsets, SeqQueryLens: b.SeqQueryLens}
			h := nn.NewKVHistory(trim(history.K()), trim(history.V()), nil)
			out := relativeCachedSDPA(row, trim(q), trim(table), trim(sink), h, window, window, scale)
			return mlx.PadConstant(out, []int{2}, []int{0}, []int{pad})
		}
	}
	return relativeCachedSDPA(b, q, table, sink, history, window, 0, scale)
}

// A cache history normally supplies its own sliding mask. A trimmed chronological
// view has no storage applier, so its fallback must use the model's window.
func relativeCachedSDPA(b *batch.Batch, q, table, sink *mlx.Array, history *nn.KVHistory, window, maskWindow int, scale float32) *mlx.Array {
	queries, keys := q.Dim(2), history.K().Dim(2)
	// An ordered prefill history can contain padding beyond its last query.
	if len(b.SeqQueryLens) == 1 && b.SeqQueryLens[0] == int32(queries) &&
		(queries <= shortTargetRows || int(b.SeqOffsets[0])+queries >= keys) {
		// The local attention variant rounds the unnormalized probabilities before P times V.
		if queries > shortTargetRows && window == 1024 && table.Dim(3) == 1024 && q.Dim(1) == 24 && history.K().Dim(1) == 8 {
			firstKey := int(b.SeqOffsets[0]) + queries - keys
			if out := mlx.RelativePositionAttentionBF16Probabilities(q, history.K(), history.V(), table, sink, firstKey, window, scale); out != nil {
				return out
			}
		}
		if out := mlx.RelativePositionAttention(q, history.K(), history.V(), table, sink, window, scale); out != nil {
			return out
		}
	}
	history = history.AsType(mlx.DTypeFloat32)
	if queries == 1 || queries > shortTargetRows {
		return relativeWindowSDPA(b, q, table, sink, keys, maskWindow, scale, nn.WithKVHistory(history))
	}

	// Short blocks share their projections and cache write. Give each query
	// the same ordered window and kernel extent it sees in single-token decode.
	output := make([]*mlx.Array, queries)
	for i := range queries {
		row := &batch.Batch{
			InputIDs:     b.InputIDs.Slice(mlx.Slice(), mlx.Slice(i, i+1)),
			SeqOffsets:   []int32{b.SeqOffsets[0] + int32(i)},
			SeqQueryLens: []int32{min(max(b.SeqQueryLens[0]-int32(i), 0), 1)},
		}
		selectQuery := func(x *mlx.Array) *mlx.Array {
			return x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice())
		}
		end := keys - queries + i + 1
		start := 0
		if window > 0 {
			start = max(0, end-window)
		}
		k := history.K().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
		v := history.V().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
		output[i] = relativeWindowSDPA(row, selectQuery(q), selectQuery(table), selectQuery(sink),
			end-start, 0, scale, nn.WithKV(k, v, nil))
	}
	return mlx.Concatenate(output, 2)
}

func relativeWindowSDPA(b *batch.Batch, q, table, sink *mlx.Array, keys, window int, scale float32, kv nn.SDPAOption) *mlx.Array {
	shifted := mlx.RelativePositionDecodeBias(table, sink, keys)
	if shifted == nil {
		shifted = relativeBiasGraph(table, sink, keys)
	}
	mask := nn.DeviceCausalMask().Intersect(nn.ArrayMask(shifted))
	if window > 0 {
		mask = mask.Intersect(nn.SlidingWindowMaskDevice(b, keys, window, q.DType()))
	}
	return nn.ScaledDotProductAttention(b, q, scale, kv, nn.WithMask(mask), nn.WithSinks(mlx.Zeros(q.DType(), q.Dim(1))))
}

func relativeBiasGraph(table, sink *mlx.Array, keys int) *mlx.Array {
	queries := table.Dim(2)
	shiftedQueries := mlx.Arange(float64(keys-queries), float64(keys), 1, mlx.DTypeInt32).Reshape(1, 1, queries, 1)
	keyPositions := mlx.Arange(0, float64(keys), 1, mlx.DTypeInt32)
	distance := shiftedQueries.Subtract(keyPositions)
	index := mlx.Clip(distance, mlx.FromValue(0), mlx.FromValue(table.Dim(-1)-1))
	// The relative bias is already in the same score domain as the sink.
	bias := table.TakeAlongAxis(index, -1).AsType(mlx.DTypeFloat32)
	bias = mlx.Where(index.Equal(distance), bias, mlx.FromValue(float32(0)))
	// Shifting the real logits by the token's sink value lets the kernel use
	// a single zero-logit sink per head with the same softmax denominator.
	return bias.Subtract(sink.AsType(mlx.DTypeFloat32))
}
