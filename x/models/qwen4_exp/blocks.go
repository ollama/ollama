package qwen4_exp

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c, side cache.Cache, positions, ropePositions *mlx.Array, cfg *Config) *mlx.Array {
	branch, state := l.AttentionConnection.Prepare(x, cfg)
	if l.Linear != nil {
		branch = l.Linear.Forward(branch, b, c, cfg)
	} else {
		branch = l.Attention.Forward(branch, b, c, side, positions, ropePositions, cfg)
	}
	x = l.AttentionConnection.Inject(state, branch, cfg)

	branch, state = l.MLPConnection.Prepare(x, cfg)
	branch = l.MoE.Forward(branch, cfg)
	return l.MLPConnection.Inject(state, branch, cfg)
}

func (a *linearAttention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	valueDim := cfg.LinearNumValueHeads * cfg.LinearValueHeadDim

	qkv := a.QKV.Forward(x)
	z := mlx.Reshape(a.Z.Forward(x), B, L, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim)
	ba := mlx.Concatenate([]*mlx.Array{a.B.Forward(x), a.A.Forward(x)}, -1)

	convTail := cfg.LinearConvKernelDim - 1
	options := []nn.RecurrentOption{nn.WithConvSiLU()}
	var recurrent *cache.RecurrentCache
	if value, ok := c.(*cache.RecurrentCache); ok {
		recurrent = value
		options = append(options, nn.WithRecurrentHistory(recurrent.Get(b, x.DType())))
		if splits := recurrent.SnapshotSplits(int(L)); len(splits) > 0 {
			options = append(options, nn.WithSnapshotSplits(splits))
		}
	} else {
		options = append(options, nn.WithRecurrentState(
			mlx.Zeros(x.DType(), int(B), int(convTail), qkv.Dim(2)),
			mlx.Zeros(mlx.DTypeFloat32, int(B), int(cfg.LinearNumValueHeads), int(cfg.LinearValueHeadDim), int(cfg.LinearKeyHeadDim)),
		))
	}

	convOut, convState := nn.CausalConv1D(b, qkv, a.Conv, int(convTail), options...)
	out, deltaState := nn.GatedDelta(b, convOut, ba, a.DtBias, a.AExp, options...)
	outType := out.DType()
	out = mlx.RMSNormFn(out, a.NormWeight, cfg.RMSNormEps)
	// Qwen 4 uses a sigmoid Gated DeltaNet output gate rather than the SiLU
	// gate used by Qwen3.5.
	gate := mlx.Sigmoid(z.AsType(mlx.DTypeFloat32))
	out = mlx.Mul(out.AsType(mlx.DTypeFloat32), gate).AsType(outType)
	out = a.Out.Forward(mlx.Reshape(out, B, L, valueDim))
	if recurrent != nil {
		recurrent.Put(b, convState, deltaState)
	}
	return out
}

func (a *fullAttention) Forward(x *mlx.Array, b *batch.Batch, c, side cache.Cache, positions, ropePositions *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	qg := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim*2)
	q := mlx.SliceStartStop(qg, []int32{0, 0, 0, 0}, []int32{B, L, cfg.NumAttentionHeads, cfg.HeadDim})
	gate := mlx.SliceStartStop(qg, []int32{0, 0, 0, cfg.HeadDim}, []int32{B, L, cfg.NumAttentionHeads, cfg.HeadDim * 2})
	gate = mlx.Reshape(gate, B, L, cfg.NumAttentionHeads*cfg.HeadDim)

	k := mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := mlx.Reshape(a.VProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	q = a.QNorm.Forward(q, cfg.RMSNormEps)
	k = a.KNorm.Forward(k, cfg.RMSNormEps)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	cos, sin := qsaRopeCosSin(cfg, ropePositions)
	q = applyQwenRoPE(q, cos, sin, cfg.RopeDim)
	k = applyQwenRoPE(k, cos, sin, cfg.RopeDim)

	projected := a.Indexer.QKProj.Forward(x)
	indexQ := mlx.SliceStartStop(projected,
		[]int32{0, 0, 0},
		[]int32{B, L, cfg.IndexerNumHeads * cfg.IndexerHeadDim},
	)
	rawIndexK := mlx.SliceStartStop(projected,
		[]int32{0, 0, cfg.IndexerNumHeads * cfg.IndexerHeadDim},
		[]int32{B, L, (cfg.IndexerNumHeads + cfg.IndexerKVHeads) * cfg.IndexerHeadDim},
	)
	indexQ = mlx.Reshape(indexQ, B, L, cfg.IndexerNumHeads, cfg.IndexerHeadDim)
	indexQ = a.Indexer.QNorm.Forward(indexQ, cfg.RMSNormEps)
	indexQ = mlx.Transpose(indexQ, 0, 2, 1, 3)
	indexQ = applyQwenRoPE(indexQ, cos, sin, cfg.RopeDim)
	rawIndexK = mlx.Transpose(
		mlx.Reshape(rawIndexK, B, L, cfg.IndexerKVHeads, cfg.IndexerHeadDim),
		0, 2, 1, 3,
	)

	var mainHistory, indexHistory *nn.KVHistory
	if c != nil {
		mainHistory = c.(cache.Attention).Update(b, k, v)
		if side == nil {
			panic("QSA side cache is missing")
		}
		indexHistory = side.(cache.Attention).Update(b, rawIndexK, ropePositions)
	}

	var out *mlx.Array
	keyLength := L
	for _, offset := range b.SeqOffsets {
		keyLength = max(keyLength, offset+L)
	}
	if mainHistory == nil {
		out = nn.ScaledDotProductAttention(b, q, cfg.Scale,
			nn.WithKV(k, v, b.SeqQueryLens), nn.WithMask(nn.CausalMask()))
	} else if keyLength <= cfg.IndexerBudget {
		// QSA selects the entire visible history within the indexer budget.
		out = nn.ScaledDotProductAttention(b, q, cfg.Scale,
			nn.WithKVHistory(mainHistory), nn.WithMask(nn.CausalMask()))
	} else {
		compressed := qsaCompressedKeys(indexHistory.K(), indexHistory.V(), a.Indexer, cfg)
		scores := mlx.Matmul(indexQ.AsType(mlx.DTypeFloat32), mlx.Transpose(compressed.AsType(mlx.DTypeFloat32), 0, 1, 3, 2))
		scores = mlx.DivScalar(mlx.Sum(mlx.ReLU(scores), 1, false), float32(math.Sqrt(float64(cfg.IndexerHeadDim))))
		logical, valid := qsaLogicalIndices(scores, b, keyLength, cfg)
		out = qsaSparseAttention(q, mainHistory, logical, valid, cfg)
	}
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	out = mlx.Mul(out, mlx.Sigmoid(gate))
	return a.OProj.Forward(out)
}

func splitLastAxis(a *mlx.Array) (*mlx.Array, *mlx.Array) {
	dims := a.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, dim := range dims {
		stop[i] = int32(dim)
	}
	mid := int32(dims[len(dims)-1]) / 2
	leftStop := append([]int32(nil), stop...)
	leftStop[len(leftStop)-1] = mid
	rightStart := append([]int32(nil), start...)
	rightStart[len(rightStart)-1] = mid
	return mlx.SliceStartStop(a, start, leftStop), mlx.SliceStartStop(a, rightStart, stop)
}

func (m *sparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	probs := mlx.SoftmaxAxis(m.Gate.Forward(x), -1, true)
	indices := mlx.Argpartition(mlx.Neg(probs), int(topK)-1, -1)
	shape := indices.Dims()
	indices = mlx.SliceStartStop(indices, []int32{0, 0, 0}, []int32{int32(shape[0]), int32(shape[1]), topK})
	scores := mlx.TakeAlongAxis(probs, indices, -1)
	scores = mlx.Div(scores, mlx.Sum(scores, -1, true))

	xFlat := mlx.Reshape(mlx.ExpandDims(mlx.ExpandDims(x, -2), -2), B*L, 1, 1, cfg.HiddenSize)
	indexFlat := mlx.Reshape(indices, B*L, topK)
	doSort := B*L >= 64
	var inverse *mlx.Array
	if doSort {
		all := mlx.Flatten(indexFlat)
		order := mlx.Argsort(all, 0)
		inverse = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		indexFlat = mlx.Reshape(mlx.Take(all, order, 0), B*L*topK, 1)
	}

	var gateUp *mlx.Array
	if m.GateUpScales != nil {
		gateUp = mlx.GatherQMM(xFlat, m.GateUpExperts, m.GateUpScales, m.GateUpBiases,
			nil, indexFlat, true, m.GateUpGroup, m.GateUpBits, m.GateUpMode, doSort)
	} else {
		gateUp = mlx.GatherMM(xFlat, m.GateUpExperts, nil, indexFlat, doSort)
	}
	gate, up := splitLastAxis(gateUp)
	hidden := mlx.SwiGLU(gate, up)
	var down *mlx.Array
	if m.DownScales != nil {
		down = mlx.GatherQMM(hidden, m.DownExperts, m.DownScales, m.DownBiases,
			nil, indexFlat, true, m.DownGroup, m.DownBits, m.DownMode, doSort)
	} else {
		down = mlx.GatherMM(hidden, m.DownExperts, nil, indexFlat, doSort)
	}
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), inverse, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	differentiated := mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
	out := mlx.Sum(mlx.Mul(differentiated, mlx.ExpandDims(scores, -1)), 2, false)

	shared := m.SharedDownProj.Forward(mlx.SwiGLU(m.SharedGateProj.Forward(x), m.SharedUpProj.Forward(x)))
	shared = mlx.Mul(shared, mlx.Sigmoid(m.SharedGate.Forward(x)))
	return mlx.Add(out, shared)
}
