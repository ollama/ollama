package bailing_moe_v3

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

// KDAAttention implements Kimi Delta Attention with per-key-dimension decay.
// Bailing stores q/k/v and their depthwise short convolutions separately; they
// are packed at load/forward time so the shared recurrent cache can keep one
// convolution state tensor.
type KDAAttention struct {
	QProj nn.LinearLayer
	KProj nn.LinearLayer
	VProj nn.LinearLayer
	FProj nn.LinearLayer
	BProj nn.LinearLayer
	GProj nn.LinearLayer
	OProj nn.LinearLayer

	// QKVProj and FBGProj, when non-nil, are the row-fused equivalents of
	// (QProj,KProj,VProj) and (FProj,BProj,GProj): one wide matmul replaces
	// three thin ones per forward, and the qkv fusion also replaces the
	// explicit concatenation feeding the shared short convolution.
	QKVProj nn.LinearLayer
	FBGProj nn.LinearLayer

	Conv1D      *nn.Conv1d
	ONormWeight *mlx.Array
	DtBias      *mlx.Array
	AExp        *mlx.Array
}

func sanitizeConvWeight(w *mlx.Array) *mlx.Array {
	if w == nil {
		return nil
	}
	if w.NumDims() == 3 {
		if w.Dim(1) == 1 {
			return mlx.Squeeze(w, 1)
		}
		if w.Dim(2) == 1 {
			return mlx.Squeeze(w, 2)
		}
	}
	return w
}

func loadKDAAttention(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, cfg *Config) (*KDAAttention, error) {
	p := prefix + ".attention"
	a := &KDAAttention{
		QProj:       linears.Make(p + ".q_proj"),
		KProj:       linears.Make(p + ".k_proj"),
		VProj:       linears.Make(p + ".v_proj"),
		FProj:       linears.Make(p + ".f_proj"),
		BProj:       linears.Make(p + ".b_proj"),
		GProj:       linears.Make(p + ".g_proj"),
		OProj:       linears.Make(p + ".o_proj"),
		ONormWeight: tensors[p+".o_norm.weight"],
		DtBias:      tensors[p+".dt_bias"],
	}
	if a.QProj == nil || a.KProj == nil || a.VProj == nil || a.FProj == nil ||
		a.BProj == nil || a.GProj == nil || a.OProj == nil || a.ONormWeight == nil || a.DtBias == nil {
		return nil, fmt.Errorf("missing KDA projection or state tensor")
	}
	aLog := tensors[p+".A_log"]
	if aLog == nil {
		return nil, fmt.Errorf("missing KDA A_log")
	}
	a.AExp = mlx.Exp(aLog.AsType(mlx.DTypeFloat32)).Clone()

	qConv := sanitizeConvWeight(tensors[p+".q_conv1d.weight"])
	kConv := sanitizeConvWeight(tensors[p+".k_conv1d.weight"])
	vConv := sanitizeConvWeight(tensors[p+".v_conv1d.weight"])
	if qConv == nil || kConv == nil || vConv == nil || qConv.NumDims() != 2 || kConv.NumDims() != 2 || vConv.NumDims() != 2 {
		return nil, fmt.Errorf("invalid KDA short convolution weights")
	}
	convWeight := mlx.Concatenate([]*mlx.Array{qConv, kConv, vConv}, 0).Clone()
	mlx.Eval(convWeight, a.AExp)
	a.Conv1D = nn.NewConv1d(mlx.ExpandDims(convWeight, 2), nil, 1, 0, 1, int32(convWeight.Dim(0)))
	delete(tensors, p+".q_conv1d.weight")
	delete(tensors, p+".k_conv1d.weight")
	delete(tensors, p+".v_conv1d.weight")

	// Row-fuse the six input projections into two wide matmuls. The fusion
	// is bit-exact (quant groups run along the input axis), so a nil result
	// just means mixed representations; the per-projection path still works.
	a.QKVProj = fuseLinearRows(a.QProj, a.KProj, a.VProj)
	if a.QKVProj != nil {
		// Forward uses only the fused projection; drop the originals so the
		// loader's pin pass does not keep both copies resident.
		a.QProj, a.KProj, a.VProj = nil, nil, nil
	}
	a.FBGProj = fuseLinearRows(a.FProj, a.BProj, a.GProj)
	if a.FBGProj != nil {
		a.FProj, a.BProj, a.GProj = nil, nil, nil
	}
	return a, nil
}

type timeRange struct{ start, end int32 }

func timeRanges(splits []int, length int32) []timeRange {
	ranges := make([]timeRange, 0, len(splits)+1)
	start := int32(0)
	for _, split := range splits {
		ranges = append(ranges, timeRange{start: start, end: int32(split)})
		start = int32(split)
	}
	return append(ranges, timeRange{start: start, end: length})
}

func sliceTime(x *mlx.Array, r timeRange) *mlx.Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[1], stop[1] = r.start, r.end
	return mlx.SliceStartStop(x, start, stop)
}

func l2Normalize(x *mlx.Array) *mlx.Array {
	x = x.AsType(mlx.DTypeFloat32)
	norm := mlx.RSqrt(mlx.AddScalar(mlx.Sum(mlx.Mul(x, x), -1, true), 1e-6))
	return mlx.Mul(x, norm)
}

// kdaStepGraph builds the single-timestep KDA recurrence as one compilable
// graph: L2 normalization, the decay gate, and the delta rule fused into a
// handful of Metal kernels instead of ~20 tiny ones. Inputs: q, k, v, a
// [B,H,D]; beta [B,H]; state [B,H,D,D] (f32); A [1,H,1]; dt [1,H,D]; qScale
// and lowerBound scalars. Outputs: y [B,H,D] in q's dtype and the new f32
// state. The math matches kdaScan's loop body op for op.
func kdaStepGraph(safeGate bool) mlx.CompileFunc {
	return func(in ...*mlx.Array) []*mlx.Array {
		q, k, v, a, beta, state, A, dt, qScale, lowerBound := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9]
		outDType := q.DType()
		qn := mlx.Mul(l2Normalize(q), qScale)
		kn := l2Normalize(k)
		vf := v.AsType(mlx.DTypeFloat32)
		b := mlx.ExpandDims(mlx.Sigmoid(beta.AsType(mlx.DTypeFloat32)), -1)

		gateInput := mlx.Add(a.AsType(mlx.DTypeFloat32), dt)
		var logDecay *mlx.Array
		if safeGate {
			logDecay = mlx.Mul(mlx.Sigmoid(mlx.Mul(A, gateInput)), lowerBound)
		} else {
			logDecay = mlx.MulScalar(mlx.Mul(A, mlx.Softplus(gateInput)), -1)
		}
		decay := mlx.Exp(logDecay)

		kx := mlx.ExpandDims(kn, 2)
		s := mlx.Mul(state, mlx.ExpandDims(decay, 2))
		memory := mlx.Sum(mlx.Mul(s, kx), -1, false)
		delta := mlx.Mul(mlx.Sub(vf, memory), b)
		s = mlx.Add(s, mlx.Mul(kx, mlx.ExpandDims(delta, -1)))
		y := mlx.Sum(mlx.Mul(s, mlx.ExpandDims(qn, 2)), -1, false)
		return []*mlx.Array{y.AsType(outDType), s}
	}
}

var (
	kdaStepSafeGate = mlx.Compile("KDAStepSafeGate", kdaStepGraph(true), mlx.Shapeless())
	kdaStepSoftplus = mlx.Compile("KDAStepSoftplus", kdaStepGraph(false), mlx.Shapeless())
)

// kdaOutGate fuses the KDA output gate out * sigmoid(gate) computed in f32.
var kdaOutGate = mlx.Compile2("KDAOutGate", func(out, gate *mlx.Array) *mlx.Array {
	dt := out.DType()
	return mlx.Mul(out.AsType(mlx.DTypeFloat32), mlx.Sigmoid(gate.AsType(mlx.DTypeFloat32))).AsType(dt)
}, mlx.Shapeless())

// kdaScan is the graph reference for the exact FLA KDA recurrence used by the
// checkpoint. It intentionally favors correctness for initial bring-up; a
// fused Metal implementation can replace it without changing model code.
// Single-token calls (every decode step) take the compiled fused-step path,
// which computes the same graph with far fewer kernel launches.
func kdaScan(q, k, v, a, betaLogits, aExp, dtBias, state *mlx.Array, safeGate bool, lowerBound float32) (*mlx.Array, *mlx.Array) {
	dims := q.Dims()
	B, T, H, D := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	outDType := q.DType()

	if T == 1 {
		step := kdaStepSoftplus
		if safeGate {
			step = kdaStepSafeGate
		}
		outs := step(
			mlx.Squeeze(q, 1), mlx.Squeeze(k, 1), mlx.Squeeze(v, 1),
			mlx.Squeeze(a, 1), mlx.Squeeze(betaLogits, 1), state,
			mlx.Reshape(aExp.AsType(mlx.DTypeFloat32), 1, H, 1),
			mlx.Reshape(dtBias.AsType(mlx.DTypeFloat32), 1, H, D),
			mlx.FromValue(float32(1/math.Sqrt(float64(D)))),
			mlx.FromValue(lowerBound),
		)
		return mlx.ExpandDims(outs[0], 1), outs[1]
	}

	q = mlx.MulScalar(l2Normalize(q), float32(1/math.Sqrt(float64(D))))
	k = l2Normalize(k)
	v = v.AsType(mlx.DTypeFloat32)
	a = a.AsType(mlx.DTypeFloat32)
	beta := mlx.Sigmoid(betaLogits.AsType(mlx.DTypeFloat32))

	dt := mlx.Reshape(dtBias.AsType(mlx.DTypeFloat32), 1, 1, H, D)
	A := mlx.Reshape(aExp.AsType(mlx.DTypeFloat32), 1, 1, H, 1)
	gateInput := mlx.Add(a, dt)
	var logDecay *mlx.Array
	if safeGate {
		logDecay = mlx.MulScalar(mlx.Sigmoid(mlx.Mul(A, gateInput)), lowerBound)
	} else {
		logDecay = mlx.MulScalar(mlx.Mul(A, mlx.Softplus(gateInput)), -1)
	}
	decay := mlx.Exp(logDecay)

	outputs := make([]*mlx.Array, 0, T)
	for t := int32(0); t < T; t++ {
		r := timeRange{start: t, end: t + 1}
		qt := mlx.Squeeze(sliceTime(q, r), 1)
		kt := mlx.Squeeze(sliceTime(k, r), 1)
		vt := mlx.Squeeze(sliceTime(v, r), 1)
		gt := mlx.Squeeze(sliceTime(decay, r), 1)
		bt := mlx.Squeeze(sliceTime(beta, r), 1)

		kExpanded := mlx.ExpandDims(kt, 2)
		state = mlx.Mul(state, mlx.ExpandDims(gt, 2))
		memory := mlx.Sum(mlx.Mul(state, kExpanded), -1, false)
		delta := mlx.Mul(mlx.Sub(vt, memory), mlx.ExpandDims(bt, -1))
		state = mlx.Add(state, mlx.Mul(kExpanded, mlx.ExpandDims(delta, -1)))
		y := mlx.Sum(mlx.Mul(state, mlx.ExpandDims(qt, 2)), -1, false)
		outputs = append(outputs, mlx.ExpandDims(y.AsType(outDType), 1))
	}
	if len(outputs) == 0 {
		return mlx.Zeros(outDType, int(B), 0, int(H), int(D)), state
	}
	return mlx.Concatenate(outputs, 1), state
}

func runKDASegments(q, k, v, a, beta, aExp, dtBias, initial *mlx.Array, splits []int, safeGate bool, lowerBound float32) (*mlx.Array, []*mlx.Array) {
	length := int32(q.Dim(1))
	outs := make([]*mlx.Array, 0, len(splits)+1)
	states := make([]*mlx.Array, 0, len(splits)+1)
	state := initial
	for _, r := range timeRanges(splits, length) {
		out, next := kdaScan(
			sliceTime(q, r), sliceTime(k, r), sliceTime(v, r),
			sliceTime(a, r), sliceTime(beta, r), aExp, dtBias, state,
			safeGate, lowerBound,
		)
		outs = append(outs, out)
		states = append(states, next)
		state = next
	}
	return mlx.Concatenate(outs, 1), states
}

func (a *KDAAttention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, _ *mlx.Array, B, L int32, cfg *Config) *mlx.Array {
	projectionDim := cfg.NumAttentionHeads * cfg.HeadDim
	var qkv *mlx.Array
	if a.QKVProj != nil {
		qkv = a.QKVProj.Forward(x)
	} else {
		qkv = mlx.Concatenate([]*mlx.Array{
			a.QProj.Forward(x),
			a.KProj.Forward(x),
			a.VProj.Forward(x),
		}, 2)
	}

	convTail := cfg.ShortConvKernelSize - 1
	var rc *cache.RecurrentCache
	var history *nn.RecurrentHistory
	opts := []nn.RecurrentOption{nn.WithConvSiLU()}
	var splits []int
	if typed, ok := c.(*cache.RecurrentCache); ok {
		rc = typed
		history = rc.Get(b, x.DType())
		opts = append(opts, nn.WithRecurrentHistory(history))
		splits = rc.SnapshotSplits(int(L))
		if len(splits) > 0 {
			opts = append(opts, nn.WithSnapshotSplits(splits))
		}
	} else {
		history = nn.NewRecurrentHistory(
			mlx.Zeros(x.DType(), int(B), int(convTail), int(3*projectionDim)),
			mlx.Zeros(mlx.DTypeFloat32, int(B), int(cfg.NumAttentionHeads), int(cfg.HeadDim), int(cfg.HeadDim)),
		)
		opts = append(opts, nn.WithRecurrentHistory(history))
	}

	convOut, convStates := nn.CausalConv1D(b, qkv, a.Conv1D, int(convTail), opts...)
	q := mlx.SliceStartStop(convOut, []int32{0, 0, 0}, []int32{B, L, projectionDim})
	k := mlx.SliceStartStop(convOut, []int32{0, 0, projectionDim}, []int32{B, L, 2 * projectionDim})
	v := mlx.SliceStartStop(convOut, []int32{0, 0, 2 * projectionDim}, []int32{B, L, 3 * projectionDim})
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k = mlx.Reshape(k, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	v = mlx.Reshape(v, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	var decayInput, beta, gateFlat *mlx.Array
	if a.FBGProj != nil {
		fbg := a.FBGProj.Forward(x)
		fDim := projectionDim
		bDim := cfg.NumAttentionHeads
		decayInput = mlx.Reshape(sliceCols(fbg, B, L, 0, fDim), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
		beta = mlx.Reshape(sliceCols(fbg, B, L, fDim, fDim+bDim), B, L, cfg.NumAttentionHeads)
		gateFlat = sliceCols(fbg, B, L, fDim+bDim, fDim+bDim+fDim)
	} else {
		decayInput = mlx.Reshape(a.FProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
		beta = mlx.Reshape(a.BProj.Forward(x), B, L, cfg.NumAttentionHeads)
		gateFlat = a.GProj.Forward(x)
	}

	out, deltaStates := runKDASegments(
		q, k, v, decayInput, beta, a.AExp, a.DtBias, history.DeltaState(), splits,
		cfg.KDASafeGate, cfg.KDALowerBound,
	)
	if rc != nil {
		rc.Put(b, convStates, deltaStates)
	}

	gate := mlx.Reshape(gateFlat, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	out = mlx.RMSNormFn(out, a.ONormWeight, cfg.RMSNormEps)
	out = kdaOutGate(out, gate)
	out = mlx.Reshape(out, B, L, projectionDim)
	return a.OProj.Forward(out)
}
