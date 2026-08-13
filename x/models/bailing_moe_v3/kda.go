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

// kdaChunkSize is the block length of the chunked prefill scan. kdaSubTile is
// the sub-block used to build the intra-chunk Gram matrices so every decay
// exponent is gc_t-gc_s with t >= s (always <= 0, hence exp() is safe).
const (
	kdaChunkSize = 64
	kdaSubTile   = 16
)

type kdaChunkConsts struct {
	identity  *mlx.Array // [C, C] f32
	strict    *mlx.Array // [S, S] f32, 1 where t > s
	inclusive *mlx.Array // [S, S] f32, 1 where t >= s
}

func newKDAChunkConsts() kdaChunkConsts {
	c, sub := kdaChunkSize, kdaSubTile
	eye := make([]float32, c*c)
	for i := 0; i < c; i++ {
		eye[i*c+i] = 1
	}
	st := make([]float32, sub*sub)
	inc := make([]float32, sub*sub)
	for t := 0; t < sub; t++ {
		for s := 0; s < sub; s++ {
			if t > s {
				st[t*sub+s] = 1
			}
			if t >= s {
				inc[t*sub+s] = 1
			}
		}
	}
	return kdaChunkConsts{
		identity:  mlx.FromValues(eye, c, c),
		strict:    mlx.FromValues(st, sub, sub),
		inclusive: mlx.FromValues(inc, sub, sub),
	}
}

// kdaChunk advances the recurrence by kdaChunkSize steps with dense matrix
// ops. q, k, v, a are raw [B,C,H,D] slices, betaLogits [B,C,H]; state is the
// f32 [B,H,D,D] carry. Returns y [B,C,H,D] in outDType and the new state.
// The math matches kdaScan's step loop exactly (see the derivation in the
// comment above runKDASegments).
func kdaChunk(
	q, k, v, a, betaLogits, state *mlx.Array,
	A4, dt4, qScale *mlx.Array,
	safeGate bool, lowerBound float32,
	cc kdaChunkConsts,
) (*mlx.Array, *mlx.Array) {
	dims := q.Dims()
	B, C, H := int32(dims[0]), int32(dims[1]), int32(dims[2])
	nTiles := int(C) / kdaSubTile
	outDType := q.DType()

	qn := mlx.Mul(l2Normalize(q), qScale)
	kn := l2Normalize(k)
	vf := v.AsType(mlx.DTypeFloat32)
	beta := mlx.Sigmoid(betaLogits.AsType(mlx.DTypeFloat32)) // [B,C,H]

	gateInput := mlx.Add(a.AsType(mlx.DTypeFloat32), dt4)
	var logDecay *mlx.Array
	if safeGate {
		logDecay = mlx.MulScalar(mlx.Sigmoid(mlx.Mul(A4, gateInput)), lowerBound)
	} else {
		logDecay = mlx.MulScalar(mlx.Mul(A4, mlx.Softplus(gateInput)), -1)
	}

	// Head-major views.
	qh := mlx.Transpose(qn, 0, 2, 1, 3) // [B,H,C,D]
	kh := mlx.Transpose(kn, 0, 2, 1, 3)
	vh := mlx.Transpose(vf, 0, 2, 1, 3)
	ldh := mlx.Transpose(logDecay, 0, 2, 1, 3)
	bh := mlx.ExpandDims(mlx.Transpose(beta, 0, 2, 1), -1) // [B,H,C,1]

	gc := mlx.CumSum(ldh, 2, false, true) // log Γ_{1:t}, <= 0
	eg := mlx.Exp(gc)
	kg := mlx.Mul(kh, eg) // k_t ⊙ Γ_{1:t}
	qg := mlx.Mul(qh, eg)

	s0T := mlx.Transpose(state, 0, 1, 3, 2) // [B,H,Dk,Dv]
	bMat := mlx.Sub(vh, kg.Matmul(s0T))     // v_t − S0·(Γ_{1:t}k_t)

	sliceC := func(x *mlx.Array, i int) *mlx.Array { // [B,H,S,D]
		return x.Slice(mlx.Slice(), mlx.Slice(),
			mlx.Slice(i*kdaSubTile, (i+1)*kdaSubTile), mlx.Slice())
	}
	sliceB := func(i int) *mlx.Array { // β columns [B,H,1,S]
		bs := bh.Slice(mlx.Slice(), mlx.Slice(),
			mlx.Slice(i*kdaSubTile, (i+1)*kdaSubTile), mlx.Slice())
		return mlx.Transpose(bs, 0, 1, 3, 2)
	}

	// buildGram assembles [B,H,C,C] of β_s · Σ_j x_tj k_sj exp(gc_tj − gc_sj)
	// over the lower triangle (strict for the delta system, inclusive for the
	// output map). Off-diagonal tiles have t > s everywhere so the exponent is
	// safe; the diagonal tile clamps before exp and masks after.
	buildGram := func(x *mlx.Array, diagMask *mlx.Array) *mlx.Array {
		rows := make([]*mlx.Array, 0, nTiles)
		for ti := 0; ti < nTiles; ti++ {
			cols := make([]*mlx.Array, 0, nTiles)
			for si := 0; si < nTiles; si++ {
				if si > ti {
					cols = append(cols, mlx.Zeros(mlx.DTypeFloat32,
						int(B), int(H), kdaSubTile, kdaSubTile))
					continue
				}
				gt := mlx.ExpandDims(sliceC(gc, ti), 3) // [B,H,S,1,D]
				gs := mlx.ExpandDims(sliceC(gc, si), 2) // [B,H,1,S,D]
				diff := mlx.Sub(gt, gs)
				if si == ti {
					diff = mlx.Minimum(diff, mlx.FromValue(float32(0)))
				}
				e := mlx.Exp(diff)
				xt := mlx.ExpandDims(sliceC(x, ti), 3)
				ks := mlx.ExpandDims(sliceC(kh, si), 2)
				t := mlx.Sum(mlx.Mul(mlx.Mul(xt, ks), e), -1, false) // [B,H,S,S]
				if si == ti {
					t = mlx.Mul(t, diagMask)
				}
				cols = append(cols, mlx.Mul(t, sliceB(si)))
			}
			rows = append(rows, mlx.Concatenate(cols, 3))
		}
		return mlx.Concatenate(rows, 2)
	}

	aMat := buildGram(kh, cc.strict) // strictly-lower system matrix
	mMat := buildGram(qh, cc.inclusive)

	// Solve (I + A) u = b exactly: A is strictly lower triangular, hence
	// nilpotent (A^C = 0), and Σ (−A)^k = Π_i (I + (−A)^(2^i)).
	n := mlx.MulScalar(aMat, -1)
	r := mlx.Add(cc.identity, n)
	p := n
	for i := 0; i < 5; i++ {
		p = p.Matmul(p)
		r = r.Matmul(mlx.Add(cc.identity, p))
	}
	u := r.Matmul(bMat) // [B,H,C,Dv]

	y := mlx.Add(qg.Matmul(s0T), mMat.Matmul(u))

	// Chunk-final state: S_C = S0 ⊙ Γ_{1:C} (key cols) + Σ_s β_s u_s (k_s ⊙ Γ_{s+1:C})ᵀ
	gcLast := gc.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(int(C-1), int(C)), mlx.Slice()) // [B,H,1,D]
	kf := mlx.Mul(kh, mlx.Exp(mlx.Sub(gcLast, gc)))
	bu := mlx.Mul(u, bh)
	newState := mlx.Add(
		mlx.Mul(state, mlx.Exp(gcLast)),
		mlx.Transpose(bu, 0, 1, 3, 2).Matmul(kf),
	)

	return mlx.Transpose(y, 0, 2, 1, 3).AsType(outDType), newState
}

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

	step := kdaStepSoftplus
	if safeGate {
		step = kdaStepSafeGate
	}
	qScale := mlx.FromValue(float32(1 / math.Sqrt(float64(D))))
	lb := mlx.FromValue(lowerBound)
	dt4 := mlx.Reshape(dtBias.AsType(mlx.DTypeFloat32), 1, 1, H, D)
	A4 := mlx.Reshape(aExp.AsType(mlx.DTypeFloat32), 1, 1, H, 1)
	dt3 := mlx.Reshape(dtBias.AsType(mlx.DTypeFloat32), 1, H, D)
	A3 := mlx.Reshape(aExp.AsType(mlx.DTypeFloat32), 1, H, 1)
	cc := newKDAChunkConsts()

	outputs := make([]*mlx.Array, 0, int(T)/kdaChunkSize+kdaChunkSize)
	t := int32(0)
	for ; t+kdaChunkSize <= T; t += kdaChunkSize {
		r := timeRange{start: t, end: t + kdaChunkSize}
		var y *mlx.Array
		y, state = kdaChunk(
			sliceTime(q, r), sliceTime(k, r), sliceTime(v, r),
			sliceTime(a, r), sliceTime(betaLogits, r), state,
			A4, dt4, qScale, safeGate, lowerBound, cc,
		)
		outputs = append(outputs, y)
	}
	for ; t < T; t++ {
		r := timeRange{start: t, end: t + 1}
		outs := step(
			mlx.Squeeze(sliceTime(q, r), 1), mlx.Squeeze(sliceTime(k, r), 1),
			mlx.Squeeze(sliceTime(v, r), 1), mlx.Squeeze(sliceTime(a, r), 1),
			mlx.Squeeze(sliceTime(betaLogits, r), 1), state,
			A3, dt3, qScale, lb,
		)
		outputs = append(outputs, mlx.ExpandDims(outs[0], 1))
		state = outs[1]
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
