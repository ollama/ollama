package mlx

// Fused Metal kernels for one Olala decode step (one token per row).
//
// Ported from mlx_engine/models/olala/metal.py in OVHai's mlx-engine fork
// (https://github.com/OVHai-LLM/mlx-engine, MIT license). A decode
// step runs as a short chain of fused kernels per layer instead of the ~60
// small ops of the graph path:
//
//	M layer: gemv[geo](in_proj) -> m3 step -> gemv(mixer_proj)
//	         -> gemv[geo](router | in_cat) -> routed experts -> gemv[relu2cat](out_cat)
//	V layer: gemv[geo](in_proj, in_proj_small) -> v prep -> KV append -> attention
//	         -> attention combine -> gemv(mixer_proj) -> MoE as above
//
// Activations stay in f32. The GEMV kernels read f32 / bf16 / f16 or affine
// 4 / 8-bit weights. The geodesic residual update of the previous sub-layer
// runs as the prologue of the next matrix-vector product.

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
)

const olalaHeader = `
#include <metal_simdgroup>
using namespace metal;

inline float olala_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-fabs(x))); }
inline float olala_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float4 relu2(float4 v) { v = max(v, 0.0f); return v * v; }
`

const (
	olalaNSG        = 16        // simdgroups per threadgroup (GEMV with a computed input, m3, MoE)
	olalaNSGDev     = 8         // simdgroups per GEMV threadgroup reading x from device memory
	olalaTGXBytes   = 28 * 1024 // threadgroup memory budget for a computed GEMV input
	olalaSplitBytes = 4000      // rows heavier than this are split across simdgroups
	olalaM3PB       = 64        // state rows per threadgroup in the m3 step
	olalaMoEFB      = 64        // expert hidden units per threadgroup
	olalaAttNW      = 4         // simdgroups per attention threadgroup
	olalaAttBlk     = 256       // keys per attention threadgroup (long contexts)
	olalaAttShort   = 1024      // contexts up to this many keys use the short kernel
)

var (
	olalaKernelsMu sync.Mutex
	olalaKernels   = map[string]*gpuKernel{}
)

func olalaKernel(key string, build func() *gpuKernel) *gpuKernel {
	olalaKernelsMu.Lock()
	defer olalaKernelsMu.Unlock()
	if k, ok := olalaKernels[key]; ok {
		return k
	}
	k := build()
	olalaKernels[key] = k
	return k
}

// flt formats a float constant for Metal source.
func flt(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".e") {
		s += ".0"
	}
	return s + "f"
}

// OlalaWeight is a (rows, cols) weight matrix (or a stack of them) in a
// format the fused kernels read.
type OlalaWeight struct {
	W, Scales, Biases *Array
	Rows, Cols        int
	GroupSize         int
	format            string
	// Views are 16-element views of the arrays above, exported so that
	// mlx.Collect keeps them alive with the model.
	Views []*Array
}

// NewOlalaWeight wraps a weight for the fused kernels. It returns nil for
// formats the kernels do not read.
func NewOlalaWeight(w, scales, biases *Array, bits, groupSize int) *OlalaWeight {
	if w == nil {
		return nil
	}
	dims := w.Dims()
	n := len(dims)
	if n < 2 {
		return nil
	}
	ow := &OlalaWeight{W: w, Scales: scales, Biases: biases, Rows: dims[n-2]}
	if scales != nil {
		if biases == nil || (bits != 4 && bits != 8) || (groupSize != 32 && groupSize != 64 && groupSize != 128) {
			return nil
		}
		ow.format = "q" + strconv.Itoa(bits)
		ow.GroupSize = groupSize
		ow.Cols = dims[n-1] * 32 / bits
	} else {
		switch w.DType() {
		case DTypeBFloat16:
			ow.format = "bf16"
		case DTypeFloat16:
			ow.format = "f16"
		case DTypeFloat32:
			ow.format = "f32"
		default:
			return nil
		}
		ow.Cols = dims[n-1]
	}
	return ow
}

func (w *OlalaWeight) quantized() bool { return w.Scales != nil }

func (w *OlalaWeight) key() string { return w.format + strconv.Itoa(w.GroupSize) }

// arrays returns 16-element views of the weight arrays. The kernels index the
// whole underlying buffer through them. MLX commits a command buffer once the
// bytes of the arrays bound in it exceed a budget, so binding the full expert
// or head tensors would force a commit almost every kernel.
func (w *OlalaWeight) arrays() []*Array {
	if w.Views == nil {
		src := []*Array{w.W}
		if w.quantized() {
			src = append(src, w.Scales, w.Biases)
		}
		for _, a := range src {
			w.Views = append(w.Views, SliceStartStop(Flatten(a), []int32{0}, []int32{16}))
		}
	}
	return w.Views
}

// Eval materializes the weight views.
func (w *OlalaWeight) Eval() { Eval(w.arrays()...) }

func olalaChunk(format string) int {
	switch format {
	case "f32":
		return 4
	case "bf16", "f16":
		return 8
	default: // q8, q4
		return 16
	}
}

func olalaBytesPerElement(format string) float64 {
	switch format {
	case "f32":
		return 4
	case "bf16", "f16":
		return 2
	case "q8":
		return 1
	default:
		return 0.5
	}
}

type olalaXExpr func(t, j string) string

// olalaDotCode accumulates acc[r][t] += dot(W[row, k:k+chunk], x[t, k:k+chunk])
// for one 16-byte chunk. xexpr(t, j) gives a float4 of x for token t at k + 4*j.
func olalaDotCode(format, name string, K, gs int, xexpr olalaXExpr, tdim int, rowvar, kvar string) string {
	var lines []string
	add := func(s string, a ...any) { lines = append(lines, fmt.Sprintf(s, a...)) }
	switch format {
	case "bf16", "f16":
		vt := "bfloat4"
		if format == "f16" {
			vt = "half4"
		}
		add("const device %s* wp = (const device %s*)(%s + (size_t)%s * %d + %s);", vt, vt, name, rowvar, K, kvar)
		add("const float4 wv0 = float4(wp[0]), wv1 = float4(wp[1]);")
		add("for (int t = 0; t < %d; ++t) acc[r][t] += dot(wv0, %s) + dot(wv1, %s);", tdim, xexpr("t", "0"), xexpr("t", "1"))
	case "f32":
		add("const float4 wv0 = *(const device float4*)(%s + (size_t)%s * %d + %s);", name, rowvar, K, kvar)
		add("for (int t = 0; t < %d; ++t) acc[r][t] += dot(wv0, %s);", tdim, xexpr("t", "0"))
	case "q8", "q4":
		if format == "q8" {
			add("const uint4 q = *(const device uint4*)(%s + (size_t)%s * (%d / 4) + %s / 4);", name, rowvar, K, kvar)
		} else {
			add("const uint2 q = *(const device uint2*)(%s + (size_t)%s * (%d / 8) + %s / 8);", name, rowvar, K, kvar)
		}
		add("const size_t gi = (size_t)%s * (%d / %d) + %s / %d;", rowvar, K, gs, kvar, gs)
		add("const float sc = float(%s_s[gi]), bi = float(%s_b[gi]);", name, name)
		add("float4 f[4];")
		if format == "q8" {
			add("for (int j = 0; j < 4; ++j) { const uint u = q[j]; f[j] = float4(u & 0xffu, (u >> 8) & 0xffu, (u >> 16) & 0xffu, u >> 24); }")
		} else {
			add("for (int j = 0; j < 2; ++j) { const uint u = q[j]; " +
				"f[2*j] = float4(u & 0xfu, (u >> 4) & 0xfu, (u >> 8) & 0xfu, (u >> 12) & 0xfu); " +
				"f[2*j+1] = float4((u >> 16) & 0xfu, (u >> 20) & 0xfu, (u >> 24) & 0xfu, u >> 28); }")
		}
		if tdim > 1 {
			// several tokens share the chunk: dequantize once, then plain dot products
			add("for (int j = 0; j < 4; ++j) f[j] = f[j] * sc + bi;")
			add("for (int t = 0; t < %d; ++t) { float s1 = 0;", tdim)
			add("  for (int j = 0; j < 4; ++j) s1 += dot(f[j], %s);", xexpr("t", "j"))
			add("  acc[r][t] += s1; }")
		} else {
			add("for (int t = 0; t < %d; ++t) { float s1 = 0, s2 = 0;", tdim)
			add("  for (int j = 0; j < 4; ++j) { const float4 xv = %s; s1 += dot(f[j], xv); s2 += xv.x + xv.y + xv.z + xv.w; }", xexpr("t", "j"))
			add("  acc[r][t] += sc * s1 + bi * s2; }")
		}
	default:
		panic("olala: unsupported weight format " + format)
	}
	return strings.Join(lines, "\n")
}

// ---------------------------------------------------------------------------
// GEMV with fused prologue

func olalaGeoPrologue(K, nsg int) string {
	NT := nsg * 32
	EPT := (K + NT - 1) / NT
	q := flt(math.Pi / 4)
	return fmt.Sprintf(`
    {
    threadgroup float red[3][%[1]d];
    const float scale = geo[0], gbias = geo[1];
    for (int t = 0; t < TN; ++t) {
        float av[%[2]d], bv[%[2]d];
        float sxx = 0, sxg = 0;
        for (int j = 0; j < %[2]d; ++j) {
            const uint i = tid + j * %[3]d;
            av[j] = i < %[4]d ? xin[t * %[4]d + i] : 0.0f;
            bv[j] = i < %[4]d ? g[t * %[4]d + i] : 0.0f;
            sxx += av[j] * av[j]; sxg += av[j] * bv[j];
        }
        sxx = simd_sum(sxx); sxg = simd_sum(sxg);
        if (lane == 0) { red[0][sg] = sxx; red[1][sg] = sxg; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sxx = 0; sxg = 0;
        for (int i = 0; i < %[1]d; ++i) { sxx += red[0][i]; sxg += red[1][i]; }
        const float xx = max(sxx, 1e-12f);
        const float proj = sxg / xx;
        float sgg = 0;
        for (int j = 0; j < %[2]d; ++j) { const float gr = bv[j] - proj * av[j]; sgg += gr * gr; }
        sgg = simd_sum(sgg);
        if (lane == 0) red[2][sg] = sgg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sgg = 0;
        for (int i = 0; i < %[1]d; ++i) sgg += red[2][i];
        const float tn = max(sqrt(sgg), 1e-8f);
        const float rr = max(sqrt(xx), 1e-6f);
        float theta = min(tn / rr, %[5]s);
        theta = min((theta * scale + gbias) / geo[2], %[5]s);
        const float c = cos(theta), s = sin(theta) * rr / tn;
        for (int j = 0; j < %[2]d; ++j) {
            const uint i = tid + j * %[3]d;
            if (i < %[4]d) {
                const float v = av[j] * c + (bv[j] - proj * av[j]) * s;
                xs[t * %[4]d + i] = v;
                if (tgid == 0) xout[t * %[4]d + i] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }
`, nsg, EPT, NT, K, q)
}

func olalaGeosPrologue(K, nsg int) string {
	NT := nsg * 32
	q := flt(math.Pi / 4)
	return fmt.Sprintf(`
    {
    const float scale = geo[0], gbias = geo[1], lidx = geo[2];
    for (int t = 0; t < TN; ++t) {
        const float xx = max(gst[4 * t], 1e-12f), xg = gst[4 * t + 1], gg = gst[4 * t + 2];
        const float proj = xg / xx;
        const float tn = max(sqrt(max(gg - 2.0f * proj * xg + proj * proj * xx, 0.0f)), 1e-8f);
        const float rr = max(sqrt(xx), 1e-6f);
        float theta = min(tn / rr, %[3]s);
        theta = min((theta * scale + gbias) / lidx, %[3]s);
        const float c = cos(theta), s = sin(theta) * rr / tn;
        for (uint i = tid; i < %[1]d; i += %[2]d) {
            const float a = xin[t * %[1]d + i];
            const float v = a * c + (g[t * %[1]d + i] - proj * a) * s;
            xs[t * %[1]d + i] = v;
            if (tgid == 0) xout[t * %[1]d + i] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    }
`, K, NT, q)
}

func olalaNormPrologue(K, nsg int) string {
	return fmt.Sprintf(`
    {
    threadgroup float red[%[2]d];
    for (int t = 0; t < TN; ++t) {
        float sxx = 0;
        for (uint i = tid; i < %[1]d; i += %[2]d * 32) { const float a = xin[t * %[1]d + i]; sxx += a * a; }
        sxx = simd_sum(sxx);
        if (lane == 0) red[sg] = sxx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sxx = 0;
        for (int i = 0; i < %[2]d; ++i) sxx += red[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = 1.0f / max(sqrt(sxx), 1e-12f);
        for (uint i = tid; i < %[1]d; i += %[2]d * 32) {
            const float v = xin[t * %[1]d + i] * inv;
            xs[t * %[1]d + i] = v;
            if (tgid == 0) xout[t * %[1]d + i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }
`, K, nsg)
}

func olalaStatsEpilogue(nsg int) string {
	return fmt.Sprintf(`
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (uint)TN * 3) {
        const int t = tid / 3, c = tid %% 3;
        float acc_s = 0;
        for (int i = 0; i < %d; ++i) acc_s += gs_part[i][t][c];
        atomic_fetch_add_explicit((device atomic<float>*)stats_in + 4 * t + c, acc_s, memory_order_relaxed);
    }
`, nsg)
}

type olalaSeg struct {
	format string
	gs     int
	rows   int
}

func olalaGEMVSplits(segs []olalaSeg, K, nsg int) []int {
	out := make([]int, len(segs))
	for i, s := range segs {
		nchunk := K / olalaChunk(s.format)
		sp := 1
		for 2*sp <= nsg && nchunk%(2*sp) == 0 && float64(K)*olalaBytesPerElement(s.format)/float64(sp) > olalaSplitBytes {
			sp *= 2
		}
		out[i] = sp
	}
	return out
}

type olalaGEMV struct {
	kernel *gpuKernel
	ptotal int
}

// Prologues of OlalaGEMV.
const (
	OlalaPlain    = "plain"
	OlalaGeo      = "geo"
	OlalaGeos     = "geos"
	OlalaNorm     = "norm"
	OlalaReLU2Cat = "relu2cat"
)

func olalaGEMVKernel(K, T int, segs []olalaSeg, prologue string, Ka, sa, sb int, xInTG bool, stats bool, nsg int) olalaGEMV {
	const RPS = 1
	key := fmt.Sprintf("gemv|%d|%d|%v|%s|%d|%d|%d|%v|%v|%d", K, T, segs, prologue, Ka, sa, sb, xInTG, stats, nsg)

	var ptotal int
	splits := olalaGEMVSplits(segs, K, nsg)
	poffs := make([]int, len(segs))
	p := 0
	for i, s := range segs {
		sp := splits[i]
		p = (p + sp - 1) / sp * sp // a row's simdgroups never straddle two threadgroups
		poffs[i] = p
		p += s.rows * sp
	}
	ptotal = p

	k := olalaKernel(key, func() *gpuKernel {
		var inputs []string
		switch prologue {
		case OlalaGeo:
			inputs = append(inputs, "xin", "g", "geo")
		case OlalaGeos:
			inputs = append(inputs, "xin", "g", "geo", "gst")
		case OlalaNorm:
			inputs = append(inputs, "xin")
		case OlalaReLU2Cat:
			inputs = append(inputs, "a", "b")
		default:
			inputs = append(inputs, "x")
		}
		for j, s := range segs {
			inputs = append(inputs, fmt.Sprintf("w%d", j))
			if strings.HasPrefix(s.format, "q") {
				inputs = append(inputs, fmt.Sprintf("w%d_s", j), fmt.Sprintf("w%d_b", j))
			}
		}
		inputs = append(inputs, "tcnt")
		if stats {
			inputs = append(inputs, "xres", "stats_in")
		}
		var outputs []string
		for j := range segs {
			outputs = append(outputs, fmt.Sprintf("y%d", j))
		}
		if prologue == OlalaGeo || prologue == OlalaGeos || prologue == OlalaNorm {
			outputs = append(outputs, "xout")
		}

		tt := func(t string) string {
			if T == 1 {
				return t
			}
			return fmt.Sprintf("min(%s, TN - 1)", t)
		}
		var decl string
		var xexpr olalaXExpr
		switch {
		case prologue == OlalaReLU2Cat:
			xexpr = func(t, j string) string {
				return fmt.Sprintf("((k + 4 * (%[2]s) < %[3]d) ? *(const device float4*)(a + %[1]s * %[4]d + k + 4 * (%[2]s)) : "+
					"relu2(*(const device float4*)(b + %[1]s * %[5]d + k + 4 * (%[2]s))))", tt(t), j, Ka, sa, sb)
			}
		case xInTG:
			decl = fmt.Sprintf("threadgroup float xs[%d * %d];", T, K)
			xexpr = func(t, j string) string {
				return fmt.Sprintf("*(const threadgroup float4*)(xs + %s * %d + k + 4 * (%s))", t, K, j)
			}
		default:
			xexpr = func(t, j string) string {
				return fmt.Sprintf("*(const device float4*)(x + %s * %d + k + 4 * (%s))", tt(t), K, j)
			}
		}

		var pro string
		switch prologue {
		case OlalaGeo:
			pro = olalaGeoPrologue(K, nsg)
		case OlalaNorm:
			pro = olalaNormPrologue(K, nsg)
		case OlalaGeos:
			pro = olalaGeosPrologue(K, nsg)
		case OlalaReLU2Cat:
		default:
			if xInTG {
				pro = fmt.Sprintf("for (uint i = tid; i < TN * %d; i += %d * 32) xs[i] = x[i];\nthreadgroup_barrier(mem_flags::mem_threadgroup);", K, nsg)
			}
		}

		total := 0
		for _, s := range segs {
			total += s.rows
		}
		anysplit := false
		for _, sp := range splits {
			if sp > 1 {
				anysplit = true
			}
		}

		var body []string
		for j, s := range segs {
			sp, po := splits[j], poffs[j]
			ch := olalaChunk(s.format)
			KP := K / sp
			body = append(body, fmt.Sprintf(`
        if (row >= %[1]d && row < %[2]d) {
            const int lr = (row - %[1]d) / %[3]d;
            const int kb = ((row - %[1]d) %% %[3]d) * %[4]d;
            for (int k = kb + lane * %[5]d; k < kb + %[4]d; k += 32 * %[5]d) {
                %[6]s
            }
        }`, po, po+s.rows*sp, sp, KP, ch, olalaDotCode(s.format, fmt.Sprintf("w%d", j), K, s.gs, xexpr, T, "lr", "k")))
		}

		store := func(v string) string {
			var lines []string
			off := 0
			for j, s := range segs {
				lines = append(lines, fmt.Sprintf("if (orow >= %d && orow < %d) y%d[t * %d + orow - %d] = %s;", off, off+s.rows, j, s.rows, off, v))
				off += s.rows
			}
			if stats {
				lines = append(lines, fmt.Sprintf("const float xr = xres[t * %d + orow]; "+
					"gs_part[sg][t][0] = xr * xr; gs_part[sg][t][1] = xr * %[2]s; gs_part[sg][t][2] = %[2]s * %[2]s;", total, v))
			}
			return strings.Join(lines, "\n")
		}

		var rowmap []string
		before := 0
		for j, s := range segs {
			sp, po := splits[j], poffs[j]
			rowmap = append(rowmap, fmt.Sprintf("if (row >= %[1]d && row < %[2]d) { orow = %[3]d + (row - %[1]d) / %[4]d; np = %[4]d; first = (row - %[1]d) %% %[4]d == 0; }",
				po, po+s.rows*sp, before, sp))
			before += s.rows
		}
		zeroStats := ""
		if stats {
			zeroStats = fmt.Sprintf("if (lane == 0) for (int t = 0; t < %d; ++t) { gs_part[sg][t][0] = 0; gs_part[sg][t][1] = 0; gs_part[sg][t][2] = 0; }", T)
		}
		var epi string
		if anysplit {
			epi = fmt.Sprintf(`
    threadgroup float pacc[%d][%d];
    {
        const int row = row0;
        for (int t = 0; t < TN; ++t) { const float v = simd_sum(acc[0][t]); if (lane == 0) pacc[sg][t] = v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int orow = -1, np = 1; bool first = false;
        %s
        %s
        if (first && lane == 0) {
            for (int t = 0; t < TN; ++t) {
                float v = 0;
                for (int i = 0; i < np; ++i) v += pacc[sg + i][t];
                %s
            }
        }
    }`, nsg, T, strings.Join(rowmap, "\n"), zeroStats, store("v"))
		} else {
			epi = fmt.Sprintf(`
    for (int r = 0; r < %d; ++r) {
        const int row = row0 + r;
        if (row >= %d) {
            %s
            break;
        }
        const int orow = row;
        for (int t = 0; t < TN; ++t) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0) { %s }
        }
    }`, RPS, total, zeroStats, store("v"))
		}

		gsDecl, statsEpi := "", ""
		if stats {
			gsDecl = fmt.Sprintf("threadgroup float gs_part[%d][%d][3];", nsg, T)
			statsEpi = olalaStatsEpilogue(nsg)
		}
		src := fmt.Sprintf(`
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32, lane = tid %% 32;
    const uint tgid = threadgroup_position_in_grid.x;
    const int TN = tcnt[0];
    %s
    %s
    const int row0 = (tgid * %d + sg) * %d;
    float acc[%d][%d];
    for (int r = 0; r < %d; ++r) for (int t = 0; t < %d; ++t) acc[r][t] = 0;
    for (int r = 0; r < %d; ++r) {
        const int row = row0 + r;
        %s
    }
    %s
    %s
    %s
`, decl, pro, nsg, RPS, RPS, T, RPS, T, RPS, strings.Join(body, "\n"), gsDecl, epi, statsEpi)

		return &gpuKernel{
			name:    fmt.Sprintf("olala_gemv_%d", len(olalaKernels)),
			inputs:  inputs,
			outputs: outputs,
			metal:   gpuSource{source: src, header: olalaHeader},
		}
	})
	return olalaGEMV{kernel: k, ptotal: ptotal}
}

func olalaRowBucket(T int) int {
	switch {
	case T == 1:
		return 1
	case T == 2:
		return 2
	case T <= 4:
		return 4
	default:
		return 8
	}
}

// OlalaGEMVMaxRows is the largest row count OlalaGEMV handles.
const OlalaGEMVMaxRows = 5

func olalaRowCount(T int) *Array {
	return NewArrayInt32([]int32{int32(T), 0, 0, 0}, []int32{4})
}

// OlalaGEMVOptions configures OlalaGEMV.
type OlalaGEMVOptions struct {
	Prologue string
	// Ka, Sa, Sb configure the relu2cat prologue: x = [a[:, :Ka] | relu(b[:, Ka:])^2],
	// with row strides Sa and Sb.
	Ka, Sa, Sb int
	// StatsX, when set, makes the GEMV also return (T, 4) sums (x.x, x.g, g.g)
	// of StatsX and its output, for the geos prologue of the next GEMV.
	StatsX *Array
}

// OlalaGEMV computes x (T, K) f32 @ W.T for each weight, after the prologue
// built from xs (see the Olala* prologue constants). It returns one (T, rows)
// f32 output per weight, then xout (T, K) for the geo / geos / norm
// prologues, then the stats when StatsX is set. T must not exceed
// OlalaGEMVMaxRows.
func OlalaGEMV(xs []*Array, ws []*OlalaWeight, T, K int, opt OlalaGEMVOptions) []*Array {
	prologue := opt.Prologue
	if prologue == "" {
		prologue = OlalaPlain
	}
	segs := make([]olalaSeg, len(ws))
	for i, w := range ws {
		segs[i] = olalaSeg{format: w.format, gs: w.GroupSize, rows: w.Rows}
	}
	TC := olalaRowBucket(T)
	xInTG := prologue == OlalaGeo || prologue == OlalaGeos || prologue == OlalaNorm
	if xInTG && TC*K*4 > olalaTGXBytes {
		TC = T
	}
	stats := opt.StatsX != nil
	nsg := olalaNSGDev
	if xInTG {
		nsg = olalaNSG
	}
	g := olalaGEMVKernel(K, TC, segs, prologue, opt.Ka, opt.Sa, opt.Sb, xInTG, stats, nsg)
	ntg := (g.ptotal + nsg - 1) / nsg

	inputs := append([]*Array(nil), xs...)
	for _, w := range ws {
		inputs = append(inputs, w.arrays()...)
	}
	inputs = append(inputs, olalaRowCount(T))
	var gst *Array
	if stats {
		// padded: MLX binds very small inputs as constant buffers, which cannot be written
		gst = OlalaZeros(T*4 + 64)
		inputs = append(inputs, opt.StatsX, gst)
	}
	var outs []gpuOutputSpec
	for i, w := range ws {
		outs = append(outs, gpuOutputSpec{name: fmt.Sprintf("y%d", i), shape: []int32{int32(T), int32(w.Rows)}, dtype: DTypeFloat32})
	}
	if xInTG {
		outs = append(outs, gpuOutputSpec{name: "xout", shape: []int32{int32(T), int32(K)}, dtype: DTypeFloat32})
	}
	res := g.kernel.run(gpuLaunch{
		outputs:     outs,
		grid:        [3]int{ntg * nsg * 32, 1, 1},
		threadGroup: [3]int{nsg * 32, 1, 1},
		inputs:      inputs,
	})
	if stats {
		res = append(res, gst)
	}
	return res
}

// OlalaZeros returns a materialized zero f32 buffer of n elements (kernels
// accumulate into it atomically, so it must not be a broadcast constant).
func OlalaZeros(n int) *Array {
	k := olalaKernel("zeros", func() *gpuKernel {
		return &gpuKernel{
			name:    "olala_zeros",
			inputs:  []string{"n_"},
			outputs: []string{"z"},
			metal:   gpuSource{source: "const uint i = thread_position_in_grid.x; if (i < (uint)n_[0]) z[i] = 0.0f;"},
		}
	})
	return k.run(gpuLaunch{
		outputs:     []gpuOutputSpec{{name: "z", shape: []int32{int32(n)}, dtype: DTypeFloat32}},
		grid:        [3]int{(n + 255) / 256 * 256, 1, 1},
		threadGroup: [3]int{256, 1, 1},
		inputs:      []*Array{NewArrayInt32([]int32{int32(n), 0, 0, 0}, []int32{4})},
	})[0]
}

// OlalaGeodesic is the standalone geodesic update of xin (T, K) towards g,
// with geo = [scale, bias, layer + 1].
func OlalaGeodesic(xin, g, geo *Array, T, K int) *Array {
	k := olalaKernel(fmt.Sprintf("geo1|%d", K), func() *gpuKernel {
		body := olalaGeoPrologue(K, olalaNSG)
		ks := strconv.Itoa(K)
		body = strings.ReplaceAll(body, "for (int t = 0; t < TN; ++t)", "for (int t = 0; t < 1; ++t)")
		body = strings.ReplaceAll(body, "xin[t * "+ks, "xin[row * "+ks)
		body = strings.ReplaceAll(body, "g[t * "+ks, "g[row * "+ks)
		body = strings.ReplaceAll(body, "xs[t * "+ks+" + i] = v;", "")
		body = strings.ReplaceAll(body, "xout[t * "+ks, "xout[row * "+ks)
		body = strings.ReplaceAll(body, "if (tgid == 0) xout", "xout")
		src := `
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint row = threadgroup_position_in_grid.x;
        const uint tgid = 0;
` + body
		return &gpuKernel{
			name:    "olala_geo1_" + ks,
			inputs:  []string{"xin", "g", "geo"},
			outputs: []string{"xout"},
			metal:   gpuSource{source: src, header: olalaHeader},
		}
	})
	return k.run(gpuLaunch{
		outputs:     []gpuOutputSpec{{name: "xout", shape: []int32{int32(T), int32(K)}, dtype: DTypeFloat32}},
		grid:        [3]int{T * olalaNSG * 32, 1, 1},
		threadGroup: [3]int{olalaNSG * 32, 1, 1},
		inputs:      []*Array{xin, g, geo},
	})[0]
}

// ---------------------------------------------------------------------------
// Mamba-3 MIMO decode step

// OlalaM3Dims are the Mamba-3 dimensions of an M layer.
type OlalaM3Dims struct{ H, P, N, R, A, NStatic int }

// StateWidth is the packed per-head state width: [S (P*N) | k (R*N) | x (P) | angles (A)].
func (d OlalaM3Dims) StateWidth() int { return d.P*d.N + d.R*d.N + d.P + d.A }

// OlalaM3Step runs one Mamba-3 MIMO token per row. proj: (T, NStatic + 2RN + A)
// f32 in_proj output; prm: the packed f32 parameters [dt_bias (H) | D (H) |
// b_norm (N) | c_norm (N) | b_bias | c_bias (H*R*N) | mimo_x | mimo_z |
// mimo_o (H*R*P)]; st: packed state (T, H, StateWidth) f32. Returns y (T, H*P)
// and the new state.
func OlalaM3Step(proj, prm, st *Array, T int, d OlalaM3Dims) (y, stOut *Array) {
	SW := d.StateWidth()
	PB := olalaM3PB
	nsg := olalaNSG
	if d.N != 128 || d.A != 32 || 2*d.R > nsg || d.P%PB != 0 {
		panic("olala: unsupported Mamba-3 dimensions")
	}
	k := olalaKernel(fmt.Sprintf("m3|%v", d), func() *gpuKernel {
		src := fmt.Sprintf(`
        constexpr int H = %d, P = %d, N = %d, R = %d, A = %d, PB = %d, NSGM = %d;
        constexpr int PH = 2 * P + 3, NS = %d, PW = %d + 2 * R * N + A;
        constexpr int SW = %d, OK_ = P * N, OX = OK_ + R * N, OA = OX + P;
        constexpr int O_DT = 0, O_D = H, O_BN = 2 * H, O_CN = O_BN + N, O_BB = O_CN + N, O_CB = O_BB + H * R * N;
        constexpr int O_MX = O_CB + H * R * N, O_MZ = O_MX + H * R * P, O_MO = O_MZ + H * R * P;
        constexpr int RPSG = PB / NSGM;
`, d.H, d.P, d.N, d.R, d.A, PB, nsg, d.NStatic, d.NStatic, SW) + `
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int npb = P / PB;
        const int b = blk / (H * npb);
        const int h = (blk / npb) % H;
        const int pb = blk % npb;
        const device float* pr = proj + (size_t)b * PW;
        const device float* ph = pr + h * PH;
        const device float* dyn = pr + NS;
        const device float* sin_ = st + ((size_t)b * H + h) * SW;
        device float* sout = st_out + ((size_t)b * H + h) * SW;

        threadgroup float4 kqs[2 * R][N / 4];
        threadgroup float vws[3][R][PB];

        // early loads for the sweep (independent of the prep)
        float4 srow[RPSG];
        for (int i = 0; i < RPSG; ++i) srow[i] = ((const device float4*)(sin_ + (pb * PB + sg + NSGM * i) * N))[lane];
        float4 kp[R];
        for (int r = 0; r < R; ++r) kp[r] = ((const device float4*)(sin_ + OK_ + r * N))[lane];

        const float dt = olala_softplus(ph[2 * P] + prm[O_DT + h]);
        const float a = min(-olala_softplus(ph[2 * P + 1]), -1e-4f);
        const float alpha = exp(a * dt);
        const float trap = olala_sigmoid(ph[2 * P + 2]);
        const float beta = (1.0f - trap) * dt * alpha;
        const float gamma = trap * dt;

        if (sg < 2 * R) {
            const int gi = sg;
            const bool isq = gi >= R;
            const int r = isq ? gi - R : gi;
            const float4 u = ((const device float4*)(dyn + gi * N))[lane];
            const float4 wn = ((const device float4*)(prm + (isq ? O_CN : O_BN)))[lane];
            const float4 bb = ((const device float4*)(prm + (isq ? O_CB : O_BB) + (h * R + r) * N))[lane];
            // rotary angle of this lane (A == 32): wrapped cumulative angle
            float thl = sin_[OA + lane] + tanh(dyn[2 * R * N + lane]) * ` + flt(math.Pi) + ` * dt;
            thl = thl - ` + flt(2*math.Pi) + ` * rint(thl * ` + flt(0.5/math.Pi) + `);
            if (gi == 0 && pb == 0) sout[OA + lane] = thl;
            const float ss = simd_sum(dot(u, u));
            float4 val = u * rsqrt(ss / N + 1e-06f) * wn + bb;
            // rotate pairs (n, n + 64) for n < 32: lanes 0..7 hold n = 0..31, lanes 16..23 hold n = 64..95
            const float4 partner = simd_shuffle_xor(val, 16);
            const int base = 4 * (lane & 15);
            const float4 th4 = float4(simd_shuffle(thl, (base + 0) & 31), simd_shuffle(thl, (base + 1) & 31),
                                      simd_shuffle(thl, (base + 2) & 31), simd_shuffle(thl, (base + 3) & 31));
            if ((lane & 15) < 8) {
                const float4 c = cos(th4), s = sin(th4);
                val = (lane < 16) ? (val * c - partner * s) : (partner * s + val * c);
            }
            kqs[gi][lane] = val;
            if (!isq && pb == 0) ((device float4*)(sout + OK_ + r * N))[lane] = val;
        } else {
            for (int i = tid - 2 * R * 32; i < R * PB; i += (NSGM - 2 * R) * 32) {
                const int r = i / PB, pl = i % PB, p = pb * PB + pl;
                const float mxv = prm[O_MX + (h * R + r) * P + p];
                const float xv = ph[P + p];
                const float zz = ph[p] * prm[O_MZ + (h * R + r) * P + p];
                vws[0][r][pl] = xv * mxv;
                vws[1][r][pl] = zz / (1.0f + exp(-zz)) * prm[O_MO + (h * R + r) * P + p];
                vws[2][r][pl] = sin_[OX + p] * mxv;
                if (r == 0) sout[OX + p] = xv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 kr[R], qr[R];
        for (int r = 0; r < R; ++r) { kr[r] = kqs[r][lane]; qr[r] = kqs[R + r][lane]; }
        const float Dh = prm[O_D + h];
        for (int i = 0; i < RPSG; ++i) {
            const int pl = sg + NSGM * i, p = pb * PB + pl;
            float4 kv = 0, kvp = 0, qw = 0;
            float d = 0;
            for (int r = 0; r < R; ++r) {
                const float vv = vws[0][r][pl], ww = vws[1][r][pl], vp = vws[2][r][pl];
                kv += vv * kr[r]; kvp += vp * kp[r]; qw += ww * qr[r];
                d += ww * vv;
            }
            const float4 s = alpha * srow[i] + beta * kvp + gamma * kv;
            ((device float4*)(sout + p * N))[lane] = s;
            const float acc = simd_sum(dot(s, qw));
            if (lane == 0) { y[(size_t)b * H * P + h * P + p] = acc + Dh * d; }
        }
`
		return &gpuKernel{
			name:    fmt.Sprintf("olala_m3_%d_%d", d.H, d.P),
			inputs:  []string{"proj", "prm", "st"},
			outputs: []string{"y", "st_out"},
			metal:   gpuSource{source: src, header: olalaHeader},
		}
	})
	npb := d.P / PB
	outs := k.run(gpuLaunch{
		outputs: []gpuOutputSpec{
			{name: "y", shape: []int32{int32(T), int32(d.H * d.P)}, dtype: DTypeFloat32},
			{name: "st_out", shape: []int32{int32(T), int32(d.H), int32(SW)}, dtype: DTypeFloat32},
		},
		grid:        [3]int{T * d.H * npb * nsg * 32, 1, 1},
		threadGroup: [3]int{nsg * 32, 1, 1},
		inputs:      []*Array{proj, prm, st},
	})
	return outs[0], outs[1]
}

// ---------------------------------------------------------------------------
// MoE

func olalaTopKCode(scaling float32) string {
	return `
            float sc[E / 32], sv[E / 32];
            for (int j = 0; j < E / 32; ++j) {
                const int e = lane + 32 * j;
                sv[j] = olala_sigmoid(logits[t * E + e]);
                sc[j] = sv[j] + ebias[e];
            }
            float psum = 0;
            for (int q = 0; q < TOPK; ++q) {
                float best = -INFINITY; int bj = 0;
                for (int j = 0; j < E / 32; ++j) if (sc[j] > best) { best = sc[j]; bj = j; }
                const float m = simd_max(best);
                const int cand = (best == m) ? (lane + 32 * bj) : 0x7fffffff;
                const int e = simd_min(cand);
                float pv = 0;
                if (e == lane + 32 * bj) { pv = sv[bj]; sc[bj] = -INFINITY; }
                pv = simd_sum(pv);
                if (lane == 0) { sel_idx[q] = e; sel_p[q] = pv; }
                psum += pv;
            }
            if (lane == 0) for (int q = 0; q < TOPK; ++q) sel_p[q] = sel_p[q] / (psum + 1e-20f) * ` + flt(float64(scaling)) + `;`
}

// OlalaMoE runs the routed experts of one MoE layer. logits: (T, E) f32 router
// logits; ebias: (E) f32 selection bias; x0: (T, x0Stride) f32 whose first D
// columns are the expert input; up: stacked (E, F, D); down: stacked (E, D, F).
// Returns the summed expert output (T, D) f32.
func OlalaMoE(logits, ebias, x0 *Array, x0Stride int, up, down *OlalaWeight, topK int, scaling float32, T int) *Array {
	E := up.W.Dim(0)
	F, D := up.Rows, up.Cols
	FB := olalaMoEFB
	NB := F / FB
	chu, chd := olalaChunk(up.format), olalaChunk(down.format)
	NT := olalaNSG * 32
	LPRU := max(1, min(32, NT/FB))
	for LPRU > 1 && (D/chu)%LPRU != 0 {
		LPRU /= 2
	}
	key := fmt.Sprintf("moe|%d|%d|%d|%s|%s|%d|%d|%v", E, F, D, up.key(), down.key(), topK, x0Stride, scaling)
	k := olalaKernel(key, func() *gpuKernel {
		xu := func(t, j string) string { return fmt.Sprintf("*(const threadgroup float4*)(xs + k + 4 * (%s))", j) }
		xd := func(t, j string) string {
			return fmt.Sprintf("*(const threadgroup float4*)(hs + (k - f0) + 4 * (%s))", j)
		}
		dotu := strings.ReplaceAll(olalaDotCode(up.format, "wu", D, up.GroupSize, xu, 1, "erow", "k"), "acc[r][t]", "acc")
		dotd := strings.ReplaceAll(olalaDotCode(down.format, "wd", F, down.GroupSize, xd, 1, "erow", "k"), "acc[r][t]", "acc")
		src := fmt.Sprintf(`
        constexpr int E = %d, F = %d, D = %d, TOPK = %d, FB = %d, NB = %d, NT = %d, LPRU = %d;
`, E, F, D, topK, FB, NB, NT, LPRU) + `
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int t = blk / (TOPK * NB);
        const int kk = (blk / NB) % TOPK;
        const int fb = blk % NB;
        const int f0 = fb * FB;
        threadgroup float xs[D];
        threadgroup float hs[FB];
        threadgroup int sel_idx[TOPK];
        threadgroup float sel_p[TOPK];
        if (sg == 0) {
            ` + olalaTopKCode(scaling) + `
        } else {
            for (int i = tid - 32; i < D; i += NT - 32) xs[i] = x0[t * ` + strconv.Itoa(x0Stride) + ` + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int e = sel_idx[kk];
        const float pk = sel_p[kk];
        // phase 1: hidden unit f = tid / LPRU, LPRU lanes split D
        {
            const int f = tid / LPRU, sl = tid % LPRU;
            float acc = 0;
            if (f < FB) {
                const int erow = e * F + f0 + f;
                for (int k = sl * ` + strconv.Itoa(chu) + `; k < D; k += LPRU * ` + strconv.Itoa(chu) + `) {
                    ` + dotu + `
                }
            }
            for (int o = LPRU / 2; o > 0; o /= 2) acc += simd_shuffle_xor(acc, o);
            if (sl == 0 && f < FB) { const float u = max(acc, 0.0f); hs[f] = u * u * pk; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // phase 2: thread o computes output o over this block's FB hidden units
        for (int o = tid; o < D; o += NT) {
            const int erow = e * D + o;
            float acc = 0;
            for (int k = f0; k < f0 + FB; k += ` + strconv.Itoa(chd) + `) {
                ` + dotd + `
            }
            atomic_fetch_add_explicit(&parts[(size_t)t * D + o], acc, memory_order_relaxed);
        }
`
		inputs := []string{"logits", "ebias", "x0", "wu"}
		if up.quantized() {
			inputs = append(inputs, "wu_s", "wu_b")
		}
		inputs = append(inputs, "wd")
		if down.quantized() {
			inputs = append(inputs, "wd_s", "wd_b")
		}
		return &gpuKernel{
			name:          fmt.Sprintf("olala_moe_%d", len(olalaKernels)),
			inputs:        inputs,
			outputs:       []string{"parts"},
			metal:         gpuSource{source: src, header: olalaHeader},
			atomicOutputs: true,
		}
	})
	inputs := []*Array{logits, ebias, x0}
	inputs = append(inputs, up.arrays()...)
	inputs = append(inputs, down.arrays()...)
	zero := float32(0)
	return k.run(gpuLaunch{
		outputs:     []gpuOutputSpec{{name: "parts", shape: []int32{int32(T), int32(D)}, dtype: DTypeFloat32}},
		grid:        [3]int{T * topK * NB * olalaNSG * 32, 1, 1},
		threadGroup: [3]int{olalaNSG * 32, 1, 1},
		inputs:      inputs,
		initValue:   &zero,
	})[0]
}

// ---------------------------------------------------------------------------
// attention

// OlalaAttnDims are the attention dimensions of a V layer.
type OlalaAttnDims struct {
	Hq, Hk, Dh, Rk, NNoise int
	Window                 int
	Scale, Softcap         float32
	GateBias               float32
}

func olalaMetalType(dt DType) string {
	switch dt {
	case DTypeBFloat16:
		return "bfloat"
	case DTypeFloat16:
		return "half"
	default:
		return "float"
	}
}

// OlalaVPrep builds q (T, Hq, Dh) f32 (normed and scalable-softmax scaled),
// the new k / v (T, Hk, 1, Dh) in kvType, and the raw k / v (T, 1, Hk, Dh) f32
// for the next token shift. big / small are the in_proj / in_proj_small
// outputs, kprev / vprev the previous raw k / v, pos (T) int32 positions.
func OlalaVPrep(big, small, kprev, vprev, qn, kn, scaler, pos *Array, T int, d OlalaAttnDims, kvType DType) (q, k, v, kraw, vraw *Array) {
	nbig := d.Hq*d.Dh + 2*d.Rk*d.Dh + (d.Hq-d.Hk)*d.Dh
	nsmall := 2*d.Hk*d.Rk + 2*d.Hk + d.NNoise
	key := fmt.Sprintf("vprep|%v|%v", d, kvType)
	kern := olalaKernel(key, func() *gpuKernel {
		ot := olalaMetalType(kvType)
		src := fmt.Sprintf(`
        constexpr int Hq = %d, Hk = %d, Dh = %d, Rk = %d;
        constexpr int NB = %d, NSM = %d;
`, d.Hq, d.Hk, d.Dh, d.Rk, nbig, nsmall) + `
        const uint tid = thread_position_in_threadgroup.x;   // Dh threads
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int t = blk / (Hq + Hk);
        const int hh = blk % (Hq + Hk);
        const device float* bg = big + (size_t)t * NB;
        const device float* sm = small + (size_t)t * NSM;
        const float p = float(pos[t]);
        threadgroup float red[Dh / 32];
        float val;
        if (hh < Hq) {
            val = bg[hh * Dh + tid];
        } else {
            const int h = hh - Hq;
            const device float* bk = bg + Hq * Dh;            // W_B_k (Rk, Dh)
            const device float* bv = bk + Rk * Dh;            // W_B_v
            float kr = 0, vr = 0;
            for (int r = 0; r < Rk; ++r) {
                kr += sm[h * Rk + r] * bk[r * Dh + tid];
                vr += sm[Hk * Rk + h * Rk + r] * bv[r * Dh + tid];
            }
            kr /= Rk; vr /= Rk;
            float ak = olala_sigmoid(sm[2 * Hk * Rk + h]);
            float av = olala_sigmoid(sm[2 * Hk * Rk + Hk + h]);
            if (p <= 0.0f) { ak = 0; av = 0; }
            const float kp = kprev[((size_t)t * Hk + h) * Dh + tid];
            const float vp = vprev[((size_t)t * Hk + h) * Dh + tid];
            kraw_out[((size_t)t * Hk + h) * Dh + tid] = kr;
            vraw_out[((size_t)t * Hk + h) * Dh + tid] = vr;
            val = ak * kp + (1 - ak) * kr;
            v_out[((size_t)t * Hk + h) * Dh + tid] = (` + ot + `)(av * vp + (1 - av) * vr);
        }
        float s = simd_sum(val * val);
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = 0;
        for (int i = 0; i < Dh / 32; ++i) s += red[i];
        const float rn = rsqrt(s / Dh + 1e-06f);
        if (hh < Hq) {
            const float lp = log(min(p + 1.0f, ` + flt(float64(d.Window)) + `));
            q_out[((size_t)t * Hq + hh) * Dh + tid] = val * rn * qn[tid] * scaler[hh] * lp;
        } else {
            const int h = hh - Hq;
            k_out[((size_t)t * Hk + h) * Dh + tid] = (` + ot + `)(val * rn * kn[tid]);
        }
`
		return &gpuKernel{
			name:    fmt.Sprintf("olala_v_prep_%d", len(olalaKernels)),
			inputs:  []string{"big", "small", "kprev", "vprev", "qn", "kn", "scaler", "pos"},
			outputs: []string{"q_out", "k_out", "v_out", "kraw_out", "vraw_out"},
			metal:   gpuSource{source: src, header: olalaHeader},
		}
	})
	i32 := func(v ...int) []int32 {
		out := make([]int32, len(v))
		for i, x := range v {
			out[i] = int32(x)
		}
		return out
	}
	outs := kern.run(gpuLaunch{
		outputs: []gpuOutputSpec{
			{name: "q_out", shape: i32(T, d.Hq, d.Dh), dtype: DTypeFloat32},
			{name: "k_out", shape: i32(T, d.Hk, 1, d.Dh), dtype: kvType},
			{name: "v_out", shape: i32(T, d.Hk, 1, d.Dh), dtype: kvType},
			{name: "kraw_out", shape: i32(T, 1, d.Hk, d.Dh), dtype: DTypeFloat32},
			{name: "vraw_out", shape: i32(T, 1, d.Hk, d.Dh), dtype: DTypeFloat32},
		},
		grid:        [3]int{T * (d.Hq + d.Hk) * d.Dh, 1, 1},
		threadGroup: [3]int{d.Dh, 1, 1},
		inputs:      []*Array{big, small, kprev, vprev, qn, kn, scaler, pos},
	})
	return outs[0], outs[1], outs[2], outs[3], outs[4]
}

func olalaReduceScatter32(v string) string {
	var out []string
	n := 32
	for _, o := range []int{16, 8, 4, 2, 1} {
		h := n / 2
		out = append(out, fmt.Sprintf("{ const bool hi = (lane & %[2]d) != 0; float w[%[3]d];"+
			"for (int j = 0; j < %[3]d; ++j) { const float send = hi ? %[1]s[j] : %[1]s[j + %[3]d];"+
			"const float keep = hi ? %[1]s[j + %[3]d] : %[1]s[j]; w[j] = keep + simd_shuffle_xor(send, %[2]d); }"+
			"for (int j = 0; j < %[3]d; ++j) %[1]s[j] = w[j]; }", v, o, h))
		n = h
	}
	return strings.Join(out, "\n")
}

func olalaVecType(dt DType) string {
	switch dt {
	case DTypeBFloat16:
		return "bfloat4"
	case DTypeFloat16:
		return "half4"
	default:
		return "float4"
	}
}

// OlalaAttnDecode computes decode attention partials with soft-capped
// logits. q: (T, Hq, Dh) f32; keys / values: (T, Hk, >=S, Dh), possibly strided
// views; valid keys of row t are [kstart[t], S). Returns o (T, Hq, NS, Dh),
// m and l (T, Hq, NS) for OlalaAttnCombine.
func OlalaAttnDecode(q, keys, values, kstart *Array, T, S int, d OlalaAttnDims) (o, m, l *Array) {
	G := d.Hq / d.Hk
	if d.Dh != 128 {
		panic("olala: attention head dim must be 128")
	}
	vt := olalaVecType(keys.DType())
	short := S <= olalaAttShort && G == 4
	var BLK int
	var kern *gpuKernel
	header := fmt.Sprintf("constexpr int Hq = %d, Hk = %d, Dh = %d, ", d.Hq, d.Hk, d.Dh)
	scale, softcap := flt(float64(d.Scale)), flt(float64(d.Softcap))
	if short {
		BLK = 8 * olalaAttNW
		kern = olalaKernel(fmt.Sprintf("attn_short|%v|%s", d, vt), func() *gpuKernel {
			src := header + fmt.Sprintf("G = 4, BLK = %d, NW = %d;", BLK, olalaAttNW) + `
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int S = slen[0];
        const int NS = (S + BLK - 1) / BLK;
        const int t = blk / (Hk * NS);
        const int hk = (blk / NS) % Hk;
        const int sp = blk % NS;
        const int j0 = max(sp * BLK, kstart[t]);
        const int j1 = min((sp + 1) * BLK, S);
        float4 qv[G];  // this lane's 4 dims of the G heads
        for (int g = 0; g < G; ++g)
            qv[g] = ((const device float4*)(q + ((size_t)t * Hq + hk * G + g) * Dh))[lane] * ` + scale + `;
        const device ` + vt + `* kb = (const device ` + vt + `*)(keys + (size_t)t * keys_strides[0] + (size_t)hk * keys_strides[1]);
        const device ` + vt + `* vb = (const device ` + vt + `*)(values + (size_t)t * values_strides[0] + (size_t)hk * values_strides[1]);
        const size_t ks4 = keys_strides[2] / 4, vs4 = values_strides[2] / 4;
        float m = -INFINITY, l = 0;  // running max / sum of head lane & 3
        float4 o[G];
        for (int g = 0; g < G; ++g) o[g] = 0;
        threadgroup float mo[NW][G];
        threadgroup float lo[NW][G];
        threadgroup float4 oo[NW][G][Dh / 4];
        for (int jb = (j0 / 8) * 8 + sg * 8; jb < j1; jb += NW * 8) {
            float4 kv[8], vv[8];
            for (int i = 0; i < 8; ++i) {
                const int j = min(jb + i, S - 1);
                kv[i] = float4(kb[(size_t)j * ks4 + lane]);
                vv[i] = float4(vb[(size_t)j * vs4 + lane]);
            }
            float v[32];
            for (int i = 0; i < 8; ++i) for (int g = 0; g < G; ++g) v[i * 4 + g] = dot(kv[i], qv[g]);
            ` + olalaReduceScatter32("v") + `
            const int jl = jb + (lane >> 2);
            const float s = (jl >= j0 && jl < j1) ? tanh(v[0] / ` + softcap + `) * ` + softcap + ` : -INFINITY;
            float bm = max(s, simd_shuffle_xor(s, 4));
            bm = max(bm, simd_shuffle_xor(bm, 8));
            bm = max(bm, simd_shuffle_xor(bm, 16));
            const float mn = max(m, bm);
            const float corr = mn == -INFINITY ? 1.0f : exp(m - mn);
            const float p = s == -INFINITY ? 0.0f : exp(s - mn);
            float ps = p + simd_shuffle_xor(p, 4);
            ps += simd_shuffle_xor(ps, 8);
            ps += simd_shuffle_xor(ps, 16);
            l = l * corr + ps;
            m = mn;
            for (int g = 0; g < G; ++g) {
                o[g] *= simd_shuffle(corr, g);
                for (int i = 0; i < 8; ++i) o[g] += simd_shuffle(p, i * 4 + g) * vv[i];
            }
        }
        for (int g = 0; g < G; ++g) {
            const float mg = simd_shuffle(m, g), lg = simd_shuffle(l, g);
            if (lane == 0) { mo[sg][g] = mg; lo[sg][g] = lg; }
            oo[sg][g][lane] = o[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < G) {
            const int g = sg;
            float M = -INFINITY;
            for (int w = 0; w < NW; ++w) M = max(M, mo[w][g]);
            float L = 0; float4 O = 0;
            for (int w = 0; w < NW; ++w) {
                const float c = (mo[w][g] == -INFINITY) ? 0.0f : exp(mo[w][g] - M);
                L += lo[w][g] * c; O += oo[w][g][lane] * c;
            }
            const size_t oi = ((size_t)t * Hq + hk * G + g) * NS + sp;
            ((device float4*)(o_out + oi * Dh))[lane] = O;
            if (lane == 0) { m_out[oi] = M; l_out[oi] = L; }
        }
`
			return &gpuKernel{
				name:    fmt.Sprintf("olala_attn_short_%d", len(olalaKernels)),
				inputs:  []string{"q", "keys", "values", "kstart", "slen"},
				outputs: []string{"o_out", "m_out", "l_out"},
				metal:   gpuSource{source: src, header: olalaHeader},
				strided: true,
			}
		})
	} else {
		BLK = olalaAttBlk
		kern = olalaKernel(fmt.Sprintf("attn|%v|%s", d, vt), func() *gpuKernel {
			src := header + fmt.Sprintf("G = %d, BLK = %d, NW = %d;", G, BLK, olalaAttNW) + `
        const uint tid = thread_position_in_threadgroup.x;
        const uint sg = tid / 32, lane = tid % 32;
        const uint blk = threadgroup_position_in_grid.x;
        const int S = slen[0];
        const int NS = (S + BLK - 1) / BLK;
        const int t = blk / (Hk * NS);
        const int hk = (blk / NS) % Hk;
        const int sp = blk % NS;
        const int j0 = max(sp * BLK, kstart[t]);
        const int j1 = min((sp + 1) * BLK, S);
        threadgroup float4 qs[G][Dh / 4];
        threadgroup float mo[NW][G];
        threadgroup float lo[NW][G];
        threadgroup float4 oo[NW][G][Dh / 4];
        for (int i = tid; i < G * Dh / 4; i += NW * 32)
            qs[i / (Dh / 4)][i % (Dh / 4)] = ((const device float4*)(q + ((size_t)t * Hq + hk * G) * Dh))[i] * ` + scale + `;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device ` + vt + `* kb = (const device ` + vt + `*)(keys + (size_t)t * keys_strides[0] + (size_t)hk * keys_strides[1]);
        const device ` + vt + `* vb = (const device ` + vt + `*)(values + (size_t)t * values_strides[0] + (size_t)hk * values_strides[1]);
        const size_t ks4 = keys_strides[2] / 4, vs4 = values_strides[2] / 4;
        float m[G], l[G];
        float4 o[G];
        for (int g = 0; g < G; ++g) { m[g] = -INFINITY; l[g] = 0; o[g] = 0; }
        // simdgroup sg takes chunks of 32 keys: scores with lane = key, values with lane = 4 dims
        for (int jb = j0 + sg * 32; jb < j1; jb += NW * 32) {
            const int j = jb + lane;
            float sc[G];
            if (j < j1) {
                const device ` + vt + `* kr = kb + (size_t)j * ks4;
                for (int g = 0; g < G; ++g) sc[g] = 0;
                for (int d4 = 0; d4 < Dh / 4; ++d4) {
                    const float4 kv = float4(kr[d4]);
                    for (int g = 0; g < G; ++g) sc[g] += dot(kv, qs[g][d4]);
                }
                for (int g = 0; g < G; ++g) sc[g] = tanh(sc[g] / ` + softcap + `) * ` + softcap + `;
            } else {
                for (int g = 0; g < G; ++g) sc[g] = -INFINITY;
            }
            float pj[G];
            for (int g = 0; g < G; ++g) {
                const float mn = max(m[g], simd_max(sc[g]));
                const float corr = exp(m[g] - mn);
                pj[g] = exp(sc[g] - mn);
                l[g] = l[g] * corr + simd_sum(pj[g]);
                o[g] *= corr;
                m[g] = mn;
            }
            const int nk = min(32, j1 - jb);
            for (int u = 0; u < nk; ++u) {
                const float4 vv = float4(vb[(size_t)(jb + u) * vs4 + lane]);
                for (int g = 0; g < G; ++g) o[g] += simd_shuffle(pj[g], u) * vv;
            }
        }
        for (int g = 0; g < G; ++g) {
            if (lane == 0) { mo[sg][g] = m[g]; lo[sg][g] = l[g]; }
            oo[sg][g][lane] = o[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < G) {
            const int g = sg;
            float M = -INFINITY;
            for (int w = 0; w < NW; ++w) M = max(M, mo[w][g]);
            float L = 0; float4 O = 0;
            for (int w = 0; w < NW; ++w) {
                const float c = (mo[w][g] == -INFINITY) ? 0.0f : exp(mo[w][g] - M);
                L += lo[w][g] * c; O += oo[w][g][lane] * c;
            }
            const size_t oi = ((size_t)t * Hq + hk * G + g) * NS + sp;
            ((device float4*)(o_out + oi * Dh))[lane] = O;
            if (lane == 0) { m_out[oi] = M; l_out[oi] = L; }
        }
`
			return &gpuKernel{
				name:    fmt.Sprintf("olala_attn_%d", len(olalaKernels)),
				inputs:  []string{"q", "keys", "values", "kstart", "slen"},
				outputs: []string{"o_out", "m_out", "l_out"},
				metal:   gpuSource{source: src, header: olalaHeader},
				strided: true,
			}
		})
	}
	NS := (S + BLK - 1) / BLK
	slen := NewArrayInt32([]int32{int32(S)}, []int32{1})
	outs := kern.run(gpuLaunch{
		outputs: []gpuOutputSpec{
			{name: "o_out", shape: []int32{int32(T), int32(d.Hq), int32(NS), int32(d.Dh)}, dtype: DTypeFloat32},
			{name: "m_out", shape: []int32{int32(T), int32(d.Hq), int32(NS)}, dtype: DTypeFloat32},
			{name: "l_out", shape: []int32{int32(T), int32(d.Hq), int32(NS)}, dtype: DTypeFloat32},
		},
		grid:        [3]int{T * d.Hk * NS * olalaAttNW * 32, 1, 1},
		threadGroup: [3]int{olalaAttNW * 32, 1, 1},
		inputs:      []*Array{q, keys, values, kstart, slen},
	})
	return outs[0], outs[1], outs[2]
}

// OlalaAttnCombine merges the key splits of OlalaAttnDecode, applies the
// differential combination and the output gate. small / big are the
// in_proj_small / in_proj outputs. Returns (T, (Hq - NNoise) * Dh) f32.
func OlalaAttnCombine(o, m, l, small, big *Array, T int, d OlalaAttnDims) *Array {
	nSig := d.Hq - d.NNoise
	snr := nSig / d.NNoise
	smallW, bigW := small.Dim(1), big.Dim(1)
	smallOff := 2*d.Hk*d.Rk + 2*d.Hk
	bigOff := d.Hq*d.Dh + 2*d.Rk*d.Dh
	kern := olalaKernel(fmt.Sprintf("comb|%v|%d|%d", d, smallW, bigW), func() *gpuKernel {
		src := fmt.Sprintf(`
        constexpr int Hq = %d, Dh = %d, SNR = %d, NN = %d, NSIG = %d;
`, d.Hq, d.Dh, snr, d.NNoise, nSig) + `
        const int NS = m_shape[2];  // key splits (run time: no recompilation as the context grows)
        const uint d = thread_position_in_threadgroup.x;
        const uint blk = threadgroup_position_in_grid.x;
        const int t = blk / NSIG;
        const int hs = blk % NSIG;            // signal head index
        const int g = hs / SNR, i = hs % SNR;
        const int hsig = g * (SNR + 1) + i, hnoi = g * (SNR + 1) + SNR;
        float res[2];
        for (int u = 0; u < 2; ++u) {
            const int hq = u == 0 ? hsig : hnoi;
            const size_t base = ((size_t)t * Hq + hq) * NS;
            float M = -INFINITY;
            for (int s = 0; s < NS; ++s) M = max(M, m[base + s]);
            float L = 0, O = 0;
            for (int s = 0; s < NS; ++s) {
                const float c = (m[base + s] == -INFINITY) ? 0.0f : exp(m[base + s] - M);
                L += l[base + s] * c; O += o[(base + s) * Dh + d] * c;
            }
            res[u] = O / L;
        }
` + fmt.Sprintf(`
        const float lam = olala_sigmoid(small[t * %d + %d + g]);
        const float gt = big[t * %d + %d + hs * Dh + d] + %s;
        out[((size_t)t * NSIG + hs) * Dh + d] = (res[0] - lam * res[1]) * (gt / (1.0f + exp(-gt)));
`, smallW, smallOff, bigW, bigOff, flt(float64(d.GateBias)))
		return &gpuKernel{
			name:    fmt.Sprintf("olala_attn_comb_%d", len(olalaKernels)),
			inputs:  []string{"o", "m", "l", "small", "big"},
			outputs: []string{"out"},
			metal:   gpuSource{source: src, header: olalaHeader},
		}
	})
	return kern.run(gpuLaunch{
		outputs:     []gpuOutputSpec{{name: "out", shape: []int32{int32(T), int32(nSig * d.Dh)}, dtype: DTypeFloat32}},
		grid:        [3]int{T * nSig * d.Dh, 1, 1},
		threadGroup: [3]int{d.Dh, 1, 1},
		inputs:      []*Array{o, m, l, small, big},
	})[0]
}

// OlalaKernelsAvailable reports whether the fused Olala kernels can run.
func OlalaKernelsAvailable() bool { return MetalIsAvailable() }
