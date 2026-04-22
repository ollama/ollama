package nn

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// newBatch constructs a synthetic batch for mask/SDPA tests.
// seqOffsets defines B (length of slice) and each row's absolute start;
// L is the padded query length along InputIDs's second axis;
// qLens is per-row real query length (defaults to all L if nil).
func newBatch(seqOffsets []int32, L int, qLens []int32) *batch.Batch {
	B := len(seqOffsets)
	if qLens == nil {
		qLens = make([]int32, B)
		for i := range qLens {
			qLens[i] = int32(L)
		}
	}
	// InputIDs values don't matter for masking, only the shape.
	ids := mlx.FromValues(make([]int32, B*L), B, L)
	return &batch.Batch{
		InputIDs:     ids,
		SeqOffsets:   seqOffsets,
		SeqQueryLens: qLens,
	}
}

func TestAttentionMaskZero(t *testing.T) {
	skipIfNoMLX(t)
	var m AttentionMask
	if !m.IsZero() {
		t.Fatal("zero value should report IsZero")
	}
	if m.IsCausal() {
		t.Fatal("zero value should not report IsCausal")
	}
	b := newBatch([]int32{0}, 2, nil)
	arr := m.AsArray(b, 3, mlx.DTypeFloat32)
	if arr == nil {
		t.Fatal("zero value AsArray should return a zeros tensor, not nil")
	}
	mlx.Eval(arr)
	got := arr.Floats()
	for i, v := range got {
		if v != 0 {
			t.Fatalf("zero mask should materialize all zeros; got[%d] = %v", i, v)
		}
	}
}

func TestAttentionMaskAsArrayCausal(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 4, 6
	b := newBatch([]int32{2}, L, nil)
	arr := CausalMask().AsArray(b, K, mlx.DTypeFloat32)
	if arr == nil {
		t.Fatal("CausalMask AsArray should return a tensor")
	}
	dims := arr.Dims()
	if len(dims) != 4 || dims[0] != 1 || dims[1] != 1 || dims[2] != L || dims[3] != K {
		t.Fatalf("want shape [1,1,%d,%d], got %v", L, K, dims)
	}
	mlx.Eval(arr)
	got := arr.Floats()
	negInf := float32(math.Inf(-1))
	want := make([]float32, L*K)
	for q := range L {
		absQ := int(b.SeqOffsets[0]) + q
		for k := range K {
			if k > absQ {
				want[q*K+k] = negInf
			}
		}
	}
	for i := range want {
		if !sameF(got[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestAttentionMaskRelaxLazy(t *testing.T) {
	skipIfNoMLX(t)
	// Relax must not materialize a tensor — the perf invariant the
	// causal-flag fast path relies on. Everything else (predicates,
	// AsArray contents) is exercised by the materialization tests.
	m := CausalMask().
		Relax(0, 1, 3, 2, 5).
		Relax(0, 0, 2, 1, 4)
	if m.array != nil {
		t.Fatal("Relax should not materialize a tensor")
	}
}

// TestAttentionMaskRelaxNoopRectsMatchCausal pins the contract that
// rectangles which can't change any cell — empty in q or k, or fully
// inside the causal triangle — must produce the same materialized
// tensor as plain causal.
func TestAttentionMaskRelaxNoopRectsMatchCausal(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 4, 6
	b := newBatch([]int32{0}, L, nil)
	want := CausalMask().AsArray(b, K, mlx.DTypeFloat32)
	mlx.Eval(want)
	wantF := want.Floats()

	cases := []struct {
		name               string
		qLo, qHi, kLo, kHi int
	}{
		{"empty Q rect", 2, 2, 0, 3},
		{"empty K rect", 0, 3, 2, 2},
		{"fully under causal", 5, 7, 0, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := CausalMask().Relax(0, tc.qLo, tc.qHi, tc.kLo, tc.kHi)
			arr := m.AsArray(b, K, mlx.DTypeFloat32)
			mlx.Eval(arr)
			got := arr.Floats()
			for i := range wantF {
				if !sameF(got[i], wantF[i]) {
					t.Fatalf("index %d: want %v, got %v", i, wantF[i], got[i])
				}
			}
		})
	}
}

func TestAttentionMaskAsArrayWithRelax(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 4, 6
	b := newBatch([]int32{0}, L, nil)
	arr := CausalMask().Relax(0, 1, 3, 2, 5).AsArray(b, K, mlx.DTypeFloat32)
	if arr == nil {
		t.Fatal("expected tensor")
	}
	mlx.Eval(arr)
	got := arr.Floats()
	negInf := float32(math.Inf(-1))
	want := make([]float32, L*K)
	for q := range L {
		for k := range K {
			if k > q {
				want[q*K+k] = negInf
			}
		}
	}
	for q := 1; q < 3; q++ {
		for k := 2; k < 5; k++ {
			want[q*K+k] = 0
		}
	}
	for i := range want {
		if !sameF(got[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestAttentionMaskAsArrayPerRow(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 3, 5
	b := newBatch([]int32{0, 2}, L, nil)
	m := CausalMask().
		Relax(0, 0, 2, 0, 3).
		Relax(1, 3, 5, 2, 5)
	arr := m.AsArray(b, K, mlx.DTypeFloat32)
	if arr == nil {
		t.Fatal("expected tensor")
	}
	dims := arr.Dims()
	if dims[0] != 2 {
		t.Fatalf("expected batch dim 2, got %v", dims)
	}
	mlx.Eval(arr)
	got := arr.Floats()
	negInf := float32(math.Inf(-1))

	want := make([]float32, 2*L*K)
	for bi, off := range b.SeqOffsets {
		for q := range L {
			absQ := int(off) + q
			for k := range K {
				if k > absQ {
					want[bi*L*K+q*K+k] = negInf
				}
			}
		}
	}
	for q := range 2 {
		for k := range 3 {
			want[0*L*K+q*K+k] = 0
		}
	}
	for q := 1; q < 3; q++ {
		for k := 2; k < 5; k++ {
			want[1*L*K+q*K+k] = 0
		}
	}
	for i := range want {
		if !sameF(got[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestQPaddingMask(t *testing.T) {
	skipIfNoMLX(t)
	L := 4
	// Row 0 fully real; row 1 has 2 real queries.
	b := newBatch([]int32{0, 0}, L, []int32{int32(L), 2})
	m := QPaddingMask(b, mlx.DTypeFloat32)
	if m.array == nil {
		t.Fatal("expected q-padding tensor")
	}
	mlx.Eval(m.array)
	got := m.array.Floats()
	negInf := float32(math.Inf(-1))
	want := make([]float32, 2*L)
	// Row 0: no blocking; row 1: q >= 2 blocked.
	for q := 2; q < L; q++ {
		want[1*L+q] = negInf
	}
	for i := range want {
		if !sameF(got[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestKPaddingMask(t *testing.T) {
	skipIfNoMLX(t)
	K := 5
	// Row 0 full keys; row 1 has 3 real keys.
	b := newBatch([]int32{0, 0}, 4, nil)
	kLens := []int32{int32(K), 3}
	m := KPaddingMask(b, K, kLens, mlx.DTypeFloat32)
	if m.array == nil {
		t.Fatal("expected k-padding tensor")
	}
	mlx.Eval(m.array)
	got := m.array.Floats()
	negInf := float32(math.Inf(-1))
	want := make([]float32, 2*K)
	for k := 3; k < K; k++ {
		want[1*K+k] = negInf
	}
	for i := range want {
		if !sameF(got[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestQPaddingMaskZeroWhenFull(t *testing.T) {
	skipIfNoMLX(t)
	b := newBatch([]int32{0}, 4, nil)
	m := QPaddingMask(b, mlx.DTypeFloat32)
	if !m.IsZero() {
		t.Fatal("QPaddingMask at full queries should be zero")
	}
}

func TestKPaddingMaskZeroWhenFull(t *testing.T) {
	skipIfNoMLX(t)
	K := 4
	b := newBatch([]int32{0}, 4, nil)
	kLens := []int32{int32(K)}
	m := KPaddingMask(b, K, kLens, mlx.DTypeFloat32)
	if !m.IsZero() {
		t.Fatal("KPaddingMask at full keys should be zero")
	}
}

func TestAttentionMaskCombineCausal(t *testing.T) {
	skipIfNoMLX(t)
	var z AttentionMask
	got := z.Intersect(CausalMask())
	if !got.IsCausal() {
		t.Fatal("zero + CausalMask should be pure causal")
	}
	got = CausalMask().Intersect(z)
	if !got.IsCausal() {
		t.Fatal("CausalMask + zero should be pure causal")
	}
	got = CausalMask().Intersect(CausalMask())
	if !got.IsCausal() {
		t.Fatal("causal + causal should stay pure causal")
	}
}

func TestAttentionMaskCombineRelaxDroppedAgainstCausal(t *testing.T) {
	skipIfNoMLX(t)
	relaxed := CausalMask().Relax(0, 1, 3, 2, 5)
	got := relaxed.Intersect(CausalMask())
	if !got.IsCausal() {
		t.Fatal("causal-with-Relax + causal should drop relaxations and stay pure causal")
	}
	got = CausalMask().Intersect(relaxed)
	if !got.IsCausal() {
		t.Fatal("causal + causal-with-Relax should drop relaxations and stay pure causal")
	}

	// Disjoint relaxations on two causals also drop — neither side
	// agrees to release the cells the other side relaxed.
	got = CausalMask().Relax(0, 1, 3, 2, 5).Intersect(CausalMask().Relax(0, 5, 7, 6, 9))
	if !got.IsCausal() {
		t.Fatal("disjoint relaxations on two causals should drop and stay pure causal")
	}
}

func TestAttentionMaskCombineRelaxIntersect(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 6, 6
	b := newBatch([]int32{0}, L, nil)

	// Overlapping rects on two causals: the surviving relaxation is
	// the geometric intersection — q in [1,3) ∩ [2,5) = [2,3),
	// k in [2,5) ∩ [3,6) = [3,5).
	m := CausalMask().Relax(0, 1, 3, 2, 5).Intersect(CausalMask().Relax(0, 2, 5, 3, 6))
	if m.IsCausal() {
		t.Fatal("overlapping relaxations should survive as their intersection, not collapse to pure causal")
	}
	arr := m.AsArray(b, K, mlx.DTypeFloat32)
	if arr == nil {
		t.Fatal("expected tensor")
	}
	mlx.Eval(arr)
	vals := arr.Floats()
	negInf := float32(math.Inf(-1))
	want := make([]float32, L*K)
	for q := range L {
		for k := range K {
			if k > q {
				want[q*K+k] = negInf
			}
		}
	}
	// Intersection rect: q ∈ [2,3), k ∈ [3,5).
	for q := 2; q < 3; q++ {
		for k := 3; k < 5; k++ {
			want[q*K+k] = 0
		}
	}
	for i := range want {
		if !sameF(vals[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], vals[i])
		}
	}
}

func TestAttentionMaskCombineRelaxKeptAgainstNonCausal(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 4, 6
	b := newBatch([]int32{0}, L, nil)

	// Pad q=3 — non-causal additive contribution that should leave
	// the relaxation intact (the rect releases above-diagonal cells
	// q in [1,3), k in [2,5) where k > q).
	pad := QPaddingMask(newBatch([]int32{0}, L, []int32{3}), mlx.DTypeFloat32)
	if pad.IsZero() {
		t.Fatal("padding mask should be non-zero")
	}
	got := CausalMask().Relax(0, 1, 3, 2, 5).Intersect(pad)
	arr := got.AsArray(b, K, mlx.DTypeFloat32)
	if arr == nil {
		t.Fatal("expected tensor")
	}
	mlx.Eval(arr)
	vals := arr.Floats()
	negInf := float32(math.Inf(-1))
	want := make([]float32, L*K)
	for q := range L {
		for k := range K {
			if k > q {
				want[q*K+k] = negInf
			}
		}
	}
	for q := 1; q < 3; q++ {
		for k := 2; k < 5; k++ {
			want[q*K+k] = 0
		}
	}
	for q := 3; q < L; q++ {
		for k := range K {
			want[q*K+k] = negInf
		}
	}
	for i := range want {
		if !sameF(vals[i], want[i]) {
			t.Fatalf("index %d: want %v, got %v", i, want[i], vals[i])
		}
	}
}

func TestAttentionMaskCombineArrays(t *testing.T) {
	skipIfNoMLX(t)
	a := mlx.FromValues([]float32{0, 0, 0, 0}, 1, 1, 2, 2)
	bb := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	sum := ArrayMask(a).Intersect(ArrayMask(bb))
	if sum.array == nil {
		t.Fatal("array + array should produce array")
	}
	mlx.Eval(sum.array)
	got := sum.array.Floats()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: want %v, got %v", i, want[i], got[i])
		}
	}
}

func TestAttentionMaskRelaxPanicOnArray(t *testing.T) {
	skipIfNoMLX(t)
	a := mlx.FromValues([]float32{0}, 1, 1, 1, 1)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Relax on ArrayMask should panic")
		}
	}()
	ArrayMask(a).Relax(0, 0, 1, 0, 1)
}

func TestAttentionMaskRelaxPanicOnZero(t *testing.T) {
	skipIfNoMLX(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Relax on zero mask should panic")
		}
	}()
	var z AttentionMask
	z.Relax(0, 0, 1, 0, 1)
}

func sameF(a, b float32) bool {
	if math.IsInf(float64(a), -1) && math.IsInf(float64(b), -1) {
		return true
	}
	return a == b
}

// sdpaInputs builds non-trivial Q/K/V so masking actually changes the
// kernel output. With zero K/V, SDPA returns zero regardless of mask
// and "parity" tests pass even when the mask path is broken.
func sdpaInputs(L, K int) (q, k, v *mlx.Array) {
	const D = 4
	qVals := make([]float32, L*D)
	for i := range qVals {
		qVals[i] = 0.1 * float32(i+1)
	}
	kVals := make([]float32, K*D)
	for i := range kVals {
		kVals[i] = 0.07 * float32(i+1)
	}
	vVals := make([]float32, K*D)
	for i := range vVals {
		vVals[i] = float32(i+1) - 0.5*float32(K*D)
	}
	q = mlx.FromValues(qVals, 1, 1, L, D)
	k = mlx.FromValues(kVals, 1, 1, K, D)
	v = mlx.FromValues(vVals, 1, 1, K, D)
	return
}

func TestSDPACausalParity(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 4, 4
	q, k, v := sdpaInputs(L, K)
	b := newBatch([]int32{int32(K - L)}, L, nil)
	got := ScaledDotProductAttention(b, q, 1.0,
		WithKV(k, v, []int32{int32(K)}),
		WithMask(CausalMask()),
	)
	want := mlx.FastScaledDotProductAttention(q, k, v, 1.0, "causal", nil)
	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("index %d: want %v, got %v", i, ws[i], gs[i])
		}
	}
}

func TestSDPAZeroMaskParity(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 4, 4
	q, k, v := sdpaInputs(L, K)
	b := newBatch([]int32{0}, L, nil)
	got := ScaledDotProductAttention(b, q, 1.0, WithKV(k, v, []int32{int32(K)}))
	want := mlx.FastScaledDotProductAttention(q, k, v, 1.0, "", nil)
	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("index %d: want %v, got %v", i, ws[i], gs[i])
		}
	}
}

func TestSDPAArrayMaskParity(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 3, 3
	q, k, v := sdpaInputs(L, K)
	b := newBatch([]int32{0}, L, nil)
	mask := mlx.FromValues([]float32{
		0, -1, -1,
		0, 0, -1,
		0, 0, 0,
	}, 1, 1, 3, 3)
	got := ScaledDotProductAttention(b, q, 1.0,
		WithKV(k, v, []int32{int32(K)}),
		WithMask(ArrayMask(mask)),
	)
	want := mlx.FastScaledDotProductAttention(q, k, v, 1.0, "array", mask)
	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("index %d: want %v, got %v", i, ws[i], gs[i])
		}
	}
}

func TestSDPARelaxMaskMaterializes(t *testing.T) {
	skipIfNoMLX(t)
	L, K := 3, 5
	q, k, v := sdpaInputs(L, K)
	b := newBatch([]int32{int32(K - L)}, L, nil)
	got := ScaledDotProductAttention(b, q, 1.0,
		WithKV(k, v, []int32{int32(K)}),
		WithMask(CausalMask().Relax(0, 3, 5, 2, 5)),
	)
	ref := CausalMask().Relax(0, 3, 5, 2, 5).AsArray(b, K, k.DType())
	want := mlx.FastScaledDotProductAttention(q, k, v, 1.0, "array", ref)
	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("index %d: want %v, got %v", i, ws[i], gs[i])
		}
	}
}

func TestSDPAPanicsWithBothKVAndHistory(t *testing.T) {
	skipIfNoMLX(t)
	L := 3
	q, k, v := sdpaInputs(L, L)
	b := newBatch([]int32{0}, L, nil)
	history := NewKVHistory(k, v, nil)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when both WithKV and WithKVHistory are supplied")
		}
	}()
	ScaledDotProductAttention(b, q, 1.0, WithKV(k, v, []int32{int32(L)}), WithKVHistory(history))
}

func TestSDPAMLAHistorySlicesVFromK(t *testing.T) {
	skipIfNoMLX(t)
	L, D, valueDim := 2, 5, 3
	kBuf := make([]float32, 1*1*L*D)
	for i := range kBuf {
		kBuf[i] = float32(i) + 1
	}
	k := mlx.FromValues(kBuf, 1, 1, L, D)
	v := mlx.Zeros(mlx.DTypeFloat32, 1, 1, L, valueDim)
	history := NewKVHistory(k, v, nil)

	q := mlx.Zeros(mlx.DTypeFloat32, 1, 1, L, D)
	b := newBatch([]int32{0}, L, nil)
	got := ScaledDotProductAttention(b, q, 1.0,
		WithMLAHistory(history, valueDim),
	)
	vRef := k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, valueDim))
	want := mlx.FastScaledDotProductAttention(q, k, vRef, 1.0, "", nil)
	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	for i := range ws {
		if gs[i] != ws[i] {
			t.Fatalf("index %d: want %v, got %v", i, ws[i], gs[i])
		}
	}
}

func TestSDPAPanicsWithoutKV(t *testing.T) {
	skipIfNoMLX(t)
	q := mlx.FromValues(make([]float32, 4), 1, 1, 1, 4)
	b := newBatch([]int32{0}, 1, nil)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic when no K/V supplied")
		}
	}()
	ScaledDotProductAttention(b, q, 1.0)
}

// fillTensor builds a [B, H, T, D] float32 tensor whose entries are
// distinct, non-zero, and predictable so per-row slices stay distinct.
func fillTensor(seed float32, B, H, T, D int) *mlx.Array {
	vals := make([]float32, B*H*T*D)
	for i := range vals {
		vals[i] = seed + 0.05*float32(i)
	}
	return mlx.FromValues(vals, B, H, T, D)
}

// TestSDPAMultiSequenceParity drives a B=2 batch with mixed
// SeqOffsets and SeqQueryLens through ScaledDotProductAttention via
// the no-cache (WithKV) path, then compares each row's real
// positions against a B=1 reference at that row's offset and length.
// Padded-tail outputs are unconstrained and not checked. Pins the
// central multi-sequence contract: right-padded rows must produce
// per-row outputs that don't depend on the padded tails.
func TestSDPAMultiSequenceParity(t *testing.T) {
	skipIfNoMLX(t)
	const H, D = 1, 4
	const L, K = 4, 6
	const qShort, kShort = 2, 2
	const scale = 1.0

	q := fillTensor(0.5, 2, H, L, D)
	k := fillTensor(-0.3, 2, H, K, D)
	v := fillTensor(0.7, 2, H, K, D)
	b := newBatch([]int32{2, 0}, L, []int32{int32(L), int32(qShort)})

	got := ScaledDotProductAttention(b, q, scale,
		WithKV(k, v, []int32{int32(K), int32(kShort)}),
		WithMask(CausalMask()))
	mlx.Eval(got)
	gotF := got.Floats()

	// Row 0: full Q at offset 2, full K. B=1 reference.
	q0 := mlx.SliceStartStop(q, []int32{0, 0, 0, 0}, []int32{1, H, L, D})
	k0 := mlx.SliceStartStop(k, []int32{0, 0, 0, 0}, []int32{1, H, K, D})
	v0 := mlx.SliceStartStop(v, []int32{0, 0, 0, 0}, []int32{1, H, K, D})
	b0 := newBatch([]int32{2}, L, nil)
	ref0 := ScaledDotProductAttention(b0, q0, scale,
		WithKV(k0, v0, []int32{int32(K)}),
		WithMask(CausalMask()))
	mlx.Eval(ref0)
	ref0F := ref0.Floats()

	// Row 1: real Q at offset 0, length qShort, with kShort real keys.
	q1 := mlx.SliceStartStop(q, []int32{1, 0, 0, 0}, []int32{2, H, int32(qShort), D})
	k1 := mlx.SliceStartStop(k, []int32{1, 0, 0, 0}, []int32{2, H, int32(kShort), D})
	v1 := mlx.SliceStartStop(v, []int32{1, 0, 0, 0}, []int32{2, H, int32(kShort), D})
	b1 := newBatch([]int32{0}, qShort, nil)
	ref1 := ScaledDotProductAttention(b1, q1, scale,
		WithKV(k1, v1, []int32{int32(kShort)}),
		WithMask(CausalMask()))
	mlx.Eval(ref1)
	ref1F := ref1.Floats()

	// got is [2, H, L, D] = [B=2, 1, 4, 4]. Row 0 is got[0,...] and
	// must match ref0 over the full [L, D]. Row 1 is got[1,...] and
	// must match ref1 over [qShort, D] only — padded positions are
	// unconstrained.
	rowStride := H * L * D
	for i := range rowStride {
		if !approxEqual(gotF[i], ref0F[i], 1e-5) {
			t.Fatalf("row 0 [%d]: got %v, want %v", i, gotF[i], ref0F[i])
		}
	}
	for q := range qShort {
		for d := range D {
			gotI := rowStride + q*D + d
			refI := q*D + d
			if !approxEqual(gotF[gotI], ref1F[refI], 1e-5) {
				t.Fatalf("row 1 [q=%d,d=%d]: got %v, want %v", q, d, gotF[gotI], ref1F[refI])
			}
		}
	}
}
