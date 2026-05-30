package ggml

// TestTQFusedFlashAttentionIndexedLocs verifies kernel_tq_fattn_vec_f16_outlier
// honours the optional `locs` buffer for indexed cell addressing. With locs,
// the kernel must read cell c from physical slot locs[c] rather than from the
// contiguous slot firstCell+c.
//
// Without the phys_cell remap inside the kernel body, the fattn-vec kernels
// silently fall back to contiguous addressing, returning wrong K (and V) for
// every cell. The symptom is wrong attention output that is hard to spot
// because the kernel still reads valid (just incorrectly-sourced) data.
//
// Setup: encode the same N keys via two paths through the same manager:
//
//	Pass A (contiguous): EncodeK(firstCell=0)         → fattn_vec( locs=nil   )
//	Pass B (indexed):    EncodeKAt(locs=permutation)  → fattn_vec( locs=perm  )
//
// Both passes attend to the same N keys in the same logical order, so the
// attention output must match within scalar-quantization slop. If the fattn
// kernel ignores `locs` it will read cells [0..N-1] in pass B (which now hold
// permuted keys), producing diverging output.

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// makeIndexedFattnSetup builds the data + tensors shared by the contiguous
// and indexed passes. Keeps the data identical across the two backends so
// the comparison isolates exactly the indexed-addressing remap.
type indexedFattnFixture struct {
	headDim, nKVHeads, nCells int
	bits, outlierBits         int
	outlierCount              int
	rotationSeed              uint64
	kData                     []float32
	qRotated                  []float32
	vF16Bytes                 []byte
	vF32                      []float32
}

func newIndexedFattnFixture() *indexedFattnFixture {
	const (
		headDim      = 128
		nKVHeads     = 4
		nCells       = 8
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
	)

	rotationSeed := turboquant.PresetTQ3K.RotationSeed
	var rngState uint64 = 0xfeedface_00000000 | uint64(headDim)

	kData := make([]float32, headDim*nKVHeads*nCells)
	for i := range kData {
		kData[i] = float32(tqGaussian(&rngState))
	}
	for c := range nCells {
		for h := range nKVHeads {
			for range 8 {
				d := int(tqSplitmix64(&rngState) % uint64(headDim))
				kData[(c*nKVHeads+h)*headDim+d] *= 4.0
			}
		}
	}

	layerSeed := rotationSeed ^ uint64(1)
	rotation := turboquant.BuildRotation(headDim, layerSeed)
	qRaw := make([]float32, headDim*nKVHeads)
	for i := range qRaw {
		qRaw[i] = float32(tqGaussian(&rngState)) * 0.5
	}
	qRotated := make([]float32, headDim*nKVHeads)
	for h := range nKVHeads {
		copy(qRotated[h*headDim:], turboquant.ApplyRotation(qRaw[h*headDim:(h+1)*headDim], rotation))
	}

	vF32 := make([]float32, headDim*nCells*nKVHeads)
	for i := range vF32 {
		vF32[i] = float32(tqGaussian(&rngState)) * 0.25
	}

	return &indexedFattnFixture{
		headDim: headDim, nKVHeads: nKVHeads, nCells: nCells,
		bits: bits, outlierBits: outlierBits, outlierCount: outlierCount,
		rotationSeed: rotationSeed,
		kData:        kData,
		qRotated:     qRotated,
		vF16Bytes:    tqF32SliceToF16Bytes(vF32),
		vF32:         vF32,
	}
}

// runFattn runs encode → fattn over the K layout produced by either contiguous
// or indexed encode; returns the GPU attention output [headDim*nKVHeads]f32.
func runFattn(t *testing.T, fx *indexedFattnFixture, indexed bool, locs []int32) []float32 {
	t.Helper()

	ctx, be := setupTQAny(t)
	b := be.(*Backend)

	mgrAny := b.NewTQCompressedKManager(
		fx.headDim, fx.nKVHeads, fx.bits,
		fx.rotationSeed,
		0, // K-only
		fx.outlierBits, fx.outlierCount,
		false, // symmetric — clean compare
	)
	if mgrAny == nil {
		t.Skip("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)

	// Capacity must cover the largest physical slot we'll write into. In
	// contiguous mode this is nCells; in indexed mode it's max(locs)+1.
	capacity := fx.nCells
	if indexed {
		for _, s := range locs {
			if int(s)+1 > capacity {
				capacity = int(s) + 1
			}
		}
	}
	mgr.EnsureLayer(0, capacity)
	if !mgr.fusedKernelSupports() {
		t.Skipf("fused kernel not supported for headDim=%d bits=%d", fx.headDim, fx.bits)
	}

	// Build K, Q, V tensors. The kernel reads cell c from physical slot:
	//   contig:  cell_rel = c, V indexed by cell_rel
	//   indexed: cell_addr = locs[c],  V indexed by cell_addr
	//
	// Production behaviour mirrors that: Causal.Put with locs writes V (and
	// K) at physical slot locs[c], so V[locs[c]] == V_for_cell_c. We have to
	// reproduce the same in the test V buffer or fattn will read uninitialised
	// V slots in the indexed pass and the comparison will be meaningless.
	kTensor := ctx.FromFloats(fx.kData, fx.headDim, fx.nKVHeads, fx.nCells)
	qTensor := ctx.FromFloats(fx.qRotated, fx.headDim, 1, fx.nKVHeads)

	var vTensor ml.Tensor
	if indexed {
		// V layout: ne[0]=headDim, ne[1]=capacity, ne[2]=nKVHeads.
		// vPerm[d, locs[c], h] = vF32[d, c, h].
		vPerm := make([]float32, fx.headDim*capacity*fx.nKVHeads)
		for c := range fx.nCells {
			dst := int(locs[c])
			for h := range fx.nKVHeads {
				srcBase := c*fx.headDim + h*fx.headDim*fx.nCells
				dstBase := dst*fx.headDim + h*fx.headDim*capacity
				copy(vPerm[dstBase:dstBase+fx.headDim], fx.vF32[srcBase:srcBase+fx.headDim])
			}
		}
		vTensor = ctx.FromBytes(ml.DTypeF16, tqF32SliceToF16Bytes(vPerm), fx.headDim, capacity, fx.nKVHeads)
	} else {
		vTensor = ctx.FromBytes(ml.DTypeF16, fx.vF16Bytes, fx.headDim, fx.nCells, fx.nKVHeads)
	}

	var enc ml.Tensor
	var tqkRaw ml.Tensor
	var ok bool
	if indexed {
		locsTensor := ctx.FromInts(locs, fx.nCells)
		enc = mgr.EncodeKAt(ctx, 0, kTensor, locsTensor)
		if enc == nil {
			t.Fatalf("EncodeKAt returned nil")
		}
		tqkRaw, ok = mgr.GetAsTQTensorAt(ctx, 0, enc, locsTensor)
	} else {
		enc = mgr.EncodeK(ctx, 0, kTensor, 0)
		if enc == nil {
			t.Fatalf("EncodeK returned nil")
		}
		tqkRaw, ok = mgr.GetAsTQTensor(ctx, 0, enc, 0, fx.nCells)
	}
	if !ok || tqkRaw == nil {
		t.Fatalf("GetAsTQTensor%v returned (nil, %v)", indexedSuffix(indexed), ok)
	}

	tqk := tqkRaw.(*tqTensor)
	attnScale := 1.0 / math.Sqrt(float64(fx.headDim))
	attnOut := b.tqFlashAttention(ctx, qTensor.(*Tensor), tqk, vTensor.(*Tensor),
		nil, attnScale, 0)
	if attnOut == nil {
		t.Fatalf("tqFlashAttention returned nil")
	}

	ctx.Forward(enc, attnOut).Compute(enc, attnOut)
	out := attnOut.Floats()
	if len(out) != fx.headDim*fx.nKVHeads {
		t.Fatalf("out len = %d, want %d", len(out), fx.headDim*fx.nKVHeads)
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("out[%d] = %v (NaN/Inf — kernel reading wrong cell)", i, v)
		}
	}
	return out
}

func indexedSuffix(indexed bool) string {
	if indexed {
		return "At"
	}
	return ""
}

func TestTQFusedFlashAttentionIndexedLocs(t *testing.T) {
	fx := newIndexedFattnFixture()

	// Permutation of [0..nCells-1] — non-identity so every cell c is mapped
	// to a different physical slot. max(perm) = nCells-1 keeps the V tensor
	// the same nCells size in both passes so the dispatcher reads the same
	// nCells (= V->ne[1]) regardless of mode. (A wider permutation that
	// reaches into a larger physical capacity would also expose the
	// orthogonal kvcache routing gap where Get() never threads locs into
	// GetAsTQTensor; this test isolates the kernel-body remap question.)
	perm := []int32{3, 1, 5, 0, 7, 2, 6, 4}
	if len(perm) != fx.nCells {
		t.Fatalf("perm len mismatch")
	}

	contigOut := runFattn(t, fx, false, nil)
	indexedOut := runFattn(t, fx, true, perm)

	// Two encodes of the SAME logical K cells (in the same logical order),
	// just laid out at different physical slots. Attention output should
	// match within scalar-quantization slop. If the fattn kernel ignores
	// args.hasLocs and reads cells [0..N-1] contiguously, the indexed pass
	// reads from physical slots {0,1,2,...,7} (zero-initialised cells in
	// the indexed encode case) and produces wildly different output.
	const tol float32 = 0.05
	var maxDiff float32
	var nMismatches int
	for i := range contigOut {
		diff := float32(math.Abs(float64(contigOut[i]) - float64(indexedOut[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			nMismatches++
			if nMismatches <= 5 {
				t.Logf("mismatch i=%d contig=%.4f indexed=%.4f diff=%.4f",
					i, contigOut[i], indexedOut[i], diff)
			}
		}
	}
	t.Logf("indexed-vs-contiguous fattn: max_diff=%.4f mismatches=%d/%d (tol=%.2f)",
		maxDiff, nMismatches, fx.headDim*fx.nKVHeads, tol)
	if nMismatches > 0 {
		t.Fatalf("%d elements differ beyond tolerance %.2f — fattn kernel does not honour args.hasLocs/locs[]; check phys_cell remap in K and V loops",
			nMismatches, tol)
	}
}
