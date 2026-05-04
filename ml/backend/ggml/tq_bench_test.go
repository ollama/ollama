package ggml

import (
	"bytes"
	"math/rand"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/turboquant"
)

// setupGPU creates a synthetic backend with one layer assigned to the first
// TQ-capable GPU. This lets us exercise the CUDA encode/dequant kernels on a
// real GPU scheduler instead of the CPU-only synthetic backend used by the
// regular unit tests.
func setupGPU(b *testing.B) (ml.Context, ml.Backend) {
	return setupGPUTB(b)
}

// setupGPUTB is the testing.TB form of setupGPU so non-benchmark tests
// (correctness tests) can share the same harness. All testing.B methods
// used here (Helper, Skip, TempDir, Fatal, Logf) are defined on testing.TB.
func setupGPUTB(b testing.TB) (ml.Context, ml.Backend) {
	b.Helper()

	initDevices()
	gpus := GPUDevices()
	if len(gpus) == 0 {
		b.Skip("no GPU devices available")
	}

	// Pick the first TQ-capable GPU (Pascal+ or RDNA1+)
	var target GPUDeviceInfo
	for _, g := range gpus {
		if g.Library == "CUDA" && g.CCMajor >= 6 {
			target = g
			break
		}
		if g.Library == "ROCm" && g.CCMajor >= 10 {
			target = g
			break
		}
	}
	if target.Name == "" {
		b.Skip("no TQ-capable GPU available")
	}
	b.Logf("using GPU: %s (cc %d.%d)", target.Name, target.CCMajor, target.CCMinor)

	f, err := os.CreateTemp(b.TempDir(), "*.bin")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	// Create a minimal model with one layer so the scheduler assigns a GPU
	// buffer type to that layer.
	kv := ggml.KV{
		"general.architecture": "test",
		"block_count":          uint32(1),
	}
	data := make([]byte, 64)
	rand.New(rand.NewSource(42)).Read(data)
	ts := []*ggml.Tensor{
		{
			Name:     "blk.0.attn_q.weight",
			Kind:     uint32(ggml.TensorTypeF32),
			Shape:    []uint64{4, 4},
			WriterTo: bytes.NewReader(data),
		},
	}

	if err := ggml.WriteGGUF(f, kv, ts); err != nil {
		b.Fatal(err)
	}

	backend, err := ml.NewBackend(f.Name(), ml.BackendParams{
		AllocMemory: true,
		GPULayers: ml.GPULayersList{
			{
				DeviceID: ml.DeviceID{ID: target.ID, Library: target.Library},
				Layers:   []int{0},
			},
		},
	})
	if err != nil {
		b.Fatal(err)
	}

	ctx := backend.NewContext().Input()
	b.Cleanup(func() {
		ctx.Close()
		backend.Close()
	})

	return ctx, backend
}

func BenchmarkTQEncodeDequantOutlierGPU(b *testing.B) {
	ctx, be := setupGPU(b)
	ggmlBackend := be.(*Backend)

	const (
		headDim      = 128
		numKVHeads   = 8
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
		nCells       = 16
	)

	mgrAny := ggmlBackend.NewTQCompressedKManager(
		headDim, numKVHeads, bits,
		turboquant.PresetTQ3K.RotationSeed,
		0, // vBits (K-only)
		outlierBits, outlierCount,
		false, // asymmetricPrimary
		0,     // qjlRows
	)
	if mgrAny == nil {
		b.Fatal("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	// Synthetic K data: Gaussian
	rng := rand.New(rand.NewSource(42))
	kData := make([]float32, headDim*numKVHeads*nCells)
	for i := range kData {
		kData[i] = float32(rng.NormFloat64())
	}

	kT := ctx.FromFloats(kData, headDim, numKVHeads, nCells)

	b.ResetTimer()
	for range b.N {
		// Encode K for all cells
		var enc ml.Tensor
		for c := range nCells {
			enc = mgr.EncodeK(ctx, 0, kT, c)
		}
		// Dequant K for all cells
		for c := range nCells {
			mgr.DequantK(ctx, 0, enc, c, 1)
		}
		ctx.Compute(kT)
	}
}

func BenchmarkTQEncodeDequantAsymmetricQJLGPU(b *testing.B) {
	ctx, be := setupGPU(b)
	ggmlBackend := be.(*Backend)

	const (
		headDim      = 128
		numKVHeads   = 8
		bits         = 3
		outlierBits  = 4
		outlierCount = 32
		nCells       = 16
		qjlRows      = 128
	)

	mgrAny := ggmlBackend.NewTQCompressedKManager(
		headDim, numKVHeads, bits,
		uint64(0x35c0ffee), // historical tq3qa rotation seed (preset retired)
		0,                  // vBits (K-only)
		outlierBits, outlierCount,
		true, // asymmetricPrimary
		qjlRows,
	)
	if mgrAny == nil {
		b.Fatal("NewTQCompressedKManager returned nil")
	}
	mgr := mgrAny.(*ggmlTQCompressedK)
	mgr.EnsureLayer(0, nCells)

	rng := rand.New(rand.NewSource(42))
	kData := make([]float32, headDim*numKVHeads*nCells)
	for i := range kData {
		kData[i] = float32(rng.NormFloat64())
	}

	kT := ctx.FromFloats(kData, headDim, numKVHeads, nCells)

	b.ResetTimer()
	for range b.N {
		var enc ml.Tensor
		for c := range nCells {
			enc = mgr.EncodeK(ctx, 0, kT, c)
		}
		for c := range nCells {
			mgr.DequantK(ctx, 0, enc, c, 1)
		}
		ctx.Compute(kT)
	}
}
