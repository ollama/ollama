package ggml

import (
	"bytes"
	"math"
	"os"
	"strconv"
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

func TestGatedDeltaNetReferenceScalarGate(t *testing.T) {
	ctx := setup(t)

	q := ctx.FromFloats([]float32{
		0.3, -0.4,
		0.6, 0.2,
	}, 2, 1, 2, 1)
	k := ctx.FromFloats([]float32{
		0.2, 0.7,
		-0.5, 0.4,
	}, 2, 1, 2, 1)
	v := ctx.FromFloats([]float32{
		1.0, -0.5,
		0.25, 0.8,
	}, 2, 1, 2, 1)
	gate := ctx.FromFloats([]float32{-0.2, 0.1}, 1, 1, 2, 1)
	beta := ctx.FromFloats([]float32{0.3, 0.6}, 1, 1, 2, 1)
	state := ctx.FromFloats([]float32{
		0.1, -0.2,
		0.05, 0.3,
	}, 2, 2, 1, 1)

	fused := q.GatedDeltaNet(ctx, k, v, gate, beta, state)
	fusedOut := gatedDeltaNetOutput(ctx, fused, 2, 1, 2, 1)
	fusedState := gatedDeltaNetState(ctx, fused, 2, 1, 2, 1)
	refOut, refState := referenceGatedDeltaNet(ctx, q, k, v, gate, beta, state)

	ctx.Forward(fusedOut, fusedState, refOut, refState).Compute(fusedOut, fusedState, refOut, refState)

	assertCloseFloats(t, "output", refOut.Floats(), fusedOut.Floats(), 1e-5)
	assertCloseFloats(t, "state", refState.Floats(), fusedState.Floats(), 1e-5)
}

func TestGatedDeltaNetReferenceMultiHeadBroadcast(t *testing.T) {
	ctx := setup(t)

	q := ctx.FromFloats([]float32{0.25, -0.6}, 2, 1, 1, 1)
	k := ctx.FromFloats([]float32{0.4, 0.1}, 2, 1, 1, 1)
	v := ctx.FromFloats([]float32{
		0.7, -0.3,
		-0.2, 0.9,
	}, 2, 2, 1, 1)
	gate := ctx.FromFloats([]float32{
		0.0, -0.4,
		0.2, -0.1,
	}, 2, 2, 1, 1)
	beta := ctx.FromFloats([]float32{0.5, 0.8}, 1, 2, 1, 1)
	state := ctx.FromFloats([]float32{
		0.2, -0.1,
		0.05, 0.3,

		-0.2, 0.4,
		0.1, -0.05,
	}, 2, 2, 2, 1)

	fused := q.GatedDeltaNet(ctx, k, v, gate, beta, state)
	fusedOut := gatedDeltaNetOutput(ctx, fused, 2, 2, 1, 1)
	fusedState := gatedDeltaNetState(ctx, fused, 2, 2, 1, 1)
	refOut, refState := referenceGatedDeltaNet(ctx, q, k, v, gate, beta, state)

	ctx.Forward(fusedOut, fusedState, refOut, refState).Compute(fusedOut, fusedState, refOut, refState)

	assertCloseFloats(t, "output", refOut.Floats(), fusedOut.Floats(), 1e-5)
	assertCloseFloats(t, "state", refState.Floats(), fusedState.Floats(), 1e-5)
}

func TestGatedDeltaNetCUDAParity(t *testing.T) {
	device, ok := firstCUDATestDevice(t)
	if !ok {
		t.Skip("CUDA device not available")
	}

	cpuBackend := newGatedDeltaNetTestBackend(t, nil)
	defer cpuBackend.Close()

	cudaBackend := newGatedDeltaNetTestBackend(t, &device)
	defer cudaBackend.Close()

	for _, headDim := range []int{16, 32, 64, 128} {
		t.Run("headDim="+strconv.Itoa(headDim), func(t *testing.T) {
			cpuOut, cpuState := runGatedDeltaNetOp(t, cpuBackend, headDim)
			cudaOut, cudaState := runGatedDeltaNetOp(t, cudaBackend, headDim)

			assertCloseFloats(t, "output", cpuOut, cudaOut, 2e-4)
			assertCloseFloats(t, "state", cpuState, cudaState, 2e-4)
		})
	}
}

func referenceGatedDeltaNet(ctx ml.Context, q, k, v, gate, beta, state ml.Tensor) (ml.Tensor, ml.Tensor) {
	headDim := v.Dim(0)
	numHeads := v.Dim(1)
	numTokens := v.Dim(2)
	numSeqs := v.Dim(3)

	state = state.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx, headDim, headDim, numHeads, numSeqs)

	var outputs []ml.Tensor
	for i := 0; i < numTokens; i++ {
		qi := q.Slice(ctx, 2, i, i+1, 1)
		ki := k.Slice(ctx, 2, i, i+1, 1)
		vi := v.Slice(ctx, 2, i, i+1, 1)
		gatei := gate.Slice(ctx, 2, i, i+1, 1)
		betai := beta.Slice(ctx, 2, i, i+1, 1)

		var out ml.Tensor
		state, out = referenceGatedDeltaNetStep(ctx, qi, ki, vi, gatei, betai, state)
		outputs = append(outputs, out)
	}

	output := outputs[0]
	for _, out := range outputs[1:] {
		output = output.Concat(ctx, out, 2)
	}

	state = state.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx, headDim, headDim, numHeads, numSeqs)
	return output.Contiguous(ctx), state
}

func firstCUDATestDevice(tb testing.TB) (ml.DeviceID, bool) {
	tb.Helper()

	backend := newGatedDeltaNetTestBackend(tb, nil)
	defer backend.Close()

	for _, gpu := range backend.BackendMemory().GPUs {
		if gpu.Library == "CUDA" {
			return gpu.DeviceID, true
		}
	}

	return ml.DeviceID{}, false
}

func newGatedDeltaNetTestBackend(tb testing.TB, device *ml.DeviceID) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	tensors := []*fsggml.Tensor{
		{
			Name:     "blk.0.dummy.weight",
			Kind:     uint32(fsggml.TensorTypeF32),
			Shape:    []uint64{1},
			WriterTo: bytes.NewReader(make([]byte, 4)),
		},
	}

	if err := fsggml.WriteGGUF(f, fsggml.KV{
		"general.architecture": "test",
		"block_count":          uint32(1),
	}, tensors); err != nil {
		tb.Fatal(err)
	}

	params := ml.BackendParams{AllocMemory: true}
	if device != nil {
		params.GPULayers = ml.GPULayersList{{
			DeviceID: *device,
			Layers:   []int{0},
		}}
	}

	backend, err := ml.NewBackend(f.Name(), params)
	if err != nil {
		tb.Fatal(err)
	}

	return backend
}

func runGatedDeltaNetOp(tb testing.TB, backend ml.Backend, headDim int) ([]float32, []float32) {
	tb.Helper()

	const (
		numQHeads = 1
		numVHeads = 2
		numTokens = 2
		numSeqs   = 2
	)

	ctx := backend.NewContext().Layer(0)
	defer ctx.Close()

	q := ctx.FromFloats(gatedDeltaNetFloats(headDim*numQHeads*numTokens*numSeqs, 0.10), headDim, numQHeads, numTokens, numSeqs)
	k := ctx.FromFloats(gatedDeltaNetFloats(headDim*numQHeads*numTokens*numSeqs, 0.20), headDim, numQHeads, numTokens, numSeqs)
	v := ctx.FromFloats(gatedDeltaNetFloats(headDim*numVHeads*numTokens*numSeqs, 0.30), headDim, numVHeads, numTokens, numSeqs)
	gate := ctx.FromFloats(gatedDeltaNetFloats(headDim*numVHeads*numTokens*numSeqs, 0.40), headDim, numVHeads, numTokens, numSeqs)
	beta := ctx.FromFloats(gatedDeltaNetFloats(numVHeads*numTokens*numSeqs, 0.50), 1, numVHeads, numTokens, numSeqs)
	state := ctx.FromFloats(gatedDeltaNetFloats(headDim*headDim*numVHeads*numSeqs, 0.60), headDim, headDim, numVHeads, numSeqs)

	fused := q.GatedDeltaNet(ctx, k, v, gate, beta, state)
	fusedOut := gatedDeltaNetOutput(ctx, fused, headDim, numVHeads, numTokens, numSeqs)
	fusedState := gatedDeltaNetState(ctx, fused, headDim, numVHeads, numTokens, numSeqs)

	ctx.Forward(fusedOut, fusedState).Compute(fusedOut, fusedState)

	return fusedOut.Floats(), fusedState.Floats()
}

func gatedDeltaNetFloats(n int, seed float64) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(math.Sin(seed+float64(i+1)*0.071) * 0.2)
	}
	return values
}

func referenceGatedDeltaNetStep(ctx ml.Context, q, k, v, gate, beta, state ml.Tensor) (ml.Tensor, ml.Tensor) {
	headDim := v.Dim(0)
	numHeads := v.Dim(1)
	numSeqs := v.Dim(3)

	gateExp := gate.Exp(ctx)
	if gate.Dim(0) == headDim {
		gateExp = gateExp.Reshape(ctx, 1, headDim, gate.Dim(1), numSeqs)
	} else {
		gateExp = gateExp.Reshape(ctx, 1, 1, gate.Dim(1), numSeqs)
	}
	state = state.Mul(ctx, gateExp)

	kKey := k.Reshape(ctx, 1, headDim, k.Dim(1), numSeqs)
	kvMem := state.Mul(ctx, kKey)
	kvMem = kvMem.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).SumRows(ctx)
	kvMem = kvMem.Permute(ctx, 1, 0, 2, 3)

	vValue := v.Reshape(ctx, headDim, 1, numHeads, numSeqs)
	betaScale := beta.Reshape(ctx, 1, 1, beta.Dim(1), numSeqs)
	delta := vValue.Sub(ctx, kvMem).Mul(ctx, betaScale)

	kUpdate := kKey.Repeat4D(ctx, headDim, headDim, numHeads, numSeqs).Mul(ctx, delta)
	state = state.Add(ctx, kUpdate)

	qKey := q.Reshape(ctx, 1, headDim, q.Dim(1), numSeqs)
	output := state.Mul(ctx, qKey)
	output = output.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).SumRows(ctx)
	output = output.Permute(ctx, 1, 0, 2, 3).Reshape(ctx, headDim, numHeads, 1, numSeqs)
	output = output.Mul(ctx, ctx.FromFloats([]float32{float32(1.0 / math.Sqrt(float64(headDim)))}, 1, 1, 1, 1))

	return state, output
}

func gatedDeltaNetOutput(ctx ml.Context, result ml.Tensor, headDim, numHeads, numTokens, numSeqs int) ml.Tensor {
	elemSize := result.Stride(0)
	stride1 := elemSize * headDim
	stride2 := stride1 * numHeads
	stride3 := stride2 * numTokens

	return result.View(ctx,
		0,
		headDim, stride1,
		numHeads, stride2,
		numTokens, stride3,
		numSeqs,
	).Contiguous(ctx)
}

func gatedDeltaNetState(ctx ml.Context, result ml.Tensor, headDim, numHeads, numTokens, numSeqs int) ml.Tensor {
	elemSize := result.Stride(0)
	offset := elemSize * headDim * numHeads * numTokens * numSeqs
	stride1 := elemSize * headDim
	stride2 := stride1 * headDim
	stride3 := stride2 * numHeads

	return result.View(ctx,
		offset,
		headDim, stride1,
		headDim, stride2,
		numHeads, stride3,
		numSeqs,
	).Contiguous(ctx)
}

func assertCloseFloats(t *testing.T, name string, want, got []float32, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}

	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > tol {
			t.Fatalf("%s[%d] = %.8f, want %.8f (diff %.8f > %.8f)\nall got:  %v\nall want: %v",
				name, i, got[i], want[i], diff, tol, got, want)
		}
	}
}
