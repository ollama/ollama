package qwen3next

import (
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

func TestDeltaNetAutoregressiveFusedMatchesExplicit(t *testing.T) {
	fusedOut, fusedState := runDeltaNetAutoregressiveCase(t, true)
	explicitOut, explicitState := runDeltaNetAutoregressiveCase(t, false)

	assertCloseFloat32s(t, "output", explicitOut, fusedOut, 1e-5)
	assertCloseFloat32s(t, "state", explicitState, fusedState, 1e-5)
}

func TestDeltaNetAutoregressiveFusedMaintainsStateAcrossSteps(t *testing.T) {
	cases := []struct {
		name      string
		headDim   int
		numKHeads int
		numVHeads int
	}{
		{
			name:      "production grouped heads",
			headDim:   128,
			numKHeads: 2,
			numVHeads: 16,
		},
		{
			name:      "single key head",
			headDim:   16,
			numKHeads: 1,
			numVHeads: 4,
		},
		{
			name:      "equal heads",
			headDim:   16,
			numKHeads: 4,
			numVHeads: 4,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			fusedOut, fusedState := runDeltaNetAutoregressiveSequenceCase(t, true, tt.headDim, tt.numKHeads, tt.numVHeads)
			explicitOut, explicitState := runDeltaNetAutoregressiveSequenceCase(t, false, tt.headDim, tt.numKHeads, tt.numVHeads)

			assertCloseFloat32s(t, "output", explicitOut, fusedOut, 2e-4)
			assertCloseFloat32s(t, "state", explicitState, fusedState, 2e-4)
		})
	}
}

func runDeltaNetAutoregressiveCase(t *testing.T, fused bool) ([]float32, []float32) {
	t.Helper()

	backend := setupDeltaNetBackend(t)
	defer backend.Close()

	const (
		headDim     = 2
		numKHeads   = 2
		numVHeads   = 4
		nSeqTokens  = 1
		nSeqs       = 1
		deltaSize   = headDim * headDim * numVHeads
		recurrentIx = 0
	)

	opts := &Options{
		eps:            1e-6,
		vHeadReordered: true,
	}

	seqBatch := input.Batch{
		Positions: []int32{0},
		Sequences: []int{0},
	}

	ctx := backend.NewContext().Input()
	defer ctx.Close()

	fusedCache := NewHybridCache(nil, 0, 0, deltaSize)
	fusedCache.Init(backend, ml.DTypeF32, 1, 8, 1)
	defer fusedCache.Close()
	if err := fusedCache.StartForward(ctx, seqBatch, false); err != nil {
		t.Fatal(err)
	}

	explicitCache := NewHybridCache(nil, 0, 0, deltaSize)
	explicitCache.Init(backend, ml.DTypeF32, 1, 8, 1)
	defer explicitCache.Close()
	if err := explicitCache.StartForward(ctx, seqBatch, false); err != nil {
		t.Fatal(err)
	}

	q := ctx.FromFloats([]float32{
		0.3, -0.4,
		0.6, 0.2,
	}, headDim, numKHeads, nSeqTokens, nSeqs)
	k := ctx.FromFloats([]float32{
		0.2, 0.7,
		-0.5, 0.4,
	}, headDim, numKHeads, nSeqTokens, nSeqs)
	v := ctx.FromFloats([]float32{
		1.0, -0.5,
		0.25, 0.8,
		-0.1, 0.9,
		0.7, -0.3,
	}, headDim, numVHeads, nSeqTokens, nSeqs)
	gate := ctx.FromFloats([]float32{
		-0.2, 0.1,
		0.0, -0.4,
	}, 1, numVHeads, nSeqTokens, nSeqs)
	beta := ctx.FromFloats([]float32{
		0.3, 0.6, -0.2, 0.1,
	}, 1, numVHeads, nSeqTokens, nSeqs)
	state := ctx.Zeros(ml.DTypeF32, headDim, headDim*numVHeads, nSeqTokens, nSeqs)

	explicitQ, explicitK := repeatQKToVHeads(ctx, q, k, headDim, numKHeads, numVHeads, nSeqTokens, nSeqs, opts.vHeadReordered)

	var gdn GatedDeltaNet
	var out ml.Tensor
	cache := explicitCache
	if fused {
		out = gdn.deltaNetAutoregressiveFused(ctx, q, k, v, gate, beta, state, opts, recurrentIx, fusedCache)
		cache = fusedCache
	} else {
		out = gdn.deltaNetAutoregressive(ctx, explicitQ, explicitK, v, gate, beta, state, opts, recurrentIx, explicitCache)
	}

	ctx.Forward(out).Compute(out)
	output := out.Floats()

	checkCtx := backend.NewContext().Input()
	defer checkCtx.Close()
	if err := cache.StartForward(checkCtx, seqBatch, false); err != nil {
		t.Fatal(err)
	}
	stateOut, err := cache.DeltaState(checkCtx, recurrentIx, headDim, numVHeads)
	if err != nil {
		t.Fatal(err)
	}
	checkCtx.Forward(stateOut).Compute(stateOut)
	return output, stateOut.Floats()
}

func runDeltaNetAutoregressiveSequenceCase(t *testing.T, fused bool, headDim, numKHeads, numVHeads int) ([]float32, []float32) {
	t.Helper()

	backend := setupDeltaNetBackend(t)
	defer backend.Close()

	const (
		nSeqTokens  = 1
		nSeqs       = 2
		steps       = 4
		recurrentIx = 0
	)

	deltaSize := headDim * headDim * numVHeads
	opts := &Options{
		eps:            1e-6,
		vHeadReordered: true,
	}

	cache := NewHybridCache(nil, 0, 0, deltaSize)
	cache.Init(backend, ml.DTypeF32, nSeqs, 16, nSeqs)
	defer cache.Close()

	seqBatch := input.Batch{
		Positions: []int32{0, 0},
		Sequences: []int{0, 1},
	}

	seedCtx := backend.NewContext().Input()
	if err := cache.StartForward(seedCtx, seqBatch, false); err != nil {
		t.Fatal(err)
	}
	initialState := seedCtx.FromFloats(deterministicFloats(deltaSize*nSeqs, 0.05), headDim, headDim*numVHeads, nSeqs)
	cache.UpdateDeltaState(seedCtx, recurrentIx, initialState)
	seedCtx.Compute()
	seedCtx.Close()

	var outputs []float32
	var gdn GatedDeltaNet
	for step := range steps {
		ctx := backend.NewContext().Input()
		seqBatch.Positions = []int32{int32(step + 1), int32(step + 1)}
		if err := cache.StartForward(ctx, seqBatch, false); err != nil {
			t.Fatal(err)
		}

		state, err := cache.DeltaState(ctx, recurrentIx, headDim, numVHeads)
		if err != nil {
			t.Fatal(err)
		}
		state = state.Reshape(ctx, headDim, headDim*numVHeads, nSeqTokens, nSeqs)

		q := ctx.FromFloats(deterministicFloats(headDim*numKHeads*nSeqs, float32(step)+0.10), headDim, numKHeads, nSeqTokens, nSeqs)
		k := ctx.FromFloats(deterministicFloats(headDim*numKHeads*nSeqs, float32(step)+0.20), headDim, numKHeads, nSeqTokens, nSeqs)
		v := ctx.FromFloats(deterministicFloats(headDim*numVHeads*nSeqs, float32(step)+0.30), headDim, numVHeads, nSeqTokens, nSeqs)
		gate := ctx.FromFloats(deterministicFloats(numVHeads*nSeqs, float32(step)+0.40), 1, numVHeads, nSeqTokens, nSeqs)
		beta := ctx.FromFloats(deterministicFloats(numVHeads*nSeqs, float32(step)+0.50), 1, numVHeads, nSeqTokens, nSeqs)

		var out ml.Tensor
		if fused {
			out = gdn.deltaNetAutoregressiveFused(ctx, q, k, v, gate, beta, state, opts, recurrentIx, cache)
		} else {
			q, k = repeatQKToVHeads(ctx, q, k, headDim, numKHeads, numVHeads, nSeqTokens, nSeqs, opts.vHeadReordered)
			out = gdn.deltaNetAutoregressive(ctx, q, k, v, gate, beta, state, opts, recurrentIx, cache)
		}

		ctx.Forward(out).Compute(out)
		outputs = append(outputs, out.Floats()...)
		ctx.Close()
	}

	checkCtx := backend.NewContext().Input()
	seqBatch.Positions = []int32{steps + 1, steps + 1}
	if err := cache.StartForward(checkCtx, seqBatch, false); err != nil {
		t.Fatal(err)
	}
	stateOut, err := cache.DeltaState(checkCtx, recurrentIx, headDim, numVHeads)
	if err != nil {
		t.Fatal(err)
	}
	checkCtx.Forward(stateOut).Compute(stateOut)
	state := stateOut.Floats()
	checkCtx.Close()

	return outputs, state
}

func TestUseFusedDeltaNetGuard(t *testing.T) {
	cases := []struct {
		name           string
		supported      bool
		nSeqTokens     int
		numKHeads      int
		numVHeads      int
		vHeadReordered bool
		want           bool
	}{
		{
			name:       "unsupported backend",
			supported:  false,
			nSeqTokens: 1,
			numKHeads:  4,
			numVHeads:  4,
			want:       false,
		},
		{
			name:       "equal heads",
			supported:  true,
			nSeqTokens: 1,
			numKHeads:  4,
			numVHeads:  4,
			want:       true,
		},
		{
			name:           "single key head reordered",
			supported:      true,
			nSeqTokens:     1,
			numKHeads:      1,
			numVHeads:      4,
			vHeadReordered: true,
			want:           true,
		},
		{
			name:           "grouped key heads reordered",
			supported:      true,
			nSeqTokens:     1,
			numKHeads:      2,
			numVHeads:      4,
			vHeadReordered: true,
			want:           true,
		},
		{
			name:           "chunked path",
			supported:      true,
			nSeqTokens:     2,
			numKHeads:      1,
			numVHeads:      4,
			vHeadReordered: true,
			want:           false,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got := useFusedDeltaNet(tt.supported, tt.nSeqTokens, tt.numKHeads, tt.numVHeads, tt.vHeadReordered)
			if got != tt.want {
				t.Fatalf("useFusedDeltaNet() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRepeatQKToVHeadsReordered(t *testing.T) {
	backend := setupDeltaNetBackend(t)
	defer backend.Close()

	ctx := backend.NewContext().Input()
	defer ctx.Close()

	q := ctx.FromFloats([]float32{0.3, -0.4}, 2, 1, 1, 1)
	k := ctx.FromFloats([]float32{0.2, 0.7}, 2, 1, 1, 1)
	q, k = repeatQKToVHeads(ctx, q, k, 2, 1, 4, 1, 1, true)

	ctx.Forward(q, k).Compute(q, k)

	assertCloseFloat32s(t, "q", []float32{0.3, -0.4, 0.3, -0.4, 0.3, -0.4, 0.3, -0.4}, q.Floats(), 1e-6)
	assertCloseFloat32s(t, "k", []float32{0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7}, k.Floats(), 1e-6)
}

func setupDeltaNetBackend(tb testing.TB) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{
		"general.architecture": "test",
		"block_count":          uint32(1),
	}, nil); err != nil {
		tb.Fatal(err)
	}

	backend, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Fatal(err)
	}

	return backend
}

func deterministicFloats(n int, seed float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		x := math.Sin(float64(seed)+float64(i+1)*0.113) * 0.25
		values[i] = float32(x)
	}
	return values
}

func assertCloseFloat32s(t *testing.T, name string, want, got []float32, tol float64) {
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
