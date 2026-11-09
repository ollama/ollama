package mlx

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// gates stay in (0,1]: both paths treat them as decay factors, and negative
// values diverge at the kernel's log clamp.
type fastGatedDeltaInputs struct {
	q, k, v, gates, beta, state *Array
}

func fastGatedDeltaPatternInputs(B, T, Hk, Dk, Hv, Dv int, dtype DType) fastGatedDeltaInputs {
	return fastGatedDeltaInputs{
		q:     patternArray(dtype, []int{B, T, Hk, Dk}, 0.01, 0.003, 37, 257),
		k:     patternArray(dtype, []int{B, T, Hk, Dk}, -0.02, 0.004, 11, 101),
		v:     patternArray(dtype, []int{B, T, Hv, Dv}, 0.05, 0.002, 23, 61),
		gates: patternArray(dtype, []int{B, T, Hv}, 0.7, 0.04, 1, 5),
		beta:  patternArray(dtype, []int{B, T, Hv}, 0.8, 0.05, 3, 7),
		state: Zeros(DTypeFloat32, B, Hv, Dv, Dk),
	}
}

// The supported geometries cover MLX's sequential (T<16) and chunked
// (T>=16) kernels; the final geometry takes the graph fallback. Float32 uses
// upstream's 1e-4 kernel tolerance, while BF16 allows one coarse output step.
func TestFastGatedDeltaUpdateMatchesGraph(t *testing.T) {
	cases := []struct {
		Hk, Dk, Hv, Dv int
		Ts             []int
		dtype          DType
		tolerance      float64
	}{
		{Hk: 16, Dk: 128, Hv: 32, Dv: 128, Ts: []int{1, 3, 20}, dtype: DTypeFloat32, tolerance: 1e-4},
		// Qwen3.8-27B's BF16 geometry, including batch-1 decode/MTP and
		// prompt lengths on both sides of MLX's chunking threshold.
		{Hk: 16, Dk: 128, Hv: 48, Dv: 128, Ts: []int{1, 11, 16, 33, 127}, dtype: DTypeBFloat16, tolerance: 0.1},
		{Hk: 4, Dk: 64, Hv: 8, Dv: 32, Ts: []int{1, 20}, dtype: DTypeFloat32, tolerance: 1e-4},
	}
	for _, c := range cases {
		for _, T := range c.Ts {
			for _, B := range []int{1, 2} {
				name := fmt.Sprintf("hk%d_dk%d_hv%d_dv%d_t%d_b%d", c.Hk, c.Dk, c.Hv, c.Dv, T, B)
				t.Run(name, func(t *testing.T) {
					withMLXThread(t, func(t *mlxthreadtest.T) {
						in := fastGatedDeltaPatternInputs(B, T, c.Hk, c.Dk, c.Hv, c.Dv, c.dtype)

						refY, refState := gatedDeltaRecurrenceGraph(in.q, in.k, in.v, in.gates, in.beta, in.state)
						y, state := fastGatedDeltaUpdate(in.q, in.k, in.v, in.gates, in.beta, in.state, nil)

						if got, want := y.Dims(), []int{B, T, c.Hv, c.Dv}; !slices.Equal(got, want) {
							t.Fatalf("%s: y dims %v, want %v", name, got, want)
						}
						if got, want := state.Dims(), []int{B, c.Hv, c.Dv, c.Dk}; !slices.Equal(got, want) {
							t.Fatalf("%s: state dims %v, want %v", name, got, want)
						}
						if y.DType() != c.dtype || state.DType() != DTypeFloat32 {
							t.Fatalf("%s: dtypes y=%v state=%v, want %v/float32", name, y.DType(), state.DType(), c.dtype)
						}

						var failures []error
						failures = appendArrayCloseError(failures, name+"/y", y.AsType(DTypeFloat32), refY.AsType(DTypeFloat32), c.tolerance)
						failures = appendArrayCloseError(failures, name+"/state", state, refState, c.tolerance)
						if len(failures) > 0 {
							t.Fatal(failures[0])
						}
					})
				})
			}
		}
	}
}

func TestFastGatedDeltaUpdateNilState(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		in := fastGatedDeltaPatternInputs(1, 3, 16, 128, 32, 128, DTypeFloat32)
		yExplicit, stateExplicit := fastGatedDeltaUpdate(in.q, in.k, in.v, in.gates, in.beta, in.state, nil)
		yNil, stateNil := fastGatedDeltaUpdate(in.q, in.k, in.v, in.gates, in.beta, nil, nil)
		if err := requireExact("y", yNil, yExplicit); err != nil {
			t.Fatal(err)
		}
		if err := requireExact("state", stateNil, stateExplicit); err != nil {
			t.Fatal(err)
		}
	})
}

// A mask forces the fallback path: a padded position leaves the state
// unchanged and emits zero output, so the final state must match a scan
// over just the real tokens.
func TestFastGatedDeltaUpdateMask(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		B, T, Hk, Dk, Hv, Dv := 1, 3, 16, 128, 32, 128
		in := fastGatedDeltaPatternInputs(B, T, Hk, Dk, Hv, Dv, DTypeFloat32)
		mask := FromValues([]bool{true, false, true}, B, T)

		y, state := fastGatedDeltaUpdate(in.q, in.k, in.v, in.gates, in.beta, in.state, mask)

		yPad := SliceStartStop(y, []int32{0, 1, 0, 0}, []int32{int32(B), 2, int32(Hv), int32(Dv)})
		if err := requireExact("masked y", yPad, Zeros(DTypeFloat32, B, 1, Hv, Dv)); err != nil {
			t.Fatal(err)
		}

		pick := func(a *Array, ndim int, t int) *Array {
			start := make([]int32, ndim)
			stop := make([]int32, ndim)
			for d := range stop {
				stop[d] = int32(a.Dim(d))
			}
			start[1], stop[1] = int32(t), int32(t+1)
			return SliceStartStop(a, start, stop)
		}
		cat := func(a *Array, ndim int) *Array {
			return Concatenate([]*Array{pick(a, ndim, 0), pick(a, ndim, 2)}, 1)
		}
		skipState := Zeros(DTypeFloat32, B, Hv, Dv, Dk)
		_, skipFinal := fastGatedDeltaUpdate(
			cat(in.q, 4), cat(in.k, 4), cat(in.v, 4), cat(in.gates, 3), cat(in.beta, 3), skipState, nil)
		// The masked run takes the fallback while the skip scan runs the kernel,
		// so equivalence holds to tolerance, not bit-for-bit.
		if failures := appendArrayCloseError(nil, "state", state, skipFinal, 1e-4); len(failures) > 0 {
			t.Fatal(failures[0])
		}
	})
}
