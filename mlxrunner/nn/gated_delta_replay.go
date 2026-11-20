package nn

import (
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
)

// GatedDeltaReplay borrows the inputs for one forward. A cache retaining it
// beyond that forward must clone the array handles into its own held scope.
type GatedDeltaReplay struct {
	Packed, BA, DtBias, AExp, Prior *mlx.Array
	ValidLengths                    []int32
	PreciseDecay                    bool
}

// GatedDeltaForReplay produces the normal output and end state without building
// interior snapshots. The caller must resolve the selected states before reuse.
func GatedDeltaForReplay(b *batch.Batch, packed, ba, dtBias, aExp *mlx.Array, opts ...RecurrentOption) (*mlx.Array, *mlx.Array, GatedDeltaReplay) {
	cfg := resolveRecurrentConfig(opts)
	prior := cfg.deltaState
	if cfg.history != nil {
		prior = cfg.history.DeltaState()
	}
	inputs := GatedDeltaReplay{packed, ba, dtBias, aExp, prior, slices.Clone(b.SeqQueryLens), cfg.preciseDecay}
	y, end := inputs.run(paddingMask(b, int32(packed.Dim(1))))
	return y, end, inputs
}

// Replay builds one existing gated-delta operation with a neutral suffix for
// each row. It does not evaluate it or change the borrowed inputs.
func (r GatedDeltaReplay) Replay(prefixes []int32) *mlx.Array {
	B, L := r.Packed.Dim(0), r.Packed.Dim(1)
	if len(prefixes) != B || len(r.ValidLengths) != B {
		panic("gated delta replay: prefix rows do not match inputs")
	}
	lengths := make([]int32, B)
	for i, n := range prefixes {
		if n < 0 || int(n) > L {
			panic("gated delta replay: prefix is outside the forward")
		}
		lengths[i] = min(n, r.ValidLengths[i])
	}
	mask := paddingMask(&batch.Batch{SeqQueryLens: lengths}, int32(L))
	_, state := r.run(mask)
	return state
}

func (r GatedDeltaReplay) run(mask *mlx.Array) (*mlx.Array, *mlx.Array) {
	run := mlx.GatedDelta
	if r.PreciseDecay {
		run = mlx.GatedDeltaPrecise
	}
	y, state, _ := run(r.Packed, r.BA, r.DtBias, r.AExp, r.Prior, mask, false)
	return y, state
}
