package qwen4_exp

import "github.com/ollama/ollama/mlx"

var (
	hyperConnectionSiLU = mlx.Compile2(
		"Qwen4HyperConnectionSiLU",
		func(x, count *mlx.Array) *mlx.Array {
			return mlx.SiLU(mlx.Div(x, count))
		},
		mlx.Shapeless(),
	)
	hyperConnectionMix = mlx.Compile3(
		"Qwen4HyperConnectionMix",
		func(streams, mix, count *mlx.Array) *mlx.Array {
			return mlx.Div(mlx.Sum(mlx.Mul(streams, mlx.Sigmoid(mix)), 2, false), count)
		},
		mlx.Shapeless(),
	)
	hyperConnectionInject = mlx.Compile(
		"Qwen4HyperConnectionInject",
		func(in ...*mlx.Array) []*mlx.Array {
			residual, branch, logits, count := in[0], in[1], in[2], in[3]
			weight := mlx.Sigmoid(mlx.Div(logits, count))
			two := mlx.FromValue(float32(2)).AsType(weight.DType())
			weight = mlx.Mul(weight, two)
			update := mlx.Mul(mlx.ExpandDims(weight, -1), mlx.ExpandDims(branch, -2))
			return []*mlx.Array{mlx.Add(residual, update)}
		},
		mlx.Shapeless(),
	)
)

type hyperConnectionState struct {
	residual *mlx.Array
	normed   *mlx.Array
	inject   *mlx.Array
}

func expandStreams(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	expanded := mlx.Tile(mlx.ExpandDims(x, -2), []int32{1, 1, cfg.HCCount, 1})
	return mlx.Reshape(expanded, B, L, cfg.HCCount*cfg.HiddenSize)
}

// Prepare reduces the residual streams into one branch input and preserves
// the state needed to inject the branch output.
func (h *hyperConnection) Prepare(residual *mlx.Array, cfg *Config) (*mlx.Array, hyperConnectionState) {
	dims := residual.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	normed := h.Norm.Forward(residual, cfg.RMSNormEps)
	var down, inject *mlx.Array
	if h.PackedInput != nil {
		packed := h.PackedInput.Forward(normed)
		down = mlx.SliceStartStop(packed,
			[]int32{0, 0, 0},
			[]int32{B, L, h.MixDownDim},
		)
		inject = mlx.SliceStartStop(packed,
			[]int32{0, 0, h.MixDownDim},
			[]int32{B, L, int32(packed.Dim(2))},
		)
	} else {
		down = h.InputMixDown.Forward(normed)
	}
	count := mlx.NewScalarArray(float32(cfg.HCCount)).AsType(down.DType())
	mixHidden := hyperConnectionSiLU(down, count)
	mix := h.InputMixUp.Forward(mixHidden)
	mix = mlx.Reshape(mix, B, L, cfg.HCCount, cfg.HiddenSize)
	streams := mlx.Reshape(normed, B, L, cfg.HCCount, cfg.HiddenSize)
	count = mlx.NewScalarArray(float32(cfg.HCCount)).AsType(streams.DType())
	branch := hyperConnectionMix(streams, mix, count)
	return branch, hyperConnectionState{residual: residual, normed: normed, inject: inject}
}

// Inject broadcasts one branch result back into the residual streams. The
// public HC post-connection uses 2*sigmoid so an untrained coefficient is one;
// block_inject_weight has exactly one dynamic coefficient per stream.
func (h *hyperConnection) Inject(state hyperConnectionState, branch *mlx.Array, cfg *Config) *mlx.Array {
	dims := branch.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	logits := state.inject
	if logits == nil {
		logits = h.BlockInject.Forward(state.normed)
	}
	count := mlx.NewScalarArray(float32(cfg.HCCount)).AsType(logits.DType())
	residual := mlx.Reshape(state.residual, B, L, cfg.HCCount, cfg.HiddenSize)
	out := hyperConnectionInject(residual, branch, logits, count)[0]
	return mlx.Reshape(out, B, L, cfg.HCCount*cfg.HiddenSize)
}

// Reduce applies the trained input mixer as the final stream reduction. The
// final checkpoint mixer has no block injection tensor, matching this role.
func (h *hyperConnection) Reduce(residual *mlx.Array, cfg *Config) *mlx.Array {
	branch, _ := h.Prepare(residual, cfg)
	return branch
}
