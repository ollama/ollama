package qwen4_exp

import "github.com/ollama/ollama/x/mlxrunner/mlx"

type hyperConnectionState struct {
	residual *mlx.Array
	normed   *mlx.Array
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
	mixHidden := mlx.SiLU(mlx.DivScalar(h.InputMixDown.Forward(normed), float32(cfg.HCCount)))
	mix := h.InputMixUp.Forward(mixHidden)
	mix = mlx.Reshape(mix, B, L, cfg.HCCount, cfg.HiddenSize)
	streams := mlx.Reshape(normed, B, L, cfg.HCCount, cfg.HiddenSize)
	branch := mlx.DivScalar(mlx.Sum(mlx.Mul(streams, mlx.Sigmoid(mix)), 2, false), float32(cfg.HCCount))
	return branch, hyperConnectionState{residual: residual, normed: normed}
}

// Inject broadcasts one branch result back into the residual streams. The
// public HC post-connection uses 2*sigmoid so an untrained coefficient is one;
// block_inject_weight has exactly one dynamic coefficient per stream.
func (h *hyperConnection) Inject(state hyperConnectionState, branch *mlx.Array, cfg *Config) *mlx.Array {
	dims := branch.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	weight := mlx.MulScalar(mlx.Sigmoid(mlx.DivScalar(h.BlockInject.Forward(state.normed), float32(cfg.HCCount))), 2)
	weight = mlx.ExpandDims(weight, -1)
	branch = mlx.ExpandDims(branch, -2)
	update := mlx.Mul(weight, branch)
	residual := mlx.Reshape(state.residual, B, L, cfg.HCCount, cfg.HiddenSize)
	return mlx.Reshape(mlx.Add(residual, update), B, L, cfg.HCCount*cfg.HiddenSize)
}

// Reduce applies the trained input mixer as the final stream reduction. The
// final checkpoint mixer has no block injection tensor, matching this role.
func (h *hyperConnection) Reduce(residual *mlx.Array, cfg *Config) *mlx.Array {
	branch, _ := h.Prepare(residual, cfg)
	return branch
}
