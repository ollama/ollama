package dflash

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

// Model2 is the path-selecting DFlash2 variant. Embedding Model keeps the
// existing DFlash implementation and runtime interface unchanged.
type Model2 struct{ *Model }

var _ base.PathBlockDraft = (*Model2)(nil)

type GroupedDynamicConv struct {
	BaseKernel       *mlx.Array
	KernelProjection nn.LinearLayer
	KernelSize       int32
	GroupSize        int32
}

func loadDynamicConv(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, cfg *Config) *GroupedDynamicConv {
	baseKernel := tensors[prefix+".base_kernel"]
	projection := linears.Make(prefix + ".kernel_projection")
	if baseKernel == nil || projection == nil {
		return nil
	}
	return &GroupedDynamicConv{
		BaseKernel:       baseKernel,
		KernelProjection: projection,
		KernelSize:       cfg.ConvKernelSize,
		GroupSize:        cfg.ConvGroupSize,
	}
}

func (c *GroupedDynamicConv) Prepare(hidden *mlx.Array) (*mlx.Array, *mlx.Array) {
	B, L, H := hidden.Dim(0), hidden.Dim(1), hidden.Dim(2)
	groups := H / int(c.GroupSize)
	dynamic := c.KernelProjection.Forward(hidden).Reshape(B, L, 2, int(c.KernelSize), groups)
	first := dynamic.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()).Squeeze(2)
	second := dynamic.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, 2), mlx.Slice(), mlx.Slice()).Squeeze(2)
	base := c.BaseKernel
	return c.convolve(hidden, first, base.Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()).Squeeze(0)), second
}

func (c *GroupedDynamicConv) Finish(hidden, dynamic *mlx.Array) *mlx.Array {
	base := c.BaseKernel.Slice(mlx.Slice(1, 2), mlx.Slice(), mlx.Slice()).Squeeze(0)
	return c.convolve(hidden, dynamic, base)
}

func (c *GroupedDynamicConv) convolve(hidden, dynamic, base *mlx.Array) *mlx.Array {
	B, L, H := hidden.Dim(0), hidden.Dim(1), hidden.Dim(2)
	groups, groupSize := H/int(c.GroupSize), int(c.GroupSize)
	blocks := hidden.Reshape(B, L, groups, groupSize)
	out := mlx.Zeros(hidden.DType(), B, L, groups, groupSize)
	for offset := range int(c.KernelSize) {
		values := blocks
		if offset > 0 {
			zeros := mlx.Zeros(hidden.DType(), B, offset, groups, groupSize)
			values = zeros.Concatenate(1, blocks.Slice(mlx.Slice(), mlx.Slice(0, L-offset), mlx.Slice(), mlx.Slice()))
		}
		baseAt := base.Slice(mlx.Slice(offset, offset+1), mlx.Slice()).Reshape(1, 1, groups, groupSize)
		dynamicAt := dynamic.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(offset, offset+1), mlx.Slice()).Squeeze(2).ExpandDims(-1)
		out = out.Add(baseAt.Add(dynamicAt).Multiply(values))
	}
	return out.Reshape(B, L, H)
}

type CandidateSelector struct {
	Predecessor      nn.EmbeddingLayer
	Successor        nn.EmbeddingLayer
	HiddenProjection nn.LinearLayer
	TopK             int
}

func loadCandidateSelector(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string, m *Model) *CandidateSelector {
	pred := loadCodebook(tensors, prefix+"candidate_selector.predecessor_codebook", m)
	succ := loadCodebook(tensors, prefix+"candidate_selector.successor_codebook", m)
	projection := linears.Make(prefix + "candidate_selector.hidden_projection")
	if pred == nil || succ == nil || projection == nil {
		return nil
	}
	return &CandidateSelector{
		Predecessor:      pred,
		Successor:        succ,
		HiddenProjection: projection,
		TopK:             m.SelectorTopK,
	}
}

func loadCodebook(tensors map[string]*mlx.Array, path string, m *Model) nn.EmbeddingLayer {
	w := tensors[path]
	if w == nil {
		return nil
	}
	if scales := tensors[path+"_scale"]; scales != nil {
		groupSize, bits, mode := model.ResolveLinearQuantParams(
			m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant,
			path, w, scales,
		)
		return &nn.QuantizedEmbedding{
			Weight: w, Scales: scales, QBiases: tensors[path+"_qbias"],
			GroupSize: groupSize, Bits: bits, Mode: mode,
		}
	}
	return nn.NewEmbedding(w)
}

// CandidateLattice constructs the K-way token candidates and all KxK
// transition scores. The runtime samples one predecessor-conditioned row per
// position and retains that exact sparse distribution for rejection sampling.
func (m *Model2) CandidateLattice(hidden, anchor *mlx.Array) (ids, scores *mlx.Array) {
	logits := m.Unembed(hidden)
	k := m.Selector.TopK
	ids = logits.Negative().ArgpartitionAxis(k-1, -1).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, k)).AsType(mlx.DTypeInt32)
	unary := logits.TakeAlongAxis(ids, -1).AsType(mlx.DTypeFloat32)
	if m.OutputMultiplier != 1 {
		unary = mlx.MulScalar(unary, m.OutputMultiplier)
	}
	if m.FinalLogitSoftcap > 0 {
		cap := mlx.FromValue(m.FinalLogitSoftcap).AsType(unary.DType())
		unary = mlx.LogitSoftcap(unary, cap)
	}

	B, L := hidden.Dim(0), hidden.Dim(1)
	projected := m.Selector.HiddenProjection.Forward(hidden)
	predecessorIDs := mlx.Tile(anchor.Reshape(B, 1, 1), []int32{1, 1, int32(k)})
	if L > 1 {
		predecessorIDs = predecessorIDs.Concatenate(1, ids.Slice(mlx.Slice(), mlx.Slice(0, L-1), mlx.Slice()))
	}
	predecessors := m.Selector.Predecessor.Forward(predecessorIDs)
	successors := m.Selector.Successor.Forward(ids)
	edges := predecessors.Multiply(projected.ExpandDims(2)).ExpandDims(3).
		Multiply(successors.ExpandDims(2)).SumAxis(-1, false)
	return ids, edges.Add(unary.ExpandDims(2))
}
