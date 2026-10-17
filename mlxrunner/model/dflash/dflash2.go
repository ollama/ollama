package dflash

import (
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

var _ model.BlockSelector = (*model2)(nil)

// model2 adds block-local dynamic convolutions and a predecessor-conditioned
// candidate selector to DFlash. Target features and context caches are shared.
// Weight fields stay exported: loading keeps only the arrays mlx.Collect
// reaches, and Collect skips unexported fields.
type model2 struct {
	*Model
	Selector *candidateSelector
}

func newModel2(root *model.Root, target model.Model) (model.DraftModel, error) {
	m, err := newModel(root, target, false)
	if err != nil {
		return nil, err
	}
	if err := m.Config.validateDFlash2(); err != nil {
		return nil, err
	}
	if m.Causal == nil {
		// DFlash2 checkpoints give the direction at the top level, bidirectional by default.
		causal := m.isCausal != nil && *m.isCausal
		m.Causal = &causal
	}
	return &model2{Model: m}, nil
}

func (c *Config) validateDFlash2() error {
	if c.ConvGroupSize <= 0 || int(c.HiddenSize)%c.ConvGroupSize != 0 || c.ConvKernelSize <= 0 || c.ConvKernelSize > c.BlockSize {
		return fmt.Errorf("dflash2 requires a convolution group dividing hidden_size and a kernel size in [1, block_size]: conv_group_size %d, hidden_size %d, conv_kernel_size %d, block_size %d", c.ConvGroupSize, c.HiddenSize, c.ConvKernelSize, c.BlockSize)
	}
	// Projection dimensions must fit the MLX shape representation.
	if int64(c.ConvKernelSize) > math.MaxInt32/(2*int64(c.HiddenSize)/int64(c.ConvGroupSize)) {
		return fmt.Errorf("dflash2 convolution projection is too large: conv_kernel_size %d, hidden_size %d, conv_group_size %d", c.ConvKernelSize, c.HiddenSize, c.ConvGroupSize)
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("dflash2 requires a positive vocab_size, got %d", c.VocabSize)
	}
	if maxTopK := min(int(c.VocabSize), maxSelectorTopK); c.SelectorRank <= 0 || c.SelectorRank > math.MaxInt32 || c.SelectorTopK <= 0 || c.SelectorTopK > maxTopK {
		return fmt.Errorf("dflash2 requires a positive selector rank and top_k in [1, %d]: selector_rank %d, selector_top_k %d", maxTopK, c.SelectorRank, c.SelectorTopK)
	}
	return nil
}

func (m *model2) LoadWeights(tensors map[string]*mlx.Array) error {
	if err := m.Model.LoadWeights(tensors); err != nil {
		return err
	}
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	for i, layer := range m.Layers {
		for name, dst := range map[string]**convolution{"attention_conv": &layer.AttentionConv, "mlp_conv": &layer.MLPConv} {
			prefix := fmt.Sprintf("%slayers.%d.%s", m.tensorPrefix, i, name)
			base := tensors[prefix+".base_kernel"]
			if base == nil || !slices.Equal(base.Dims(), []int{2, m.ConvKernelSize, int(m.HiddenSize)}) {
				return fmt.Errorf("dflash2 %s.base_kernel must have shape [2, %d, %d]", prefix, m.ConvKernelSize, m.HiddenSize)
			}
			projection, err := loadProjection(linears, tensors, prefix+".kernel_projection", int(m.HiddenSize), 2*m.ConvKernelSize*int(m.HiddenSize)/m.ConvGroupSize)
			if err != nil {
				return err
			}
			*dst = &convolution{Base: base, Projection: projection, Group: m.ConvGroupSize, Kernel: m.ConvKernelSize}
		}
	}
	prefix := m.tensorPrefix + "candidate_selector."
	pred, succ := tensors[prefix+"predecessor_codebook"], tensors[prefix+"successor_codebook"]
	for name, weight := range map[string]*mlx.Array{"predecessor_codebook": pred, "successor_codebook": succ} {
		if weight == nil || !slices.Equal(weight.Dims(), []int{int(m.VocabSize), m.SelectorRank}) {
			return fmt.Errorf("dflash2 %s%s must have shape [%d, %d]", prefix, name, m.VocabSize, m.SelectorRank)
		}
	}
	projection, err := loadProjection(linears, tensors, prefix+"hidden_projection", int(m.HiddenSize), m.SelectorRank)
	if err != nil {
		return err
	}
	m.Selector = &candidateSelector{Projection: projection, Predecessor: pred, Successor: succ, TopK: m.SelectorTopK}
	return nil
}

func loadProjection(linears model.LinearFactory, tensors map[string]*mlx.Array, path string, input, output int) (nn.LinearLayer, error) {
	w := tensors[path+".weight"]
	if w == nil {
		return nil, fmt.Errorf("dflash2 %s: missing projection weight", path)
	}
	if w.NumDims() != 2 || w.Dim(0) != output {
		return nil, fmt.Errorf("dflash2 %s: projection shape %v, want %d output rows", path, w.Dims(), output)
	}
	if scales := tensors[path+".weight_scale"]; scales != nil && (scales.NumDims() != 2 || scales.Dim(0) != output || scales.Dim(1) == 0) {
		return nil, fmt.Errorf("dflash2 %s: scale shape %v, want %d rows", path, scales.Dims(), output)
	}
	projection := linears.Make(path)
	width := w.Dim(1)
	if q, ok := projection.(*nn.QuantizedLinear); ok {
		if q.Bits <= 0 || q.Bits > 32 || q.GroupSize <= 0 || q.Scales.Dim(1)*q.GroupSize != input {
			return nil, fmt.Errorf("dflash2 %s: quantization bits %d, group size %d and scale shape %v do not cover input width %d", path, q.Bits, q.GroupSize, q.Scales.Dims(), input)
		}
		width = width * 32 / q.Bits
	}
	if width != input {
		return nil, fmt.Errorf("dflash2 %s: input width %d, want %d", path, width, input)
	}
	return projection, nil
}

type convolution struct {
	Base       *mlx.Array
	Projection nn.LinearLayer
	Group      int
	Kernel     int
}

// prepare convolves x with the input taps and returns the dynamic taps
// [B, L, 2, taps, groups] for finish.
func (c *convolution) prepare(x *mlx.Array) (*mlx.Array, *mlx.Array) {
	if c == nil {
		return x, nil
	}
	b, l, h := x.Dim(0), x.Dim(1), x.Dim(2)
	d := c.Projection.Forward(x).Reshape(b, l, 2, c.Kernel, h/c.Group)
	return c.convolveHalf(x, d, 0), d
}

func (c *convolution) finish(x, d *mlx.Array) *mlx.Array {
	if c == nil {
		return x
	}
	return c.convolveHalf(x, d, 1)
}

func (c *convolution) convolveHalf(x, d *mlx.Array, half int) *mlx.Array {
	taps := d.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(half, half+1), mlx.Slice(), mlx.Slice()).Squeeze(2)
	base := c.Base.Slice(mlx.Slice(half, half+1), mlx.Slice(), mlx.Slice()).Squeeze(0)
	return convolve(x, taps, base)
}

// convolve combines per-channel static taps and per-group dynamic taps.
// Left padding is zero; no convolution state carries across proposal blocks.
var convolve = mlx.Compile3("dflash_convolve", convolveGraph)

func convolveGraph(hidden, dynamic, base *mlx.Array) *mlx.Array {
	b, length, width := hidden.Dim(0), hidden.Dim(1), hidden.Dim(2)
	groups, taps := dynamic.Dim(3), base.Dim(0)
	groupSize := width / groups
	blocks := hidden.Reshape(b, length, groups, groupSize)
	output := mlx.Zeros(hidden.DType(), b, length, groups, groupSize)
	for offset := range taps {
		values := blocks
		if offset > 0 {
			shift := min(offset, length)
			values = mlx.PadConstant(blocks.Slice(mlx.Slice(), mlx.Slice(0, length-shift), mlx.Slice(), mlx.Slice()), []int{1}, []int{shift}, []int{0})
		}
		kernel := base.Slice(mlx.Slice(offset, offset+1), mlx.Slice()).Reshape(1, 1, groups, groupSize).AsType(hidden.DType())
		varying := dynamic.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(offset, offset+1), mlx.Slice()).Reshape(b, length, groups, 1)
		// Separate additions preserve the reference's BF16 rounding boundaries.
		output = output.Add(kernel.Multiply(values))
		output = output.Add(varying.Multiply(values))
	}
	return output.Reshape(b, length, width)
}

type candidateSelector struct {
	Projection  nn.LinearLayer
	Predecessor *mlx.Array
	Successor   *mlx.Array
	TopK        int
}

// Select implements model.BlockSelector.
func (m *model2) Select(hidden, logits, anchor *mlx.Array, choose func(ids, scores *mlx.Array) (tokens, probs *mlx.Array)) (tokens, ids, probs *mlx.Array) {
	s := m.Selector
	b, length := logits.Dim(0), logits.Dim(1)
	k := s.TopK
	ids = logits.Negative().ArgpartitionAxis(k-1, -1).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, k)).AsType(mlx.DTypeInt32)
	unary := logits.TakeAlongAxis(ids, -1)
	gates := s.Projection.Forward(hidden)

	// A row's predecessor is the anchor or one of the previous row's candidates,
	// so every (predecessor, candidate) pair is scored and sampled in one pass
	// and the chosen path is walked afterward. Row 0's slots all hold the anchor.
	previous := mlx.BroadcastTo(anchor.Reshape(b, 1, 1).AsType(mlx.DTypeInt32), int32(b), 1, int32(k))
	if length > 1 {
		previous = previous.Concatenate(1, ids.Slice(mlx.Slice(), mlx.Slice(0, length-1), mlx.Slice()))
	}
	scores := selectorScores(s.Predecessor, s.Successor, previous, ids, gates, unary)[0]
	candidates := mlx.BroadcastTo(ids.ExpandDims(2), int32(b), int32(length), int32(k), int32(k))
	next, q := choose(candidates.Reshape(b*length*k, k), scores.Reshape(b*length*k, k))
	next, q = next.Reshape(b, length, k), q.Reshape(b, length, k, k)
	// The sampled candidate's place in its row is the next row's predecessor slot.
	slots := next.ExpandDims(-1).Equal(ids.ExpandDims(2)).AsType(mlx.DTypeInt32).Argmax(-1, false).AsType(mlx.DTypeInt32)

	slot := mlx.Zeros(mlx.DTypeInt32, b, 1)
	path, rows := make([]*mlx.Array, length), make([]*mlx.Array, length)
	for pos := range length {
		at := mlx.Slice(pos, pos+1)
		path[pos] = next.Slice(mlx.Slice(), at, mlx.Slice()).Reshape(b, k).TakeAlongAxis(slot, -1)
		rows[pos] = q.Slice(mlx.Slice(), at, mlx.Slice(), mlx.Slice()).Reshape(b, k, k).TakeAlongAxis(slot.Reshape(b, 1, 1), 1)
		slot = slots.Slice(mlx.Slice(), at, mlx.Slice()).Reshape(b, k).TakeAlongAxis(slot, -1)
	}
	return mlx.Concatenate(path, 1), ids, mlx.Concatenate(rows, 1)
}

var selectorScores = mlx.Compile("dflash_selector_scores", selectorScoresGraph)

// selectorScoresGraph scores candidates [B,L,K] after predecessors [B,L,P] as
// [B,L,P,K], each pair as the reference's unary plus gated edge term.
func selectorScoresGraph(in ...*mlx.Array) []*mlx.Array {
	predecessor, successor, previous, candidates, gates, unary := in[0], in[1], in[2], in[3], in[4], in[5]
	pred := predecessor.TakeAxis(previous, 0).ExpandDims(3)
	succ := successor.TakeAxis(candidates, 0).ExpandDims(2)
	gate := gates.ExpandDims(2).ExpandDims(2)
	edges := pred.Multiply(gate).Multiply(succ).SumAxis(-1, false)
	return []*mlx.Array{unary.ExpandDims(2).Add(edges)}
}
