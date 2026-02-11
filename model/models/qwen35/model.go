package qwen35

import (
	"bytes"
	"cmp"
	"fmt"
	"image"
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Options struct {
	hiddenSize  int
	numHeads    int
	numKVHeads  int
	keyLength   int
	valueLength int
	ropeDim     int

	eps                   float32
	ropeBase              float32
	ropeScale             float32
	ropeType              string
	originalContextLength int
	attentionScale        float64

	numExperts     int
	numExpertsUsed int
	normTopKProb   bool

	ssmDInner      int
	ssmDState      int
	ssmNGroup      int
	ssmDtRank      int
	convKernelSize int

	isRecurrent []bool

	masks *Masks

	mropeSections []int
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	if len(o.mropeSections) > 0 {
		return nn.RoPE(ctx, states, positions, o.headDim(), o.ropeBase, 1/float32(math.Sqrt(float64(o.ropeScale))),
			rope.WithInterleaveMRoPE(o.mropeSections),
		)
	}
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(1.),
			rope.WithAttentionFactor(attnFactor),
		)
	}
	ropeDim := cmp.Or(o.ropeDim, o.headDim())
	return nn.RoPE(ctx, states, positions, ropeDim, o.ropeBase, 1./o.ropeScale, opts...)
}

type Operator interface {
	Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error)
}

type MLP interface {
	Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor
}

type dense struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *dense) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ *Options) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type Layer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	AttentionPostNorm *nn.RMSNorm `gguf:"post_attention_norm,alt:attn_post_norm"`
	Operator          Operator

	MLP MLP
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenStates, positions, outputs ml.Tensor, cache *HybridCache, opts *Options) (ml.Tensor, error) {
	residual := hiddenStates

	hiddenStates = l.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)

	var err error
	hiddenStates, err = l.Operator.Forward(ctx, hiddenStates, positions, cache, opts)
	if err != nil {
		return nil, err
	}

	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	ffnResidual := hiddenStates

	hiddenStates = l.AttentionPostNorm.Forward(ctx, hiddenStates, opts.eps)

	hiddenStates = l.MLP.Forward(ctx, hiddenStates, opts)

	return hiddenStates.Add(ctx, ffnResidual), nil
}

const (
	tokenVision      int32 = 248056
	tokenVisionStart int32 = 248053
	tokenVisionEnd   int32 = 248054
)

type modelInput struct {
	position int32
	*input.Input
}

type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Layers []Layer `gguf:"blk"`

	*VisionModel `gguf:"v"`
	ImageProcessor

	*Options

	positionCache []int32
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	pixelValues, grid, err := m.ImageProcessor.ProcessImage(ctx, img)
	if err != nil {
		return nil, err
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues, grid)
	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	m.positionCache = m.positionCache[:0]
	return slices.Collect(func(yield func(*input.Input) bool) {
		for i := range inputs {
			s := []modelInput{{Input: inputs[i]}}
			if mm := inputs[i].Multimodal; mm != nil {
				t := mm[0].Tensor
				s = slices.Repeat([]modelInput{
					{
						position: int32(i + 1),
						Input:    &input.Input{Token: tokenVision},
					},
				}, t.Dim(1)+1+1)

				s[0] = modelInput{
					Input:    &input.Input{Token: tokenVisionStart},
					position: int32(i),
				}

				s[len(s)-1] = modelInput{
					Input:    &input.Input{Token: tokenVisionEnd},
					position: int32(i + mm[0].Data.(*Grid).Width/m.VisionModel.spatialMergeSize + 1),
				}

				s[1] = modelInput{
					Input: &input.Input{
						Token:          tokenVision,
						Multimodal:     inputs[i].Multimodal,
						MultimodalHash: inputs[i].MultimodalHash,
						SameBatch:      t.Dim(1),
					},
					position: int32(i + 1),
				}
			}

			for _, e := range s {
				position := e.position
				if position == 0 && len(m.positionCache) > 0 {
					position = m.positionCache[len(m.positionCache)-1] + 1
				}

				m.positionCache = append(m.positionCache, position)
				if !yield(e.Input) {
					return
				}
			}
		}
	}), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	var positions ml.Tensor

	if len(m.mropeSections) > 0 {
		positionSlice := slices.Collect(makeSlice2D[int32](4, len(batch.Positions)))
		for i, id := range batch.Positions {
			if id < int32(len(m.positionCache)) {
				id = m.positionCache[id]
			} else if len(m.positionCache) > 0 {
				id = id - int32(len(m.positionCache)) + m.positionCache[len(m.positionCache)-1] + 1
			}

			positionSlice[0][i] = id
			positionSlice[1][i] = id
			positionSlice[2][i] = id
		}

		hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

		for _, mi := range batch.Multimodal {
			visionOutputs := mi.Multimodal[0].Tensor
			ctx.Forward(visionOutputs.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))

			if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
				for i := range visionOutputs.Dim(1) {
					w := grid.Width / m.VisionModel.spatialMergeSize
					positionSlice[1][mi.Index+i] += int32(i / w)
					positionSlice[2][mi.Index+i] += int32(i % w)
				}
			}
		}

		positions = ctx.Input().FromInts(slices.Concat(positionSlice...), len(positionSlice[0])*len(positionSlice))

		cache := m.Cache.(*HybridCache)
		m.Options.masks = createMasks(ctx)

		for i, layer := range m.Layers {
			cache.SetLayer(i)

			var outputs ml.Tensor
			if i == len(m.Layers)-1 {
				outputs = batch.Outputs
			}

			var err error
			hiddenStates, err = layer.Forward(ctx, i, hiddenStates, positions, outputs, cache, m.Options)
			if err != nil {
				return nil, err
			}
		}

		hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
		return m.Output.Forward(ctx, hiddenStates), nil
	}

	positions = ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	cache := m.Cache.(*HybridCache)
	m.Options.masks = createMasks(ctx)

	for i, layer := range m.Layers {
		cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		var err error
		hiddenStates, err = layer.Forward(ctx, i, hiddenStates, positions, outputs, cache, m.Options)
		if err != nil {
			return nil, err
		}
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	if len(m.mropeSections) > 0 {
		m.positionCache = nil
		shift = shift.Repeat(ctx, 1, 4).Reshape(ctx, -1)
	}
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

var _ model.Model = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	if c.Uint("expert_count") > 0 {
		return nil, fmt.Errorf("qwen35: expected dense model but expert_count > 0")
	}

	numLayers := int(c.Uint("block_count"))
	layers := make([]Layer, numLayers)

	type headCounts interface {
		HeadCount() []uint64
		HeadCountKV() []uint64
	}

	var isRecurrent []bool
	var headCountKV []uint64
	if hc, ok := c.(headCounts); ok {
		headCountKV = hc.HeadCountKV()
	}

	isRecurrent = make([]bool, numLayers)
	hasZero := false
	hasFull := false
	for i := range numLayers {
		if i < len(headCountKV) && headCountKV[i] == 0 {
			isRecurrent[i] = true
			hasZero = true
		} else if i < len(headCountKV) && headCountKV[i] > 0 {
			hasFull = true
		}
	}
	if !hasZero || !hasFull {
		return nil, fmt.Errorf("qwen35: invalid attention.head_count_kv array; expected mix of zero and non-zero values")
	}

	for i := range layers {
		if isRecurrent[i] {
			layers[i].Operator = &GatedDeltaNet{Layer: i}
		} else {
			layers[i].Operator = &FullAttention{}
		}

		layers[i].MLP = &dense{}
	}

	opts := &Options{
		hiddenSize: int(c.Uint("embedding_length")),
		numHeads:   int(c.Uint("attention.head_count")),
		numKVHeads: func() int {
			for _, v := range headCountKV {
				if v > 0 {
					return int(v)
				}
			}
			return 0
		}(),
		keyLength:             int(c.Uint("attention.key_length")),
		valueLength:           int(c.Uint("attention.value_length")),
		ropeDim:               int(c.Uint("rope.dimension_count")),
		eps:                   c.Float("attention.layer_norm_rms_epsilon"),
		ropeType:              c.String("rope.scaling.type"),
		ropeBase:              c.Float("rope.freq_base"),
		ropeScale:             c.Float("rope.scaling.factor", 1),
		originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
		attentionScale:        float64(c.Float("attention.scale")),
		numExperts:            int(c.Uint("expert_count")),
		numExpertsUsed:        int(c.Uint("expert_used_count")),
		normTopKProb:          c.Bool("norm_top_k_prob", true),
		ssmDInner:             int(c.Uint("ssm.inner_size")),
		ssmDState:             int(c.Uint("ssm.state_size")),
		ssmNGroup:             int(c.Uint("ssm.group_count")),
		ssmDtRank:             int(c.Uint("ssm.time_step_rank")),
		convKernelSize:        int(c.Uint("ssm.conv_kernel")),
		isRecurrent:           isRecurrent,
	}
	if opts.numKVHeads == 0 {
		return nil, fmt.Errorf("qwen35: attention.head_count_kv array must include at least one non-zero value")
	}

	convDim := max(0, opts.convKernelSize-1)
	convChannels := opts.ssmDInner + 2*opts.ssmNGroup*opts.ssmDState
	headVDim := 0
	numVHeads := opts.ssmDtRank
	if numVHeads > 0 {
		headVDim = opts.ssmDInner / numVHeads
	}
	deltaStateSize := headVDim * headVDim * numVHeads

	headKDim := opts.ssmDState
	if headKDim != headVDim && headKDim > 0 && headVDim > 0 {
		return nil, fmt.Errorf("qwen35: headKDim (%d) != headVDim (%d) not supported; state computations require equal dimensions", headKDim, headVDim)
	}

	m := Model{
		Tokenizer: tokenizer.NewBytePairEncoding(
			&tokenizer.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Layers:         layers,
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
		Options:        opts,
	}

	if sections := c.Ints("mrope_sections"); len(sections) > 0 {
		opts.mropeSections = make([]int, len(sections))
		for i, s := range sections {
			opts.mropeSections[i] = int(s)
		}
	}

	m.Cache = NewHybridCache(m.Shift, convDim, convChannels, deltaStateSize)
	return &m, nil
}

func init() {
	model.Register("qwen35", New)
	model.Register("qwen3_5", New)
	model.Register("qwen3.5", New)
}
