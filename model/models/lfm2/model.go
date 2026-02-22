package lfm2

import (
	"bytes"
	"cmp"
	"errors"
	"fmt"
	"image"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Options struct {
	hiddenSize       int
	headDim, ropeDim int

	eps, ropeBase, ropeScale float32

	ropeType              string
	originalContextLength int

	// per-layer head counts (LFM2 alternates attention and recurrent layers)
	numHeadsByLayer   []int
	numKVHeadsByLayer []int
}

func (o Options) headDimValue() int {
	// Head dim is shared across layers; fall back to first attention layer head count.
	for _, h := range o.numHeadsByLayer {
		if h > 0 {
			return cmp.Or(o.headDim, o.hiddenSize/h)
		}
	}
	return cmp.Or(o.headDim, o.hiddenSize)
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(1.),
			rope.WithAttentionFactor(attnFactor),
		)
	}

	headCount := 1
	for _, h := range o.numHeadsByLayer {
		if h > 0 {
			headCount = h
			break
		}
	}
	return nn.RoPE(ctx, states, positions, cmp.Or(o.ropeDim, o.headDim, o.hiddenSize/headCount), o.ropeBase, 1./o.ropeScale, opts...)
}

type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm,alt:token_embd_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	VisionModel      *VisionModel     `gguf:"v"`
	VisionProjector  *VisionProjector `gguf:"mm"`
	ImageProcessor   ImageProcessor
	imageTokenID     int32
	imageStartToken  int32
	imageEndToken    int32
	imageThumbnailID int32
	imageRowColIDs   map[imageGridPos]int32
	useSpecialTokens bool
	projectorOptions VisionProjectorOptions

	Options
}

var _ model.MultimodalProcessor = (*Model)(nil)

type imageGridPos struct {
	row int
	col int
}

type visionEmbeddingLayout struct {
	rows         int
	cols         int
	hasThumbnail bool
}

type visionChunkData struct {
	tokens    int
	row       int
	col       int
	thumbnail bool
	layout    *visionEmbeddingLayout
}

func (m *Model) Validate() error {
	if m.TokenEmbedding == nil {
		return errors.New("lfm2: missing token_embd tensor")
	}
	if m.OutputNorm == nil {
		return errors.New("lfm2: missing output_norm tensor")
	}
	if m.Output == nil {
		return errors.New("lfm2: missing output tensor")
	}

	for i, layer := range m.Layers {
		if layer.AttentionNorm == nil {
			return fmt.Errorf("lfm2: missing blk.%d.attn_norm tensor", i)
		}
		if layer.MLPNorm == nil {
			return fmt.Errorf("lfm2: missing blk.%d.ffn_norm tensor", i)
		}
		if layer.MLP == nil || layer.MLP.Up == nil || layer.MLP.Down == nil || layer.MLP.Gate == nil {
			return fmt.Errorf("lfm2: missing blk.%d feed-forward tensors", i)
		}

		switch op := layer.Operator.(type) {
		case *Attention:
			if op == nil || op.Query == nil || op.Key == nil || op.Value == nil || op.Output == nil || op.QueryNorm == nil || op.KeyNorm == nil {
				return fmt.Errorf("lfm2: missing blk.%d attention tensors", i)
			}
		case *ShortConv:
			if op == nil || op.Conv == nil || op.Conv.Weight == nil || op.InProj == nil || op.OutProj == nil {
				return fmt.Errorf("lfm2: missing blk.%d shortconv tensors", i)
			}
		default:
			return fmt.Errorf("lfm2: unsupported operator at blk.%d", i)
		}
	}

	if m.VisionModel != nil {
		if m.VisionModel.PatchEmbedding == nil {
			return errors.New("lfm2: missing vision patch embedding tensors")
		}
		if m.VisionModel.PositionEmbedding == nil {
			return errors.New("lfm2: missing vision position embedding tensors")
		}
		if m.VisionModel.PostLayerNorm == nil {
			return errors.New("lfm2: missing vision post layer norm tensors")
		}
		if len(m.VisionModel.Layers) == 0 {
			return errors.New("lfm2: missing vision encoder layers")
		}
		for i, layer := range m.VisionModel.Layers {
			if layer.LayerNorm1 == nil || layer.LayerNorm2 == nil || layer.SelfAttention == nil || layer.MLP == nil {
				return fmt.Errorf("lfm2: missing vision layer tensors at v.blk.%d", i)
			}
			if layer.SelfAttention.Query == nil || layer.SelfAttention.Key == nil || layer.SelfAttention.Value == nil || layer.SelfAttention.Output == nil {
				return fmt.Errorf("lfm2: missing vision attention tensors at v.blk.%d", i)
			}
			if layer.MLP.Up == nil || layer.MLP.Down == nil {
				return fmt.Errorf("lfm2: missing vision feed-forward tensors at v.blk.%d", i)
			}
		}

		if m.VisionProjector == nil || m.VisionProjector.Linear1 == nil || m.VisionProjector.Linear2 == nil {
			return errors.New("lfm2: missing multimodal projector tensors")
		}
	}

	return nil
}

func New(c fs.Config) (model.Model, error) {
	if c.Uint("expert_count") > 0 {
		return nil, model.ErrUnsupportedModel
	}

	if c.String("tokenizer.ggml.model") != "gpt2" {
		return nil, model.ErrUnsupportedTokenizer
	}

	vocabulary := tokenizer.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Scores: c.Floats("tokenizer.ggml.scores"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	var pretokenizers []string
	switch c.String("tokenizer.ggml.pre") {
	case "default":
		// use default BPE pretokenizer
	default:
		// llama-bpe style (default for LFM2)
		pretokenizers = []string{
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		}
	}

	m := Model{
		Tokenizer:       tokenizer.NewBytePairEncoding(&vocabulary, pretokenizers...),
		Layers:          make([]Layer, c.Uint("block_count")),
		ImageProcessor:  newImageProcessor(c),
		VisionModel:     newVisionModel(c),
		VisionProjector: &VisionProjector{},
		imageRowColIDs:  make(map[imageGridPos]int32),
		projectorOptions: VisionProjectorOptions{
			scaleFactor:  int(c.Uint("vision.projector.scale_factor", 2)),
			useLayerNorm: c.Bool("vision.projector.use_layernorm", false),
		},
		Options: Options{
			hiddenSize:            int(c.Uint("embedding_length")),
			headDim:               int(c.Uint("attention.key_length")),
			ropeDim:               int(c.Uint("rope.dimension_count")),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeType:              c.String("rope.scaling.type"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.scaling.factor", 1),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
		},
	}

	lookupTokenID := func(token string) int32 {
		for i, t := range vocabulary.Values {
			if t == token {
				return int32(i)
			}
		}
		return 0
	}

	resolveTokenID := func(explicitKey, token string, fallback uint32) int32 {
		if explicitKey != "" {
			if id := c.Uint(explicitKey); id != 0 {
				return int32(id)
			}
		}
		if tokenID := lookupTokenID(token); tokenID != 0 {
			return tokenID
		}
		return int32(fallback)
	}

	m.imageTokenID = resolveTokenID("vision.image_token_id", "<image>", 396)
	m.imageStartToken = resolveTokenID("vision.image_start_token_id", "<|image_start|>", 0)
	m.imageEndToken = resolveTokenID("vision.image_end_token_id", "<|image_end|>", 0)
	m.imageThumbnailID = resolveTokenID("vision.image_thumbnail_token_id", "<|img_thumbnail|>", 0)
	m.useSpecialTokens = c.Bool("vision.use_image_special_tokens", true)

	maxGridTokens := int(c.Uint("vision.max_tiles", 10))
	if maxGridTokens <= 0 {
		maxGridTokens = 10
	}
	for row := 1; row <= maxGridTokens; row++ {
		for col := 1; col <= maxGridTokens; col++ {
			token := fmt.Sprintf("<|img_row_%d_col_%d|>", row, col)
			if tokenID := lookupTokenID(token); tokenID > 0 {
				m.imageRowColIDs[imageGridPos{row: row, col: col}] = tokenID
			}
		}
	}

	if !m.useSpecialTokens {
		m.imageStartToken = 0
		m.imageEndToken = 0
		m.imageThumbnailID = 0
		m.imageRowColIDs = map[imageGridPos]int32{}
	}

	if c.Uint("vision.block_count") == 0 {
		m.VisionModel = nil
		m.VisionProjector = nil
	}

	type headCounts interface {
		HeadCount() []uint64
		HeadCountKV() []uint64
	}
	hc, ok := c.(headCounts)
	if !ok {
		return nil, model.ErrUnsupportedModel
	}

	headCount := hc.HeadCount()
	headCountKV := hc.HeadCountKV()

	m.numHeadsByLayer = make([]int, len(m.Layers))
	m.numKVHeadsByLayer = make([]int, len(m.Layers))
	for i := range m.Layers {
		m.numHeadsByLayer[i] = int(headCount[i])
		m.numKVHeadsByLayer[i] = int(headCountKV[i])

		if m.numKVHeadsByLayer[i] == 0 {
			m.Layers[i].Operator = &ShortConv{}
		} else {
			m.Layers[i].Operator = &Attention{}
		}
	}

	lCache := int(c.Uint("shortconv.l_cache"))
	dConv := max(0, lCache-1)
	m.Cache = NewHybridCache(m.Shift, m.hiddenSize, dConv)
	return &m, nil
}

type Operator interface {
	Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, layer int, opts *Options) ml.Tensor
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output,alt:attn_out"`
}

func (sa *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, layer int, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)
	headDim := opts.headDimValue()
	numHeads := opts.numHeadsByLayer[layer]
	numKVHeads := opts.numKVHeadsByLayer[layer]

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, headDim, numHeads, batchSize)
	key = key.Reshape(ctx, headDim, numKVHeads, batchSize)
	value = value.Reshape(ctx, headDim, numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Operator      Operator
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenState, positions, outputs ml.Tensor, cache *HybridCache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.Operator.Forward(ctx, hiddenState, positions, cache, layer, opts)

	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

func multimodalTokenCount(mm input.Multimodal) int {
	if mm.Tensor != nil {
		return mm.Tensor.Dim(1)
	}

	switch data := mm.Data.(type) {
	case int:
		return data
	case int32:
		return int(data)
	case visionChunkData:
		return data.tokens
	case *visionChunkData:
		if data != nil {
			return data.tokens
		}
	}

	return 0
}

func multimodalChunkInfo(mm input.Multimodal) visionChunkData {
	switch data := mm.Data.(type) {
	case visionChunkData:
		return data
	case *visionChunkData:
		if data != nil {
			return *data
		}
	}

	return visionChunkData{
		tokens: multimodalTokenCount(mm),
	}
}

func multimodalLayout(mm []input.Multimodal) visionEmbeddingLayout {
	layout := visionEmbeddingLayout{rows: 1, cols: 1}
	if len(mm) == 0 {
		return layout
	}

	first := multimodalChunkInfo(mm[0])
	if first.layout != nil {
		return *first.layout
	}

	return layout
}

func (m *Model) imageRowColToken(row, col int) int32 {
	if row <= 0 || col <= 0 {
		return 0
	}
	return m.imageRowColIDs[imageGridPos{row: row, col: col}]
}

func (m *Model) appendImageChunk(result []*input.Input, chunk input.Multimodal, imageToken int32, hash uint64) ([]*input.Input, error) {
	tokenCount := multimodalTokenCount(chunk)
	if tokenCount <= 0 {
		return nil, errors.New("lfm2: multimodal input has no tokens")
	}

	result = append(result, &input.Input{
		Token:          imageToken,
		Multimodal:     []input.Multimodal{chunk},
		MultimodalHash: hash,
		SameBatch:      tokenCount - 1,
	})

	for range tokenCount - 1 {
		result = append(result, &input.Input{Token: imageToken})
	}

	return result, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if m.VisionModel == nil || m.VisionProjector == nil || len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	processedImages, layout, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, err
	}

	if m.ImageProcessor.patchSize <= 0 {
		return nil, errors.New("lfm2: invalid vision patch size")
	}

	layoutInfo := &visionEmbeddingLayout{
		rows:         layout.rows,
		cols:         layout.cols,
		hasThumbnail: layout.hasThumbnail,
	}

	mm := make([]input.Multimodal, 0, len(processedImages))
	for i, processed := range processedImages {
		patches := visionPatchGrid{
			Width:  processed.size.X / m.ImageProcessor.patchSize,
			Height: processed.size.Y / m.ImageProcessor.patchSize,
		}
		if patches.Width == 0 || patches.Height == 0 {
			return nil, errors.New("lfm2: invalid resized image dimensions")
		}

		pixelValues := ctx.Input().FromFloats(processed.data, processed.size.X, processed.size.Y, m.ImageProcessor.numChannels)
		visionOutputs := m.VisionModel.Forward(ctx, pixelValues, patches)
		projected := m.VisionProjector.Forward(ctx, visionOutputs, patches, m.projectorOptions)

		chunk := visionChunkData{
			tokens:    projected.Dim(1),
			row:       processed.row,
			col:       processed.col,
			thumbnail: processed.thumbnail,
		}
		if i == 0 {
			chunk.layout = layoutInfo
		}

		mm = append(mm, input.Multimodal{
			Tensor: projected,
			Data:   chunk,
		})
	}

	return mm, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	imageToken := m.imageTokenID
	if imageToken == 0 {
		imageToken = 396
	}
	useSpecialTokens := m.useSpecialTokens || m.imageStartToken > 0 || m.imageEndToken > 0 || m.imageThumbnailID > 0 || len(m.imageRowColIDs) > 0

	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
			continue
		}

		layout := multimodalLayout(inp.Multimodal)
		if layout.rows <= 0 {
			layout.rows = 1
		}
		if layout.cols <= 0 {
			layout.cols = 1
		}
		tiles := layout.rows * layout.cols
		multitile := tiles > 1

		if useSpecialTokens && m.imageStartToken > 0 {
			result = append(result, &input.Input{Token: m.imageStartToken})
		}

		for i, mm := range inp.Multimodal {
			chunk := multimodalChunkInfo(mm)
			if chunk.tokens <= 0 {
				chunk.tokens = multimodalTokenCount(mm)
			}

			if multitile && !chunk.thumbnail && chunk.row == 0 && chunk.col == 0 && i < tiles {
				chunk.row = i/layout.cols + 1
				chunk.col = i%layout.cols + 1
			}
			if multitile && layout.hasThumbnail && i == tiles {
				chunk.thumbnail = true
			}

			if useSpecialTokens && multitile {
				if chunk.thumbnail {
					if m.imageThumbnailID > 0 {
						result = append(result, &input.Input{Token: m.imageThumbnailID})
					}
				} else if marker := m.imageRowColToken(chunk.row, chunk.col); marker > 0 {
					result = append(result, &input.Input{Token: marker})
				}
			}

			var err error
			result, err = m.appendImageChunk(result, input.Multimodal{
				Tensor: mm.Tensor,
				Data:   chunk,
			}, imageToken, inp.MultimodalHash)
			if err != nil {
				return nil, err
			}
		}

		if useSpecialTokens && m.imageEndToken > 0 {
			result = append(result, &input.Input{Token: m.imageEndToken})
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	for _, mm := range batch.Multimodal {
		offset := mm.Index
		for _, multimodal := range mm.Multimodal {
			if multimodal.Tensor == nil {
				continue
			}

			visionOutputs := multimodal.Tensor
			ctx.Forward(visionOutputs.Copy(ctx, hiddenState.View(ctx, offset*hiddenState.Stride(1), visionOutputs.Dim(0)*visionOutputs.Dim(1))))
			offset += visionOutputs.Dim(1)
		}
	}

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, outputs, m.Cache.(*HybridCache), &m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState), nil
}

func init() {
	model.Register("lfm2", New)
}
