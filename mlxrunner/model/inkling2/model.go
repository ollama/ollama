package inkling2

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("Inkling2LoopForConditionalGeneration", newModel)
}

type targetModel struct {
	config    textConfig
	plan      []layerVisit
	tok       *tokenizer.Tokenizer
	Vision    *visionEncoder
	draftTaps []int

	Embed       *nn.Embedding
	EmbedNorm   *mlx.Array
	OutputNorm  *mlx.Array
	OutputQuery *mlx.Array
	Output      *nn.Linear
	ShortOutput *nn.Linear
	fp32Output  bool
	Layers      []decoderLayer
}

type decoderLayer struct {
	Attention          selfAttention
	AttentionNorm      []*mlx.Array
	AttentionNorms     *mlx.Array // [3, width]: the query, key and value norms
	AttentionQuery     *mlx.Array
	AttentionOutput    shortConv
	FeedForward        routedExperts
	FeedForwardNorm    *mlx.Array
	FeedForwardQuery   *mlx.Array
	FeedForwardOutput  shortConv
	MessageProjections []messageGate
	MessageQueries     *mlx.Array
	Controllers        *loopControllers
}

const numVisitCaches = 5

func newModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Text   textConfig    `json:"text_config"`
		Vision *visionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse Inkling2 config: %w", err)
	}
	if err := cfg.Text.validate(); err != nil {
		return nil, err
	}
	plan, err := cfg.Text.executionPlan()
	if err != nil {
		return nil, err
	}
	tokData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	tokConfig, err := root.Manifest.ReadConfig("tokenizer_config.json")
	if err != nil {
		return nil, err
	}
	var generation []byte
	if _, exists := root.Manifest.ConfigLayer("generation_config.json"); exists {
		generation, err = root.Manifest.ReadConfig("generation_config.json")
		if err != nil {
			return nil, err
		}
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokData, &tokenizer.TokenizerConfig{
		ConfigJSON: raw, GenerationConfigJSON: generation, TokenizerConfigJSON: tokConfig,
	})
	if err != nil {
		return nil, fmt.Errorf("parse Inkling2 tokenizer: %w", err)
	}

	m := &targetModel{config: cfg.Text, plan: plan, tok: tok}
	if cfg.Vision != nil {
		if err := cfg.Vision.validate(); err != nil {
			return nil, err
		}
		var processor struct {
			ImageToken string `json:"image_token"`
		}
		if err := root.Manifest.ReadConfigJSON("processor_config.json", &processor); err != nil {
			return nil, fmt.Errorf("load Inkling2 image processor: %w", err)
		}
		id, ok := tok.GetSpecialToken(processor.ImageToken)
		if !ok {
			return nil, fmt.Errorf("Inkling2 image token %q is missing from the tokenizer", processor.ImageToken)
		}
		m.Vision = &visionEncoder{config: *cfg.Vision, imageToken: id}
	}
	if _, exists := root.Manifest.ConfigLayer("hf_quant_config.json"); exists {
		var quant struct {
			Model struct {
				Groups []struct {
					Tensors []string `json:"tensors"`
					Output  string   `json:"fwd_output_dtype"`
				} `json:"parameter_groups"`
			} `json:"model"`
		}
		if err := root.Manifest.ReadConfigJSON("hf_quant_config.json", &quant); err != nil {
			return nil, fmt.Errorf("parse Inkling2 precision config: %w", err)
		}
		for _, group := range quant.Model.Groups {
			if slices.Contains(group.Tensors, "model.llm.unembed.weight") {
				m.fp32Output = group.Output == "float32"
			}
		}
	}
	return m, nil
}

func (m *targetModel) MaxContextLength() int           { return m.config.Context }
func (m *targetModel) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// Routed low-precision experts can make different choices when the same rows
// are processed in a different size of token batch.
func (*targetModel) AlignPrefillToSnapshots() {}

func (m *targetModel) NewCaches() []cache.Cache {
	states := make([]cache.Cache, 0, numVisitCaches*len(m.plan))
	for _, visit := range m.plan {
		a := m.Layers[visit.layer].Attention
		if a.window > 0 {
			states = append(states, cache.NewOrderedWindowKVCache(a.window))
		} else {
			states = append(states, cache.NewKVCache())
		}
		for _, width := range []int{a.kvHeads * a.headDim, a.kvHeads * a.headDim, m.config.HiddenSize, m.config.HiddenSize} {
			states = append(states, cache.NewRecurrentCache(int32(m.config.ConvTaps-1), int32(width), 1, 1, 1))
		}
	}
	return states
}

func (m *targetModel) embedTokens(ids *mlx.Array) *mlx.Array {
	return modelRMS(m.Embed.Forward(ids), m.EmbedNorm, m.config.NormEps)
}

// submitsEarly reports whether a target or draft forward submits work before
// it finishes building, so Metal executes it while the host builds the rest of
// the round. It is limited to the decode, verification and proposal rounds it
// was measured on: one or two requests, short blocks, each past its prompt. The
// load-time shape check runs at offset zero and evaluates nothing, so it must
// not queue GPU work.
func submitsEarly(b *batch.Batch) bool {
	if b.InputIDs.Dim(0) > 2 || b.InputIDs.Dim(1) > shortTargetRows || !mlx.MetalIsAvailable() {
		return false
	}
	return len(b.SeqOffsets) > 0 && !slices.Contains(b.SeqOffsets, 0)
}

func (m *targetModel) Forward(b *batch.Batch, caches []cache.Layer) (hidden, auxHidden *mlx.Array) {
	submitShort := submitsEarly(b)
	embeddings := m.embedTokens(b.InputIDs)
	state := residualStream{partial: m.imageEmbeddings(b, embeddings)}
	var prelude residualStream
	var previous []*mlx.Array
	features := make([]*mlx.Array, 0, len(m.draftTaps))
	for i, visit := range m.plan {
		if loop := m.config.Loop; loop != nil && visit.layer == loop.Start {
			if visit.pass == 0 {
				prelude = state
			} else {
				blocks := state.blocks[len(prelude.blocks)+1:]
				previous = append(slices.Clone(blocks), state.partial)
				state = prelude
			}
		}
		if len(features) < len(m.draftTaps) && i == m.draftTaps[len(features)] {
			features = append(features, state.partial)
		}
		var layerCaches [numVisitCaches]cache.Layer
		if caches != nil {
			copy(layerCaches[:], caches[i*numVisitCaches:(i+1)*numVisitCaches])
		}
		var messages *mlx.Array
		layer := m.Layers[visit.layer]
		if visit.pass > 0 {
			messages = mixResiduals(previous, layer.MessageQueries, true)
		}
		state = layer.forward(state, messages, b, layerCaches, visit.layer%m.config.ResidualBlockSize == 0, m.config)
		if submitShort && i%2 == 1 && i+1 < len(m.plan) {
			mlx.AsyncEval(state.partial)
		}
	}
	hidden = modelRMS(state.mix(m.OutputQuery, m.config.NormalizeResidualValues), m.OutputNorm, m.config.NormEps)
	if len(features) != 0 {
		features = append(features, hidden)
		return hidden, mlx.Concatenate(features, -1)
	}
	return hidden, hidden
}

// shortVocabularyProjection returns FP32 logits for up to eight positions per
// row from the BF16 head, or nil where only the FP32 head applies.
func (m *targetModel) shortVocabularyProjection(hidden *mlx.Array) *mlx.Array {
	if m.ShortOutput == nil || hidden.NumDims() != 3 || hidden.Dim(1) > shortTargetRows {
		return nil
	}
	return projectTarget(m.ShortOutput, hidden.AsType(mlx.DTypeFloat32))
}

// shortVocabularyLogits is Unembed for short forwards in launches that read
// the BF16 rows and apply the FP32 softcap to their sums, one per
// shortTargetRows flattened rows. It returns nil where they cannot run.
func (m *targetModel) shortVocabularyLogits(hidden *mlx.Array) *mlx.Array {
	if m.ShortOutput == nil || hidden.NumDims() != 3 || hidden.Dim(1) > shortTargetRows {
		return nil
	}
	rows, tokens := hidden.Dim(0), hidden.Dim(1)
	total := rows * tokens
	flat := hidden.Reshape(total, hidden.Dim(2))
	var tiles []*mlx.Array
	for begin := 0; begin < total; begin += shortTargetRows {
		tile := flat
		if total > shortTargetRows {
			tile = flat.Slice(mlx.Slice(begin, min(total, begin+shortTargetRows)), mlx.Slice())
		}
		logits := mlx.ShortMatmulSoftcap(tile, m.ShortOutput.Weight, m.config.LogitCap)
		if logits == nil {
			return nil
		}
		tiles = append(tiles, logits)
	}
	out := tiles[0]
	if len(tiles) > 1 {
		out = mlx.Concatenate(tiles, 0)
	}
	return out.Reshape(rows, tokens, out.Dim(1))
}

func (m *targetModel) Unembed(hidden *mlx.Array) *mlx.Array {
	if logits := m.shortVocabularyLogits(hidden); logits != nil {
		return logits
	}
	if logits := m.shortVocabularyProjection(hidden); logits != nil {
		return softcap(logits, m.config.LogitCap)
	}
	if m.fp32Output {
		hidden = hidden.AsType(mlx.DTypeFloat32)
	}
	return softcap(projectTarget(m.Output, hidden).AsType(mlx.DTypeFloat32), m.config.LogitCap)
}

func modelRMS(x, weight *mlx.Array, epsilon float32) *mlx.Array {
	if y := mlx.RMSNormRoundBF16(x, weight, epsilon); y != nil {
		return y
	}
	return mlx.RMSNormFn(x.AsType(mlx.DTypeFloat32), weight.AsType(mlx.DTypeFloat32), epsilon).AsType(mlx.DTypeBFloat16)
}

func (l decoderLayer) forward(state residualStream, messages *mlx.Array, b *batch.Batch, caches [numVisitCaches]cache.Layer, boundary bool, cfg textConfig) residualStream {
	attention := l.attention(state, messages, b, caches, cfg)
	var subtotal residualStream
	if boundary {
		subtotal = state.closeBlock(l.AttentionOutput.forward(attention, nil, b, caches[3]))
	} else {
		// Within a block the feed-forward input sees the FP32 sum.
		subtotal = residualStream{blocks: state.blocks, partial: l.AttentionOutput.forwardAdd(attention, state.partial, mlx.DTypeFloat32, b, caches[3])}
	}
	var ffnInput *mlx.Array
	if messages != nil {
		ffnInput = modelRMS(l.MessageProjections[3].apply(subtotal.mix(l.FeedForwardQuery, cfg.NormalizeResidualValues), messages, 3), l.FeedForwardNorm, cfg.NormEps)
	} else {
		ffnInput = subtotal.mixNorm(l.FeedForwardQuery, l.FeedForwardNorm, cfg)
	}
	ffn := l.FeedForward.forward(ffnInput)
	subtotal.partial = l.FeedForwardOutput.forwardAdd(ffn, subtotal.partial, state.partial.DType(), b, caches[4])
	return subtotal
}

// attention returns the attention block's output projection for one visit.
func (l decoderLayer) attention(state residualStream, messages *mlx.Array, b *batch.Batch, caches [numVisitCaches]cache.Layer, cfg textConfig) *mlx.Array {
	if all := l.stackedProjection(state, messages, cfg); all != nil {
		return l.Attention.forwardStacked(all, b, caches[0], caches[1], caches[2])
	}
	if messages == nil && len(state.blocks) != 0 && l.AttentionQuery != nil && l.AttentionQuery.NumDims() == 1 &&
		l.AttentionNorm[0] == l.AttentionNorm[1] && l.AttentionNorm[1] == l.AttentionNorm[2] {
		// Local layers share one normalized input for all three projections.
		normalized := state.mixNorm(l.AttentionQuery, l.AttentionNorm[0], cfg)
		return l.Attention.forward(normalized, normalized, normalized, b, caches[0], caches[1], caches[2])
	}
	inputs := [3]*mlx.Array{state.partial, state.partial, state.partial}
	if len(state.blocks) != 0 && l.AttentionQuery != nil {
		mixed := state.mix(l.AttentionQuery, cfg.NormalizeResidualValues)
		if l.AttentionQuery.NumDims() == 1 {
			inputs = [3]*mlx.Array{mixed, mixed, mixed}
		} else {
			for index := range inputs {
				inputs[index] = selectMessage(mixed, index)
			}
		}
	}
	raw := inputs
	for index, input := range raw {
		if messages != nil {
			input = l.MessageProjections[index].apply(input, messages, index)
		} else if index > 0 && input == raw[index-1] && l.AttentionNorm[index] == l.AttentionNorm[index-1] {
			// Local layers share one input and norm for all three projections.
			inputs[index] = inputs[index-1]
			continue
		}
		inputs[index] = modelRMS(input, l.AttentionNorm[index], cfg.NormEps)
	}
	return l.Attention.forward(inputs[0], inputs[1], inputs[2], b, caches[0], caches[1], caches[2])
}

func selectMessage(inputs *mlx.Array, index int) *mlx.Array {
	return inputs.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(index, index+1), mlx.Slice()).Squeeze(2)
}

// stackedProjection returns the product of the whole stacked attention
// projection for a short forward whose query, key and value inputs differ:
// the inputs of a looped visit's controllers or a global layer's three
// mixtures. A global layer's residual mixer normalizes the mixtures, then one
// launch multiplies each column group by its own mixture. A looped visit's gate
// launch rounds the gated inputs times the norm weights to BF16, and the
// product launch scales each group's FP32 sums by its input's inverse RMS. It
// returns nil where the separate products apply.
func (l decoderLayer) stackedProjection(state residualStream, messages *mlx.Array, cfg textConfig) *mlx.Array {
	query := l.AttentionQuery
	if query == nil || len(state.blocks) == 0 || !cfg.NormalizeResidualValues ||
		state.partial.NumDims() != 3 || state.partial.Dim(1) > shortTargetRows {
		return nil
	}
	if messages == nil && query.NumDims() == 1 {
		return nil
	}
	a := l.Attention
	batch, tokens, width := state.partial.Dim(0), state.partial.Dim(1), state.partial.Dim(2)
	rows := batch * tokens
	qRows, kRows := a.QueryRows.Weight.Dim(0), a.KeyRows.Weight.Dim(0)
	ends := [2]int{qRows, qRows + kRows}
	var all *mlx.Array
	if messages == nil {
		sources := append(state.blocks[:len(state.blocks):len(state.blocks)], state.partial)
		normalized := mlx.BlockResidualMixNorm(sources, query, l.AttentionNorms, residualEpsilon, cfg.NormEps)
		if normalized == nil {
			return nil
		}
		all = mlx.GroupedShortMatmul(normalized, rows, 3*width, width, ends, a.Projection.Weight, nil, 0)
	} else {
		inputs := 1
		if query.NumDims() == 2 {
			inputs = 3
		}
		mixed := state.mix(query, cfg.NormalizeResidualValues)
		compressed := mlx.LoopControllerDown(mixed, inputs, messages, 0, 3, l.Controllers.AttentionDown, residualEpsilon)
		if compressed == nil {
			return nil
		}
		scaled, squares := mlx.LoopControllerGateNorm(mixed, inputs, messages, 0, compressed, l.Controllers.AttentionGates, l.AttentionNorms)
		if scaled == nil {
			return nil
		}
		all = mlx.GroupedShortMatmul(scaled, rows, width, rows*width, ends, a.Projection.Weight, squares, cfg.NormEps)
	}
	if all == nil {
		return nil
	}
	return all.Reshape(batch, tokens, all.Dim(1))
}
