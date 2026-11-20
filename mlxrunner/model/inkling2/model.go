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

	Embed         *nn.Embedding
	EmbedNorm     *mlx.Array
	OutputNorm    *mlx.Array
	OutputQuery   *mlx.Array
	Output        *nn.Linear
	CompactOutput *mlx.Array
	fp32Output    bool
	Layers        []decoderLayer
}

type decoderLayer struct {
	Attention          selfAttention
	AttentionNorm      []*mlx.Array
	AttentionQuery     *mlx.Array
	AttentionOutput    shortConv
	FeedForward        routedExperts
	FeedForwardNorm    *mlx.Array
	FeedForwardQuery   *mlx.Array
	FeedForwardOutput  shortConv
	MessageProjections []messageGate
	MessageQueries     *mlx.Array
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

func (m *targetModel) Forward(b *batch.Batch, caches []cache.Layer) (hidden, auxHidden *mlx.Array) {
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
	}
	hidden = modelRMS(state.mix(m.OutputQuery, m.config.NormalizeResidualValues), m.OutputNorm, m.config.NormEps)
	if len(features) != 0 {
		features = append(features, hidden)
		return hidden, mlx.Concatenate(features, -1)
	}
	return hidden, hidden
}

func (m *targetModel) shortVocabularyProjection(hidden *mlx.Array) *mlx.Array {
	if m.CompactOutput == nil || hidden.NumDims() != 3 {
		return nil
	}
	b, t, width := hidden.Dim(0), hidden.Dim(1), hidden.Dim(2)
	if b < 1 || b > 2 || t < 1 || t > shortTargetRows {
		return nil
	}
	scores := mlx.MatmulBF16Weight(hidden.Reshape(b*t, width), m.CompactOutput)
	if scores == nil {
		return nil
	}
	vocab := m.config.UnpaddedVocab
	return scores.Slice(mlx.Slice(), mlx.Slice(0, vocab)).Reshape(b, t, vocab)
}

func (m *targetModel) Unembed(hidden *mlx.Array) *mlx.Array {
	if logits := m.shortVocabularyProjection(hidden); logits != nil {
		return softcap(logits, m.config.LogitCap)
	}
	if m.fp32Output {
		hidden = hidden.AsType(mlx.DTypeFloat32)
	}
	return softcap(projectTarget(m.Output, hidden).AsType(mlx.DTypeFloat32), m.config.LogitCap)
}

func modelRMS(x, weight *mlx.Array, epsilon float32) *mlx.Array {
	return mlx.RMSNormFn(x.AsType(mlx.DTypeFloat32), weight.AsType(mlx.DTypeFloat32), epsilon).AsType(mlx.DTypeBFloat16)
}

func (l decoderLayer) forward(state residualStream, messages *mlx.Array, b *batch.Batch, caches [numVisitCaches]cache.Layer, boundary bool, cfg textConfig) residualStream {
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
	for index, input := range inputs {
		if messages != nil {
			input = l.MessageProjections[index].apply(input, selectMessage(messages, index))
		}
		inputs[index] = modelRMS(input, l.AttentionNorm[index], cfg.NormEps)
	}
	attention := l.Attention.forward(inputs[0], inputs[1], inputs[2], b, caches[0], caches[1], caches[2])
	attention = l.AttentionOutput.forward(attention, nil, b, caches[3])
	subtotal := state.afterAttention(attention, boundary)
	ffnInput := subtotal.mix(l.FeedForwardQuery, cfg.NormalizeResidualValues)
	if messages != nil {
		ffnInput = l.MessageProjections[3].apply(ffnInput, selectMessage(messages, 3))
	}
	ffn := l.FeedForward.forward(modelRMS(ffnInput, l.FeedForwardNorm, cfg.NormEps))
	ffn = l.FeedForwardOutput.forward(ffn, nil, b, caches[4])
	subtotal.partial = mlx.Add(subtotal.partial.AsType(mlx.DTypeFloat32), ffn.AsType(mlx.DTypeFloat32)).AsType(state.partial.DType())
	return subtotal
}

func selectMessage(inputs *mlx.Array, index int) *mlx.Array {
	return inputs.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(index, index+1), mlx.Slice()).Squeeze(2)
}
