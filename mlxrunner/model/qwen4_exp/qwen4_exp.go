// Package qwen4_exp implements the Qwen 4 MLX model family.
package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

func init() {
	model.Register("Qwen4ExpForConditionalGeneration", NewModel)
}

var (
	_ model.Model               = (*Model)(nil)
	_ model.SkippedTensorsModel = (*Model)(nil)
	_ model.SelfDraft           = (*Model)(nil)
	_ model.DraftModel          = (*mtpDraft)(nil)
	_ model.MediaModel          = (*Model)(nil)
)

// Model carries the validated architecture and tokenizer.
type Model struct {
	EmbedTokens nn.EmbeddingLayer
	Layers      []*Layer
	OutputMixer *hyperConnection
	LMHead      nn.LinearLayer
	MTP         *MTPHead
	Vision      *qwen3_5.VisionAdapter

	*Config
	tok         *tokenizer.Tokenizer
	quantBits   int
	quantGroup  int
	quantMode   string
	tensorQuant map[string]*model.TensorQuantInfo
	configData  []byte
	hostPLE     map[string]*hostPLETable
	numCaches   int
}

// NewModel validates the publisher config before constructing runtime state.
func NewModel(root *model.Root) (model.Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}
	cfg, err := parseConfig(configData)
	if err != nil {
		return nil, err
	}

	tokenizerData, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	tokenizerConfig := &tokenizer.TokenizerConfig{ConfigJSON: configData}
	if data, err := root.Manifest.ReadConfig("generation_config.json"); err == nil {
		tokenizerConfig.GenerationConfigJSON = data
	}
	if data, err := root.Manifest.ReadConfig("tokenizer_config.json"); err == nil {
		tokenizerConfig.TokenizerConfigJSON = data
	}
	tok, err := tokenizer.LoadFromBytesWithConfig(tokenizerData, tokenizerConfig)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}
	group, bits, mode := model.QuantizationParams(root.QuantType())
	if root.GroupSize() > 0 {
		group = root.GroupSize()
	}
	hostPLE, err := openHostPLETables(root, &cfg)
	if err != nil {
		return nil, fmt.Errorf("open host-backed PLE: %w", err)
	}
	return &Model{
		Config:      &cfg,
		Layers:      make([]*Layer, cfg.NumHiddenLayers),
		tok:         tok,
		quantBits:   bits,
		quantGroup:  group,
		quantMode:   mode,
		tensorQuant: root.AllTensorQuant(),
		configData:  configData,
		hostPLE:     hostPLE,
	}, nil
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	linears := model.NewLinearFactory(tensors, m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
	var err error
	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, "model.language_model.embed_tokens", m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing embedding weight")
	}
	if m.LMHead, err = requiredLinear(linears, "lm_head"); err != nil {
		return err
	}
	if m.OutputMixer, err = loadHyperConnection(linears, tensors, "model.language_model.hyper_connection_mixer", false, m.Config); err != nil {
		return err
	}

	m.numCaches = len(m.Layers)
	for i := range m.Layers {
		prefix := fmt.Sprintf("model.language_model.layers.%d", i)
		if m.Layers[i], err = m.loadLayer(linears, tensors, prefix, m.LayerTypes[i], i); err != nil {
			return fmt.Errorf("load layer %d: %w", i, err)
		}
		layer := m.Layers[i]
		if layer.Attention != nil {
			layer.indexerCache = m.numCaches
			m.numCaches++
		}
		if layer.PLE != nil {
			layer.pleCache = m.numCaches
			m.numCaches++
		}
	}
	if m.MTP, err = m.loadMTP(linears, tensors); err != nil {
		return fmt.Errorf("load MTP: %w", err)
	}
	m.Vision, err = qwen3_5.NewVisionAdapter(m.configData, tensors, qwen3_5.VisionAdapterConfig{
		HiddenSize:   m.HiddenSize,
		RopeDim:      m.RopeDim,
		RopeTheta:    m.RopeParameters.RopeTheta,
		MropeSection: m.RopeParameters.MRoPESection,
		QuantGroup:   m.quantGroup,
		QuantBits:    m.quantBits,
		QuantMode:    m.quantMode,
		TensorQuant:  m.tensorQuant,
	})
	if err != nil {
		return fmt.Errorf("load vision tower: %w", err)
	}

	return nil
}

func (m *Model) NewCaches() []cache.Cache {
	caches := make([]cache.Cache, m.numCaches)
	convTail := m.LinearConvKernelDim - 1
	convDim := 2*m.LinearNumKeyHeads*m.LinearKeyHeadDim + m.LinearNumValueHeads*m.LinearValueHeadDim
	for i, layer := range m.Layers {
		if layer.Linear != nil {
			caches[i] = cache.NewRecurrentCache(convTail, convDim, m.LinearNumValueHeads, m.LinearValueHeadDim, m.LinearKeyHeadDim)
		} else {
			caches[i] = cache.NewKVCache()
			// QSA retains raw index keys and exact 3-axis positions separately
			// from the attention's rotated keys and values.
			caches[layer.indexerCache] = cache.NewKVCache()
		}
		if layer.PLE != nil {
			caches[layer.pleCache] = layer.PLE.NewCache(m.Config)
		}
	}
	return caches
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Layer) (*mlx.Array, *mlx.Array) {
	// The runner may append draft caches; only the target's slots belong here.
	if len(caches) != 0 && len(caches) < m.numCaches {
		panic(fmt.Sprintf("Qwen cache count = %d, need at least %d", len(caches), m.numCaches))
	}
	L := int32(b.InputIDs.Dim(1))
	var mrope []int32
	if m.Vision != nil {
		mrope = m.Vision.RopePositions(b, L)
	}
	ropePositions := canonicalRopePositionRows(b, L, mrope)
	embedding := m.EmbedTokens.Forward(b.InputIDs)
	if len(b.Media) > 0 {
		embedding = m.Vision.ScatterMedia(embedding, b, 0)
	}
	hidden := expandStreams(embedding, m.Config)

	for i, layer := range m.Layers {
		if layer.PLE != nil {
			var tokens engramState
			if len(caches) > 0 {
				tokens = caches[layer.pleCache].(engramState)
			} else {
				temporary := layer.PLE.NewCache(m.Config)
				defer temporary.Free()
				tokens = temporary
			}
			hidden = mlx.Add(hidden, layer.PLE.Forward(hidden, b, tokens, m.Config))
		}
		var c, side cache.Layer
		if len(caches) > 0 {
			c = caches[i]
			if layer.Attention != nil {
				side = caches[layer.indexerCache]
			}
		}
		hidden = layer.Forward(hidden, b, c, side, ropePositions, m.Config)
	}
	multiHidden := hidden
	hidden = m.OutputMixer.Reduce(multiHidden, m.Config)
	return hidden, multiHidden
}

func (m *Model) Unembed(hidden *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(hidden)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

func (m *Model) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	if m.Vision == nil {
		return nil, fmt.Errorf("this model does not support media input")
	}
	return m.Vision.PrepareMedia(segments)
}

func (m *Model) EncodeMedia(item *model.PreparedItem, data *mlx.Array) *mlx.Array {
	return m.Vision.EncodeMedia(item, data)
}

type Layer struct {
	AttentionConnection *hyperConnection
	MLPConnection       *hyperConnection
	Linear              *linearAttention
	Attention           *fullAttention
	MoE                 *sparseMoE
	PLE                 *PLE

	// Main attention/recurrent caches occupy the first len(Layers) slots.
	// These indices identify this layer's additional state in the same slice.
	indexerCache int
	pleCache     int
}

func (m *Model) loadLayer(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix, kind string, layer int) (*Layer, error) {
	l := &Layer{}
	var err error
	if l.AttentionConnection, err = loadHyperConnection(linears, tensors, prefix+".attn_hyper_connection", true, m.Config); err != nil {
		return nil, err
	}
	if l.MLPConnection, err = loadHyperConnection(linears, tensors, prefix+".mlp_hyper_connection", true, m.Config); err != nil {
		return nil, err
	}
	if kind == "linear_attention" {
		if l.Linear, err = loadLinearAttention(linears, tensors, prefix+".linear_attn"); err != nil {
			return nil, err
		}
	} else if l.Attention, err = loadFullAttention(linears, tensors, prefix+".self_attn", m.RMSNormEps); err != nil {
		return nil, err
	}
	if l.MoE, err = m.loadSparseMoE(linears, tensors, prefix+".mlp"); err != nil {
		return nil, err
	}

	for _, id := range m.PLELayerIDs {
		// PLE layer IDs are one-based and identify the layer receiving the
		// additive n-gram features before its attention branch.
		if int(id-1) == layer {
			l.PLE, err = m.loadPLE(linears, tensors, prefix+".ple")
			if err != nil {
				return nil, err
			}
		}
	}
	return l, nil
}

func (l *Layer) Forward(x *mlx.Array, b *batch.Batch, c, side cache.Layer, ropePositions *mlx.Array, cfg *Config) *mlx.Array {
	branch, state := l.AttentionConnection.Prepare(x, cfg)
	if l.Linear != nil {
		branch = l.Linear.Forward(branch, b, c, cfg)
	} else {
		branch = l.Attention.Forward(branch, b, c, side, ropePositions, cfg)
	}
	x = l.AttentionConnection.Inject(state, branch, cfg)

	branch, state = l.MLPConnection.Prepare(x, cfg)
	branch = l.MoE.Forward(branch, cfg)
	return l.MLPConnection.Inject(state, branch, cfg)
}
