// Package qwen4_exp implements the Qwen 4 MLX model family.
package qwen4_exp

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
	"github.com/ollama/ollama/x/models/qwen3_5"
	"github.com/ollama/ollama/x/tokenizer"
)

func init() {
	base.Register("Qwen4ExpForConditionalGeneration", NewModel)
}

var (
	_ base.Model      = (*Model)(nil)
	_ base.SelfDraft  = (*Model)(nil)
	_ base.DraftModel = (*mtpDraft)(nil)
	_ base.MediaModel = (*Model)(nil)
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
}

// NewModel validates the publisher config before constructing runtime state.
func NewModel(root *model.Root) (base.Model, error) {
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
	return &Model{
		Config:      &cfg,
		Layers:      make([]*Layer, cfg.NumHiddenLayers),
		tok:         tok,
		quantBits:   bits,
		quantGroup:  group,
		quantMode:   mode,
		tensorQuant: root.AllTensorQuant(),
		configData:  configData,
	}, nil
}

func (m *Model) NewCaches() []cache.Cache {
	fullLayers := 0
	for _, layer := range m.Layers {
		if layer.Attention != nil {
			fullLayers++
		}
	}
	caches := make([]cache.Cache, 0, len(m.Layers)+fullLayers+1)
	convTail := m.LinearConvKernelDim - 1
	convDim := 2*m.LinearNumKeyHeads*m.LinearKeyHeadDim + m.LinearNumValueHeads*m.LinearValueHeadDim
	for i := range m.Layers {
		if m.layerIsLinear(i) {
			caches = append(caches, cache.NewRecurrentCache(convTail, convDim, m.LinearNumValueHeads, m.LinearValueHeadDim, m.LinearKeyHeadDim))
		} else {
			caches = append(caches, cache.NewKVCache())
		}
	}
	for range fullLayers {
		// Full-history raw QSA keys plus exact 3-axis positions. This is the
		// simple semantic implementation; a compressed streaming cache can
		// replace it without changing the attention contract.
		caches = append(caches, cache.NewKVCache())
	}
	caches = append(caches, newEngramCache(
		int(m.NGramSize-1),
		int((m.PLEConvKernelSize-1)*m.NGramSize),
		int(m.HCCount*m.HiddenSize),
		m.EOSTokenID,
	))
	return caches
}

func (m *Model) Forward(b *batch.Batch, caches []cache.Cache) (*mlx.Array, *mlx.Array) {
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
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

	var tokens *engramCache
	pleCacheIndex := len(m.Layers)
	for _, layer := range m.Layers {
		if layer.Attention != nil {
			pleCacheIndex++
		}
	}
	if len(caches) > pleCacheIndex {
		tokens, _ = caches[pleCacheIndex].(*engramCache)
	}
	if tokens == nil {
		tokens = newEngramCache(
			int(m.NGramSize-1),
			int((m.PLEConvKernelSize-1)*m.NGramSize),
			int(m.HCCount*m.HiddenSize),
			m.EOSTokenID,
		)
	}

	for i, layer := range m.Layers {
		if layer.PLE != nil {
			hidden = mlx.Add(hidden, layer.PLE.Forward(hidden, b, tokens, m.Config))
		}
		var c cache.Cache
		if i < len(caches) {
			c = caches[i]
		}
		var side cache.Cache
		if layer.Attention != nil && layer.Attention.SideCacheIndex < len(caches) {
			side = caches[layer.Attention.SideCacheIndex]
		}
		hidden = layer.Forward(hidden, b, c, side, positions, ropePositions, m.Config)
	}
	multiHidden := hidden
	hidden = m.OutputMixer.Reduce(multiHidden, m.Config)
	return hidden, multiHidden
}

func (m *Model) Unembed(hidden *mlx.Array) *mlx.Array {
	return m.LMHead.Forward(hidden)
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	if m.Vision == nil {
		return nil, fmt.Errorf("this model does not support media input")
	}
	return m.Vision.PrepareMedia(segments)
}

func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	return m.Vision.EncodeMedia(item, data)
}

func (m *Model) MaxContextLength() int {
	return int(m.MaxPositionEmbeddings)
}

// mtpDraft views the checkpoint's inline predictor through DraftModel.
type mtpDraft Model

func (m *Model) SelfDraft() base.DraftModel {
	if m.MTP == nil {
		return nil
	}
	return (*mtpDraft)(m)
}

func (m *mtpDraft) NewCaches() []cache.Cache {
	if m.MTP == nil {
		return nil
	}
	return []cache.Cache{cache.NewKVCache(), cache.NewKVCache()}
}

func (m *mtpDraft) LoadWeights(map[string]*mlx.Array) error { return nil }

func (m *mtpDraft) Unembed(x *mlx.Array) *mlx.Array { return (*Model)(m).Unembed(x) }

func (m *mtpDraft) Forward(b *batch.Batch, _, draftCaches []cache.Cache) (hidden, auxHidden *mlx.Array) {
	B, L := int32(b.InputIDs.Dim(0)), int32(b.InputIDs.Dim(1))
	positions := mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets))
	var mrope []int32
	if m.Vision != nil {
		mrope = m.Vision.RopePositions(b, L)
	}
	ropePositions := canonicalRopePositionRows(b, L, mrope)

	embedding := m.EmbedTokens.Forward(b.InputIDs)
	// The reference MTP is text-only: visual information reaches it through
	// the target's multi-stream hidden state, while this branch embeds raw IDs.
	embedding = m.MTP.FCEmbedding.Forward(m.MTP.EmbeddingNorm.Forward(embedding, m.RMSNormEps))

	backbone := m.MTP.HiddenNorm.Forward(b.Hidden, m.RMSNormEps)
	backbone = mlx.Reshape(backbone, B, L, m.HCCount, m.HiddenSize)
	backbone = m.MTP.FCHidden.Forward(backbone)
	backbone = mlx.Add(backbone, mlx.ExpandDims(embedding, -2))
	backbone = mlx.Reshape(backbone, B, L, m.HCCount*m.HiddenSize)

	var main, side cache.Cache
	if len(draftCaches) > 0 {
		main = draftCaches[0]
	}
	if len(draftCaches) > 1 {
		side = draftCaches[1]
	}
	multi := m.MTP.Layer.Forward(backbone, b, main, side, positions, ropePositions, m.Config)
	sample := m.MTP.Mixer.Reduce(multi, m.Config)
	return sample, multi
}
