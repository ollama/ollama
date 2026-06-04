package gemma4

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

var (
	_ base.DraftModel    = (*AssistantModel)(nil)
	_ base.MTPDraftModel = (*AssistantModel)(nil)
)

type AssistantConfig struct {
	TextConfig               TextConfig `json:"text_config"`
	BackboneHiddenSize       int32      `json:"backbone_hidden_size"`
	UseOrderedEmbeddings     bool       `json:"use_ordered_embeddings"`
	NumCentroids             int32      `json:"num_centroids"`
	CentroidIntermediateTopK int32      `json:"centroid_intermediate_top_k"`
}

type AssistantModel struct {
	PreProjection  nn.LinearLayer
	PostProjection nn.LinearLayer
	EmbedTokens    nn.EmbeddingLayer
	LMHead         nn.LinearLayer
	Centroids      nn.LinearLayer
	TokenOrdering  *mlx.Array
	Layers         []*AssistantLayer
	Norm           *nn.RMSNorm

	NormScaled *mlx.Array

	*AssistantConfig
	tensorPrefix string

	QuantGroupSize int
	QuantBits      int
	QuantMode      string
	TensorQuant    map[string]*model.TensorQuantInfo
}

type AssistantLayer struct {
	InputNorm    *nn.RMSNorm
	PostAttnNorm *nn.RMSNorm
	PreFFNorm    *nn.RMSNorm
	PostFFNorm   *nn.RMSNorm

	InputNormScaled    *mlx.Array
	PostAttnNormScaled *mlx.Array
	PreFFNormScaled    *mlx.Array
	PostFFNormScaled   *mlx.Array

	Attention   *AssistantAttention
	MLP         *MLP
	LayerScalar *mlx.Array
	IsSliding   bool
}

type AssistantAttention struct {
	QProj nn.LinearLayer
	OProj nn.LinearLayer
	QNorm *nn.RMSNorm

	QNormScaled *mlx.Array
}

func parseAssistantConfig(configData []byte) (AssistantConfig, error) {
	var raw struct {
		TextConfig json.RawMessage `json:"text_config"`

		BackboneHiddenSize       int32 `json:"backbone_hidden_size"`
		UseOrderedEmbeddings     bool  `json:"use_ordered_embeddings"`
		NumCentroids             int32 `json:"num_centroids"`
		CentroidIntermediateTopK int32 `json:"centroid_intermediate_top_k"`
	}
	if err := json.Unmarshal(configData, &raw); err != nil {
		return AssistantConfig{}, fmt.Errorf("parse assistant config: %w", err)
	}
	if len(raw.TextConfig) == 0 {
		return AssistantConfig{}, fmt.Errorf("assistant config missing text_config")
	}

	text, err := parseTextConfig(raw.TextConfig)
	if err != nil {
		return AssistantConfig{}, err
	}
	if raw.NumCentroids == 0 {
		raw.NumCentroids = 2048
	}
	if raw.CentroidIntermediateTopK == 0 {
		raw.CentroidIntermediateTopK = 32
	}

	return AssistantConfig{
		TextConfig:               text,
		BackboneHiddenSize:       raw.BackboneHiddenSize,
		UseOrderedEmbeddings:     raw.UseOrderedEmbeddings,
		NumCentroids:             raw.NumCentroids,
		CentroidIntermediateTopK: raw.CentroidIntermediateTopK,
	}, nil
}

func newAssistantModel(root *model.Root, target base.Model) (base.DraftModel, error) {
	if root == nil || root.Draft == nil {
		return nil, fmt.Errorf("draft metadata missing")
	}

	configPath := root.Draft.Config
	if configPath == "" {
		configPath = "draft/config.json"
	}
	configData, err := root.Manifest.ReadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("load draft config: %w", err)
	}

	cfg, err := parseAssistantConfig(configData)
	if err != nil {
		return nil, err
	}

	targetGemma, ok := target.(*Model)
	if !ok {
		return nil, fmt.Errorf("gemma4 assistant requires gemma4 target, got %T", target)
	}
	if cfg.BackboneHiddenSize != 0 && cfg.BackboneHiddenSize != targetGemma.HiddenSize {
		return nil, fmt.Errorf("assistant backbone hidden size %d does not match target hidden size %d", cfg.BackboneHiddenSize, targetGemma.HiddenSize)
	}
	if cfg.TextConfig.VocabSize != targetGemma.VocabSize {
		return nil, fmt.Errorf("assistant vocab size %d does not match target vocab size %d", cfg.TextConfig.VocabSize, targetGemma.VocabSize)
	}

	tensorPrefix := root.Draft.TensorPrefix
	if tensorPrefix == "" {
		tensorPrefix = "draft."
	}

	m := &AssistantModel{
		AssistantConfig: &cfg,
		tensorPrefix:    tensorPrefix,
		Layers:          make([]*AssistantLayer, cfg.TextConfig.NumHiddenLayers),
		TensorQuant:     root.AllTensorQuant(),
	}
	if qt := root.QuantType(); qt != "" {
		m.QuantGroupSize, m.QuantBits, m.QuantMode = model.QuantizationParams(qt)
		if gs := root.GroupSize(); gs > 0 {
			m.QuantGroupSize = gs
		}
	}
	for i := range m.Layers {
		m.Layers[i] = &AssistantLayer{
			IsSliding: isLayerSliding(int32(i), &m.TextConfig),
			Attention: &AssistantAttention{},
			MLP:       &MLP{},
		}
	}
	return m, nil
}

func (m *AssistantModel) LoadWeights(tensors map[string]*mlx.Array) error {
	prefix := m.tensorPrefix
	linears := model.NewLinearFactory(tensors, m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)

	m.PreProjection = linears.Make(prefix + "pre_projection")
	m.PostProjection = linears.Make(prefix + "post_projection")
	if m.PreProjection == nil || m.PostProjection == nil {
		return fmt.Errorf("missing assistant projection weights")
	}

	m.EmbedTokens = model.MakeEmbeddingLayer(tensors, prefix+"model.embed_tokens", m.QuantGroupSize, m.QuantBits, m.QuantMode, m.TensorQuant)
	if m.EmbedTokens == nil {
		return fmt.Errorf("missing assistant embedding weight")
	}
	m.LMHead = m.EmbedTokens.AsLinear()

	if m.UseOrderedEmbeddings {
		m.Centroids = linears.Make(prefix + "masked_embedding.centroids")
		m.TokenOrdering = tensors[prefix+"masked_embedding.token_ordering"]
		if m.Centroids == nil || m.TokenOrdering == nil {
			return fmt.Errorf("missing ordered embedding tensors: %smasked_embedding.centroids.weight and %smasked_embedding.token_ordering", prefix, prefix)
		}
		m.TokenOrdering = m.TokenOrdering.AsType(mlx.DTypeInt32)
	}

	normWeight := tensors[prefix+"model.norm.weight"]
	if normWeight == nil {
		return fmt.Errorf("missing assistant final norm")
	}
	m.Norm = nn.NewRMSNorm(normWeight, m.TextConfig.RMSNormEps)

	for i := range m.TextConfig.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%smodel.layers.%d", prefix, i)
		layer := &AssistantLayer{
			IsSliding: isLayerSliding(i, &m.TextConfig),
			Attention: &AssistantAttention{
				QProj: linears.Make(layerPrefix + ".self_attn.q_proj"),
				OProj: linears.Make(layerPrefix + ".self_attn.o_proj"),
			},
			MLP: &MLP{
				GateProj: linears.Make(layerPrefix + ".mlp.gate_proj"),
				UpProj:   linears.Make(layerPrefix + ".mlp.up_proj"),
				DownProj: linears.Make(layerPrefix + ".mlp.down_proj"),
			},
			LayerScalar: tensors[layerPrefix+".layer_scalar"],
		}

		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, m.TextConfig.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, m.TextConfig.RMSNormEps)
		}
		if w := tensors[layerPrefix+".pre_feedforward_layernorm.weight"]; w != nil {
			layer.PreFFNorm = nn.NewRMSNorm(w, m.TextConfig.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
			layer.PostFFNorm = nn.NewRMSNorm(w, m.TextConfig.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, m.TextConfig.RMSNormEps)
		}

		if layer.InputNorm == nil || layer.PostAttnNorm == nil || layer.PreFFNorm == nil || layer.PostFFNorm == nil {
			return fmt.Errorf("assistant layer %d: missing norm weights", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.OProj == nil || layer.Attention.QNorm == nil {
			return fmt.Errorf("assistant layer %d: missing attention weights", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("assistant layer %d: missing mlp weights", i)
		}

		m.Layers[i] = layer
	}

	m.precomputeScaledWeights()
	return nil
}

func (m *AssistantModel) precomputeScaledWeights() {
	if m.Norm != nil {
		m.NormScaled = m.Norm.Weight
	}
	for _, layer := range m.Layers {
		if layer.InputNorm != nil {
			layer.InputNormScaled = layer.InputNorm.Weight
		}
		if layer.PostAttnNorm != nil {
			layer.PostAttnNormScaled = layer.PostAttnNorm.Weight
		}
		if layer.PreFFNorm != nil {
			layer.PreFFNormScaled = layer.PreFFNorm.Weight
		}
		if layer.PostFFNorm != nil {
			layer.PostFFNormScaled = layer.PostFFNorm.Weight
		}
		if layer.Attention != nil && layer.Attention.QNorm != nil {
			layer.Attention.QNormScaled = layer.Attention.QNorm.Weight
		}
	}
}

func (m *AssistantModel) Draft(inputsEmbeds *mlx.Array, position int32, caches []cache.Cache) (logits, hidden *mlx.Array) {
	dims := inputsEmbeds.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, int(B), int(L)),
		SeqOffsets:   []int32{position},
		SeqQueryLens: []int32{L},
	}

	sliding, full := m.sharedHistories(b, caches)
	h := m.PreProjection.Forward(inputsEmbeds)

	positions := mlx.FromValues([]int32{position}, 1)
	for _, layer := range m.Layers {
		h = layer.Forward(h, b, positions, B, L, &m.TextConfig, sliding, full)
	}

	hidden = mlx.RMSNormFn(h, m.NormScaled, m.TextConfig.RMSNormEps)
	projected := m.PostProjection.Forward(hidden)
	return m.unembed(hidden), projected
}

func (m *AssistantModel) sharedHistories(b *batch.Batch, caches []cache.Cache) (sliding, full *nn.KVHistory) {
	if len(caches) < 2 {
		return nil, nil
	}
	if v, ok := caches[len(caches)-2].(cache.Viewer); ok {
		sliding = v.View(b)
	}
	if v, ok := caches[len(caches)-1].(cache.Viewer); ok {
		full = v.View(b)
	}
	return sliding, full
}

func (m *AssistantModel) unembed(hidden *mlx.Array) *mlx.Array {
	if m.UseOrderedEmbeddings {
		return m.applyCentroidMasking(hidden)
	}
	return m.LMHead.Forward(hidden)
}

func (m *AssistantModel) applyCentroidMasking(hidden *mlx.Array) *mlx.Array {
	B, L := hidden.Dim(0), hidden.Dim(1)
	vocab := int(m.TextConfig.VocabSize)
	numCentroids := int(m.NumCentroids)
	vocabPerCentroid := vocab / numCentroids
	topK := int(m.CentroidIntermediateTopK)

	centroidLogits := m.Centroids.Forward(hidden)
	topKIndices := centroidLogits.Negative().ArgpartitionAxis(topK-1, -1).Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, topK))
	ordering := m.TokenOrdering.Reshape(numCentroids, vocabPerCentroid)
	selectedCanonical := ordering.TakeAxis(topKIndices, 0)
	selectedFlat := selectedCanonical.Reshape(B * L * topK * vocabPerCentroid)

	embeddings := m.EmbedTokens.Forward(selectedFlat)
	embeddings = embeddings.Reshape(B, L, topK*vocabPerCentroid, int(m.TextConfig.HiddenSize))
	selectedLogits := hidden.ExpandDims(2).Matmul(embeddings.Transpose(0, 1, 3, 2)).Squeeze(2)

	out := mlx.Zeros(selectedLogits.DType(), B, L, vocab)
	out = mlx.AddScalar(out, -1.0e30)
	return out.PutAlongAxis(selectedCanonical.Reshape(B, L, topK*vocabPerCentroid), selectedLogits, -1)
}

func (l *AssistantLayer) Forward(x *mlx.Array, b *batch.Batch, positions *mlx.Array, B, L int32, cfg *TextConfig, sliding, full *nn.KVHistory) *mlx.Array {
	normed := mlx.RMSNormFn(x, l.InputNormScaled, cfg.RMSNormEps)
	attnOut := l.Attention.Forward(normed, b, positions, B, L, l.IsSliding, cfg, sliding, full)
	attnOut = mlx.RMSNormFn(attnOut, l.PostAttnNormScaled, cfg.RMSNormEps)
	h := mlx.Add(x, attnOut)

	normed = mlx.RMSNormFn(h, l.PreFFNormScaled, cfg.RMSNormEps)
	mlpOut := l.MLP.Forward(normed)
	mlpOut = mlx.RMSNormFn(mlpOut, l.PostFFNormScaled, cfg.RMSNormEps)
	h = mlx.Add(h, mlpOut)

	if l.LayerScalar != nil {
		h = mlx.Mul(h, l.LayerScalar)
	}
	return h
}

func (a *AssistantAttention) Forward(x *mlx.Array, b *batch.Batch, positions *mlx.Array, B, L int32, isSliding bool, cfg *TextConfig, sliding, full *nn.KVHistory) *mlx.Array {
	headDim := cfg.HeadDim
	scale := cfg.SlidingScale
	ropeDims := cfg.SlidingRopeDims
	ropeBase := cfg.SlidingRopeBase
	history := sliding
	if !isSliding {
		headDim = cfg.GlobalHeadDim
		scale = cfg.FullScale
		ropeDims = cfg.FullRopeDims
		ropeBase = cfg.FullRopeBase
		history = full
	}
	if history == nil {
		panic("gemma4 assistant missing shared target KV history")
	}

	q := a.QProj.Forward(x)
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, headDim)
	q = mlx.Transpose(q, 0, 2, 1, 3)
	q = mlx.RMSNormFn(q, a.QNormScaled, cfg.RMSNormEps)

	var ropeFreqs *mlx.Array
	if !isSliding {
		ropeFreqs = cfg.FullRopeFreqs
	}
	q = mlx.RoPEWithFreqs(q, ropeDims, false, ropeBase, 1.0, positions, ropeFreqs)

	mask := nn.CausalMask()
	if isSliding && cfg.SlidingWindow > 0 {
		mask = mask.Intersect(nn.SlidingWindowMask(b, history.K().Dim(2), int(cfg.SlidingWindow), q.DType()))
	}

	out := nn.ScaledDotProductAttention(b, q, scale, nn.WithKVHistory(history), nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*headDim)
	if !mlx.MetalIsAvailable() {
		out = mlx.Contiguous(out, false)
	}
	return a.OProj.Forward(out)
}
