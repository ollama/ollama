// Package hy4_flash implements Tencent's Hy4 Flash model for the MLX runner.
package hy4_flash

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

const Architecture = "HYV4FlashForConditionalGeneration"

func init() {
	model.Register(Architecture, newModel)
}

type Model struct {
	cfg         config
	tok         *tokenizer.Tokenizer
	quant       map[string]*model.TensorQuantInfo
	group, bits int
	mode        string
	processor   processorConfig
	skipped     []string

	Embed     nn.EmbeddingLayer
	Head      nn.LinearLayer
	Reduce    hyperConnection
	Norm      *nn.RMSNorm
	Layers    []decoderLayer
	Vision    *visionTower
	Draft     *dflyDraft
	tokenSlot int
	numCaches int
}

type decoderLayer struct {
	Engram                 *engram
	Attention              attention
	AttentionHC, MLPHC     hyperConnection
	AttentionNorm, MLPNorm *nn.RMSNorm
	Dense                  denseMLP
	MoE                    *latentMoE
	OutNorm                *nn.RMSNorm
}

func newModel(root *model.Root) (model.Model, error) {
	raw, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	cfg, err := parseConfig(raw)
	if err != nil {
		return nil, err
	}
	if root.Draft != nil {
		// The runner gives an attached draft priority over the inline draft.
		cfg.DraftLayers = 0
	}
	data, err := root.Manifest.ReadConfig("tokenizer.json")
	if err != nil {
		return nil, err
	}
	tc := &tokenizer.TokenizerConfig{ConfigJSON: raw}
	tc.TokenizerConfigJSON, _ = root.Manifest.ReadConfig("tokenizer_config.json")
	tc.GenerationConfigJSON, _ = root.Manifest.ReadConfig("generation_config.json")
	tok, err := tokenizer.LoadFromBytesWithConfig(data, tc)
	if err != nil {
		return nil, err
	}
	group, bits, mode := model.QuantizationParams(root.QuantType())
	if root.GroupSize() > 0 {
		group = root.GroupSize()
	}
	m := &Model{
		cfg: cfg, tok: tok, group: group, bits: bits, mode: mode, quant: root.AllTensorQuant(),
		Layers: make([]decoderLayer, cfg.Text.Layers), tokenSlot: -1,
		skipped: checkpointExclusions(cfg, root.Manifest.TensorLayers()),
	}
	if cfg.Vision != nil {
		if err := root.Manifest.ReadConfigJSON("processor_config.json", &m.processor); err != nil {
			return nil, err
		}
		img := m.processor.Image
		if len(img.Mean) != 3 || len(img.Std) != 3 || img.Scale <= 0 || img.Size.Min <= 0 || img.Size.Max < img.Size.Min ||
			img.Std[0] <= 0 || img.Std[1] <= 0 || img.Std[2] <= 0 {
			return nil, fmt.Errorf("invalid Hy4 Flash image processor configuration")
		}
	}
	if err := m.loadEngrams(root); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

func (m *Model) SkippedTensorNames() []string { return m.skipped }

// CheckpointSkippedTensorNames reports which tensors need no device allocation.
func CheckpointSkippedTensorNames(data []byte, layers []manifest.Layer) []string {
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil
	}
	return checkpointExclusions(cfg, layers)
}

func checkpointExclusions(cfg config, layers []manifest.Layer) []string {
	var names []string
	for _, id := range cfg.Text.EngramLayers {
		if id > 0 && id <= cfg.Text.Layers {
			names = append(names, engramTableName(id))
		}
	}
	for _, layer := range layers {
		if strings.HasPrefix(layer.Name, dflyPrefix+".") &&
			(!cfg.hasDFly() || strings.HasPrefix(layer.Name, dflyPrefix+".confidence_head.")) {
			names = append(names, layer.Name)
		}
	}
	return names
}

func (m *Model) LoadWeights(tensors map[string]*mlx.Array) error {
	cfg := &m.cfg.Text
	var missing error
	w := weights{tensors: tensors, linears: model.NewLinearFactory(tensors, m.group, m.bits, m.mode, m.quant), eps: cfg.RMSEps, err: &missing}
	m.Embed = model.MakeEmbeddingLayer(tensors, "model.embed_tokens", m.group, m.bits, m.mode, m.quant)
	if m.Embed == nil {
		return fmt.Errorf("missing model.embed_tokens.weight")
	}
	m.Head = loadOutputHead(w.tensor("lm_head.weight"))
	m.Norm = w.norm("model.norm")
	m.Reduce = hyperConnection{
		Projection: nn.NewLinear(w.tensor("model.hc_head.hc_head_fn").AsType(mlx.DTypeFloat32), nil),
		Bias:       w.tensor("model.hc_head.hc_head_base").AsType(mlx.DTypeFloat32),
		Scale:      w.tensor("model.hc_head.hc_head_scale").AsType(mlx.DTypeFloat32),
	}
	if m.cfg.Vision != nil {
		m.Vision = loadVision(w, m.cfg.Vision)
	}
	m.numCaches = len(m.Layers)
	for i := range m.Layers {
		if err := m.loadLayer(w, i); err != nil {
			return fmt.Errorf("load Hy4 Flash layer %d: %w", i, err)
		}
	}
	if len(cfg.EngramLayers) > 0 {
		m.tokenSlot = m.numCaches
		m.numCaches++
	}
	if missing == nil && m.cfg.hasDFly() {
		for name := range tensors {
			if strings.HasPrefix(name, dflyPrefix+".") {
				draft, err := m.loadDFly(w)
				if err != nil {
					return err
				}
				m.Draft = draft
				break
			}
		}
	}
	return missing
}

func (m *Model) loadLayer(w weights, i int) error {
	prefix := fmt.Sprintf("model.layers.%d", i)
	l, c := &m.Layers[i], &m.cfg.Text
	if c.AttentionTypes[i] == indexedAttention {
		l.Attention = loadAttention(w, prefix+".self_attn", c.Indexed, c.KVRank)
		l.Attention.Indexer = loadIndexer(w, prefix+".self_attn.compressed_indexer")
		l.Attention.indexSlot = m.numCaches
		m.numCaches++
	} else {
		l.Attention = loadAttention(w, prefix+".self_attn", c.Sliding, c.KVRank)
	}
	l.AttentionHC = loadHyperConnection(w, prefix+".hc_attn_layer.hc_pre.hc_")
	l.MLPHC = loadHyperConnection(w, prefix+".hc_mlp_layer.hc_pre.hc_")
	l.AttentionNorm = w.norm(prefix + ".input_layernorm")
	l.MLPNorm = w.norm(prefix + ".post_attention_layernorm")

	mlp := prefix + ".mlp"
	if c.MLPTypes[i] == "dense" {
		l.Dense = loadMLP(w, mlp)
		l.OutNorm = w.norm(mlp + ".post_fc2_norm")
	} else {
		gateUp, err := model.LoadExpertProjection(w.tensors, mlp+".experts.gate_up_proj", m.group, m.bits, m.mode, m.quant)
		if err != nil {
			return err
		}
		down, err := model.LoadExpertProjection(w.tensors, mlp+".experts.down_proj", m.group, m.bits, m.mode, m.quant)
		if err != nil {
			return err
		}
		l.MoE = &latentMoE{
			Router:     nn.NewLinear(w.tensor(mlp+".gate.weight").AsType(mlx.DTypeFloat32), nil),
			RouterBias: w.tensor(mlp + ".gate.e_score_correction_bias").AsType(mlx.DTypeFloat32),
			GateUp:     gateUp, Down: down,
			RouteDown: w.linear(mlp + ".routed_expert_down_proj"),
			RouteUp:   w.linear(mlp + ".routed_expert_up_proj"),
			Shared:    loadMLP(w, mlp+".shared_experts"),
		}
		l.OutNorm = w.norm(mlp + ".combine_output_norm")
	}
	if *w.err != nil {
		return *w.err
	}
	if l.Engram != nil {
		return l.Engram.loadWeights(w, prefix+".engram_layer", c)
	}
	return nil
}

func (m *Model) NewCaches() []cache.Cache {
	slots := make([]cache.Cache, m.numCaches)
	for i := range m.Layers {
		a := &m.Layers[i].Attention
		if a.Indexer == nil {
			slots[i] = cache.NewRotatingKVCache(m.cfg.Text.Window)
		} else {
			slots[i] = cache.NewKVCache()
			slots[a.indexSlot] = cache.NewKVCache()
		}
	}
	if m.tokenSlot >= 0 {
		slots[m.tokenSlot] = cache.NewKVCache()
	}
	return slots
}

func (m *Model) Forward(b *batch.Batch, slots []cache.Layer) (*mlx.Array, *mlx.Array) {
	c := &m.cfg.Text
	B, L := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	positions := rotaryPositions{batch: b}
	embedded := m.Embed.Forward(b.InputIDs)
	if len(b.Media) > 0 {
		embedded = m.scatterImageFeatures(embedded, b)
	}
	hidden := mlx.BroadcastTo(embedded.ExpandDims(2), int32(B), int32(L), int32(c.Streams), int32(c.HiddenSize))
	var tokens [][]int32
	if m.tokenSlot >= 0 {
		var slot cache.Layer
		if len(slots) != 0 {
			slot = slots[m.tokenSlot]
		}
		tokens = engramTokens(b, slot, c.EngramOrder)
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		if l.Engram != nil {
			hidden = mlx.Add(hidden, l.Engram.forward(hidden, tokens, b, &m.cfg))
		}
		input, post := l.AttentionHC.prepare(hidden, c)
		input = l.AttentionNorm.Forward(input, 0)
		delta := l.Attention.forward(input, b, slots, i, positions, c)
		hidden = injectStream(hidden, delta, post)
		input, post = l.MLPHC.prepare(hidden, c)
		input = l.MLPNorm.Forward(input, 0)
		if l.MoE == nil {
			delta = l.Dense.forward(input)
		} else {
			delta = l.MoE.forward(input, c)
		}
		hidden = injectStream(hidden, l.OutNorm.Forward(delta, 0), post)
	}
	out := m.Norm.Forward(m.Reduce.reduce(hidden, c), 0)
	if m.Draft != nil {
		return out, m.Draft.conditioning(b, out)
	}
	return out, out
}

func (m *Model) Unembed(x *mlx.Array) *mlx.Array {
	return m.Head.Forward(x.AsType(mlx.DTypeFloat32))
}

func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }
func (m *Model) MaxContextLength() int           { return m.cfg.Text.Context }

func (m *Model) Close() error {
	var err error
	for i := range m.Layers {
		if e := m.Layers[i].Engram; e != nil {
			err = errors.Join(err, e.table.Close())
			m.Layers[i].Engram = nil
		}
	}
	return err
}
