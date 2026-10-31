package mlxrunner

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// MemoryEstimate is the memory an MLX runner is predicted to reach once its
// caches have filled for the context it serves. The runner reports only what
// it holds right now, which at load time is the weights; the scheduler books
// the runner at this estimate instead so the cache has room to grow.
type MemoryEstimate struct {
	// Weights is the size of the tensor blobs in the manifest.
	Weights uint64
	// KVCache is the attention cache at NumCtx tokens.
	KVCache uint64
	// Recurrent is the fixed state of linear-attention and SSM layers.
	Recurrent uint64
	// PrefixCache is the budget for paged-out prefix snapshots.
	PrefixCache uint64
	// NumCtx is the context the cache terms were sized for.
	NumCtx int
}

// Total is the memory to book for the runner.
func (e MemoryEstimate) Total() uint64 {
	return e.Weights + e.KVCache + e.Recurrent + e.PrefixCache
}

// kvConfig is the part of a safetensors config.json that sizes the caches.
// Each architecture parses its own config in the runner process; these are
// the fields they share, read here so the scheduler can size the caches
// before the runner exists.
type kvConfig struct {
	HiddenSize            int    `json:"hidden_size"`
	NumHiddenLayers       int    `json:"num_hidden_layers"`
	NumAttentionHeads     int    `json:"num_attention_heads"`
	NumKeyValueHeads      int    `json:"num_key_value_heads"`
	HeadDim               int    `json:"head_dim"`
	GlobalHeadDim         int    `json:"global_head_dim"`
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	TorchDtype            string `json:"torch_dtype"`
	Dtype                 string `json:"dtype"`

	// Layer layout of hybrid and sliding-window architectures.
	LayerTypes            []string `json:"layer_types"`
	LayersBlockType       []string `json:"layers_block_type"`
	HybridOverridePattern string   `json:"hybrid_override_pattern"`
	FullAttentionInterval int      `json:"full_attention_interval"`
	SlidingWindow         int      `json:"sliding_window"`
	SlidingWindowPattern  int      `json:"sliding_window_pattern"`
	NumKVSharedLayers     int      `json:"num_kv_shared_layers"`

	// Gated delta-net state (qwen3_5, qwen4_exp).
	LinearNumValueHeads int `json:"linear_num_value_heads"`
	LinearNumKeyHeads   int `json:"linear_num_key_heads"`
	LinearKeyHeadDim    int `json:"linear_key_head_dim"`
	LinearValueHeadDim  int `json:"linear_value_head_dim"`
	LinearConvKernelDim int `json:"linear_conv_kernel_dim"`

	// Mamba state (nemotron_h).
	ConvKernel    int `json:"conv_kernel"`
	SSMStateSize  int `json:"ssm_state_size"`
	MambaNumHeads int `json:"mamba_num_heads"`
	MambaHeadDim  int `json:"mamba_head_dim"`
	NGroups       int `json:"n_groups"`

	// Short convolution state (lfm3).
	ConvLCache int `json:"conv_L_cache"`

	// Multi-head latent attention (glm4_moe_lite): the cache holds the
	// compressed KV and the rope keys as one head.
	KVLoraRank    int `json:"kv_lora_rank"`
	QKRopeHeadDim int `json:"qk_rope_head_dim"`
}

type layerKind int

const (
	layerNoCache layerKind = iota
	layerFullAttention
	layerSlidingAttention
	layerRecurrent
)

// PredictMemory estimates the memory the MLX runner for modelName reaches
// serving numCtx tokens of context. numCtx is capped at the model's own
// context length, as the runner caps it; zero means the model's length.
func PredictMemory(modelName string, numCtx int) (MemoryEstimate, error) {
	m, err := manifest.ParseNamedManifest(model.ParseName(modelName))
	if err != nil {
		return MemoryEstimate{}, err
	}

	configData, err := m.ReadConfig("config.json")
	if err != nil {
		return MemoryEstimate{}, err
	}

	return predictMemory(m, configData, numCtx)
}

func predictMemory(m *manifest.Manifest, configData []byte, numCtx int) (MemoryEstimate, error) {
	cfg, err := parseKVConfig(configData)
	if err != nil {
		return MemoryEstimate{}, err
	}

	var estimate MemoryEstimate
	for _, layer := range m.TensorLayers() {
		estimate.Weights += uint64(layer.Size)
	}

	estimate.NumCtx = numCtx
	if cfg.MaxPositionEmbeddings > 0 && (numCtx <= 0 || numCtx > cfg.MaxPositionEmbeddings) {
		estimate.NumCtx = cfg.MaxPositionEmbeddings
	}

	for _, kind := range cfg.layerKinds() {
		switch kind {
		case layerFullAttention:
			estimate.KVCache += cfg.attentionBytes(estimate.NumCtx, cfg.fullHeadDim())
		case layerSlidingAttention:
			estimate.KVCache += cfg.attentionBytes(min(estimate.NumCtx, cfg.SlidingWindow), cfg.headDim())
		case layerRecurrent:
			estimate.Recurrent += cfg.recurrentBytes()
		}
	}

	// Paged-out prefix snapshots are copies of cache state, so a model
	// without caches has nothing to page out.
	if estimate.KVCache+estimate.Recurrent > 0 {
		estimate.PrefixCache = uint64(maxPagedOutBytes)
	}

	return estimate, nil
}

// parseKVConfig reads the text model's fields. Multimodal checkpoints nest
// them under text_config (or llm_config), where they take precedence over
// the top level.
func parseKVConfig(configData []byte) (kvConfig, error) {
	var cfg kvConfig
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return kvConfig{}, fmt.Errorf("parse config.json: %w", err)
	}

	var nested struct {
		TextConfig json.RawMessage `json:"text_config"`
		LLMConfig  json.RawMessage `json:"llm_config"`
	}
	if err := json.Unmarshal(configData, &nested); err != nil {
		return kvConfig{}, fmt.Errorf("parse config.json: %w", err)
	}

	for _, raw := range [][]byte{nested.TextConfig, nested.LLMConfig} {
		if len(raw) == 0 || string(raw) == "null" {
			continue
		}
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return kvConfig{}, fmt.Errorf("parse nested text config: %w", err)
		}
	}

	return cfg, nil
}

// layerKinds returns what each layer caches, following the rules the
// architectures use to pick a cache per layer: an explicit per-layer list
// first, then a repeating pattern, and plain full attention otherwise.
func (cfg *kvConfig) layerKinds() []layerKind {
	kinds := make([]layerKind, cfg.NumHiddenLayers)
	for i := range kinds {
		kinds[i] = cfg.layerKind(i)
	}

	// Gemma 4's last num_kv_shared_layers layers reuse an earlier layer's
	// cache and allocate none of their own.
	if len(cfg.LayerTypes) > 0 {
		for i := cfg.NumHiddenLayers - cfg.NumKVSharedLayers; i >= 0 && i < len(kinds); i++ {
			kinds[i] = layerNoCache
		}
	}

	return kinds
}

func (cfg *kvConfig) layerKind(i int) layerKind {
	if len(cfg.LayerTypes) == cfg.NumHiddenLayers {
		return layerKindFromName(cfg.LayerTypes[i])
	}

	if len(cfg.LayersBlockType) == cfg.NumHiddenLayers {
		return layerKindFromName(cfg.LayersBlockType[i])
	}

	if len(cfg.HybridOverridePattern) == cfg.NumHiddenLayers {
		return layerKindFromPattern(cfg.HybridOverridePattern[i])
	}

	if cfg.FullAttentionInterval > 0 {
		if (i+1)%cfg.FullAttentionInterval == 0 {
			return layerFullAttention
		}
		return layerRecurrent
	}

	if cfg.SlidingWindow > 0 && cfg.SlidingWindowPattern > 0 {
		if (i+1)%cfg.SlidingWindowPattern == 0 {
			return layerFullAttention
		}
		return layerSlidingAttention
	}

	return layerFullAttention
}

// layerKindFromName maps a layer_types or layers_block_type entry. Names
// vary by family (full_attention, attention, linear_attention, mamba,
// conv, sliding_attention, mlp, moe); an unknown name is sized as full
// attention, the largest cache.
func layerKindFromName(name string) layerKind {
	name = strings.ToLower(name)
	switch {
	case strings.Contains(name, "sliding"):
		return layerSlidingAttention
	case strings.Contains(name, "linear"), strings.Contains(name, "mamba"),
		strings.Contains(name, "conv"), strings.Contains(name, "recurrent"):
		return layerRecurrent
	case name == "mlp", name == "moe":
		return layerNoCache
	default:
		return layerFullAttention
	}
}

// layerKindFromPattern maps a hybrid_override_pattern character: M is
// Mamba, * and A are attention, - and E are MLP or MoE blocks.
func layerKindFromPattern(c byte) layerKind {
	switch c {
	case 'M':
		return layerRecurrent
	case '*', 'A':
		return layerFullAttention
	default:
		return layerNoCache
	}
}

func (cfg *kvConfig) headDim() int {
	if cfg.HeadDim > 0 {
		return cfg.HeadDim
	}
	if cfg.NumAttentionHeads > 0 {
		return cfg.HiddenSize / cfg.NumAttentionHeads
	}
	return 0
}

// fullHeadDim is the head size of full-attention layers, which Gemma 4 sets
// apart from its sliding layers.
func (cfg *kvConfig) fullHeadDim() int {
	if cfg.GlobalHeadDim > 0 {
		return cfg.GlobalHeadDim
	}
	return cfg.headDim()
}

func (cfg *kvConfig) kvHeads() int {
	if cfg.NumKeyValueHeads > 0 {
		return cfg.NumKeyValueHeads
	}
	return cfg.NumAttentionHeads
}

// elementSize is the size of a cache element. The caches take the
// activation dtype, which is the checkpoint's dtype; MLX runs these models
// in 16-bit unless the checkpoint is float32.
func (cfg *kvConfig) elementSize() uint64 {
	dtype := cfg.Dtype
	if dtype == "" {
		dtype = cfg.TorchDtype
	}
	if strings.EqualFold(dtype, "float32") {
		return 4
	}
	return 2
}

// attentionBytes is one layer's keys and values for tokens of context.
func (cfg *kvConfig) attentionBytes(tokens, headDim int) uint64 {
	if tokens <= 0 {
		return 0
	}

	// MLA keeps a single compressed head holding the latent KV and the
	// rope keys; the values are a view of the same tensor.
	if cfg.KVLoraRank > 0 {
		return uint64(tokens) * uint64(cfg.KVLoraRank+cfg.QKRopeHeadDim) * cfg.elementSize()
	}

	perToken := uint64(2) * uint64(cfg.kvHeads()) * uint64(headDim) * cfg.elementSize()
	return uint64(tokens) * perToken
}

// recurrentBytes is one layer's fixed state: a convolution tail in the
// activation dtype and a float32 recurrent state, mirroring
// cache.NewRecurrentCache's arguments in each architecture.
func (cfg *kvConfig) recurrentBytes() uint64 {
	var convTail, convDim, stateElems int
	switch {
	case cfg.LinearConvKernelDim > 0:
		convTail = cfg.LinearConvKernelDim - 1
		convDim = 2*cfg.LinearNumKeyHeads*cfg.LinearKeyHeadDim + cfg.LinearNumValueHeads*cfg.LinearValueHeadDim
		stateElems = cfg.LinearNumValueHeads * cfg.LinearValueHeadDim * cfg.LinearKeyHeadDim
	case cfg.ConvKernel > 0:
		convTail = cfg.ConvKernel - 1
		convDim = cfg.MambaNumHeads*cfg.MambaHeadDim + 2*cfg.NGroups*cfg.SSMStateSize
		stateElems = cfg.MambaNumHeads * cfg.MambaHeadDim * cfg.SSMStateSize
	case cfg.ConvLCache > 0:
		convTail = cfg.ConvLCache - 1
		convDim = cfg.HiddenSize
		stateElems = 1
	default:
		return 0
	}

	return uint64(convTail)*uint64(convDim)*cfg.elementSize() + uint64(stateElems)*4
}
