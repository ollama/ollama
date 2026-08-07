package gemma4

// Vision support for Gemma 4 MLX checkpoints, ported from mlx_vlm (main
// branch) models/gemma4/vision.py and models/gemma4_unified. Two lineages
// share this file:
//
//   - gemma4_unified_vision (12b): an encoder-free embedder — LayerNorm →
//     Linear over flat 48px patches → LayerNorm → learned per-axis position
//     embeddings → LayerNorm.
//   - gemma4_vision (26b/31b): a 27-layer encoder over 16px patches with 2D
//     RoPE attention (scale 1.0), 3×3 average pooling and standardization.
//
// Both project into the text hidden size through embed_vision: a weightless
// RMSNorm followed by a linear. Image sizing follows ADR 0008's budget-fill
// ladder via llm.BudgetFillSize — grids are always exact 48-multiples, so the
// padding/masking paths of the reference are deliberately not ported.

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

// VisionConfig holds the vision_config block of a multimodal checkpoint.
type VisionConfig struct {
	ModelType string `json:"model_type"`

	// Unified embedder (12b).
	MMEmbedDim     int32 `json:"mm_embed_dim"`
	MMPosembSize   int32 `json:"mm_posemb_size"`
	ModelPatchSize int32 `json:"model_patch_size"`
	OutputProjDims int32 `json:"output_proj_dims"`

	// Full tower (26b/31b).
	HiddenSize            int32 `json:"hidden_size"`
	IntermediateSize      int32 `json:"intermediate_size"`
	NumAttentionHeads     int32 `json:"num_attention_heads"`
	HeadDim               int32 `json:"head_dim"`
	NumHiddenLayers       int32 `json:"num_hidden_layers"`
	PositionEmbeddingSize int32 `json:"position_embedding_size"`
	Standardize           bool  `json:"standardize"`
	RopeParameters        struct {
		RopeTheta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`

	// Shared.
	PatchSize         int32   `json:"patch_size"`
	PoolingKernelSize int32   `json:"pooling_kernel_size"`
	RMSNormEps        float32 `json:"rms_norm_eps"`
}

// IsUnified reports the encoder-free 12b lineage.
func (c *VisionConfig) IsUnified() bool { return c.ModelType == "gemma4_unified_vision" }

// multimodalTokens are the top-level special token ids bracketing one
// image's soft tokens in the prompt.
type multimodalTokens struct {
	BOI   int32 `json:"boi_token_id"`
	EOI   int32 `json:"eoi_token_id"`
	Image int32 `json:"image_token_id"`
}

// parseVisionConfig reads vision_config and the top-level multimodal token
// ids from the raw config bytes. It is a separate unmarshal on purpose:
// parseTextConfig replaces its result wholesale with the nested text_config,
// which would silently discard these top-level keys.
func parseVisionConfig(configData []byte) (*VisionConfig, multimodalTokens, error) {
	var wrapped struct {
		multimodalTokens
		VisionConfig *VisionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(configData, &wrapped); err != nil {
		return nil, multimodalTokens{}, fmt.Errorf("parse vision config: %w", err)
	}
	cfg := wrapped.VisionConfig
	if cfg == nil {
		return nil, wrapped.multimodalTokens, nil
	}
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 16
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = 3
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.RopeParameters.RopeTheta == 0 {
		cfg.RopeParameters.RopeTheta = 100
	}
	if cfg.IsUnified() && cfg.ModelPatchSize == 0 {
		cfg.ModelPatchSize = cfg.PatchSize * cfg.PoolingKernelSize
	}
	return cfg, wrapped.multimodalTokens, nil
}

// VisionEmbedder is the encoder-free unified path (12b).
type VisionEmbedder struct {
	PatchLN1   *nn.LayerNorm
	PatchDense nn.LinearLayer
	PatchLN2   *nn.LayerNorm
	PosEmbX    *mlx.Array // [MMPosembSize, MMEmbedDim] — pos_embedding[:, 0, :]
	PosEmbY    *mlx.Array // [MMPosembSize, MMEmbedDim] — pos_embedding[:, 1, :]
	PosNorm    *nn.LayerNorm
}

// VisionAttention is one tower block's attention: per-head q/k RMSNorm with
// weights, weightless v RMSNorm, 2D RoPE, SDPA at scale 1.0.
type VisionAttention struct {
	QProj, KProj, VProj, OProj nn.LinearLayer
	QNormWeight, KNormWeight   *mlx.Array // [HeadDim]
}

// VisionLayer is one sandwich-norm tower block. Norm weights apply directly
// (no +1 shift, matching the reference's plain rms_norm).
type VisionLayer struct {
	InputNorm, PostAttnNorm, PreFFNorm, PostFFNorm *mlx.Array
	Attn                                           *VisionAttention
	GateProj, UpProj, DownProj                     nn.LinearLayer
}

// VisionTower is the shared 27-layer encoder (26b/31b).
type VisionTower struct {
	InputProj  nn.LinearLayer
	PosTableX  *mlx.Array // [PositionEmbeddingSize, HiddenSize] — table[0]
	PosTableY  *mlx.Array // [PositionEmbeddingSize, HiddenSize] — table[1]
	Layers     []*VisionLayer
	NegStdBias *mlx.Array // -std_bias, precomputed; nil unless Standardize
	StdScale   *mlx.Array
}

// resolveVisionPrefix probes the spellings vision tensors ship under.
func resolveVisionPrefix(tensors map[string]*mlx.Array, marker string) (string, bool) {
	for _, prefix := range []string{"model.", ""} {
		if tensors[prefix+marker] != nil {
			return prefix, true
		}
	}
	return "", false
}

// materialize detaches a lazy view (slice/negate of a loaded tensor) into
// its own buffer so the parent blob can be released.
func materialize(a *mlx.Array) *mlx.Array {
	c := a.Clone()
	mlx.Eval(c)
	return c
}

func visionLayerNorm(tensors map[string]*mlx.Array, path string) (*nn.LayerNorm, error) {
	w := tensors[path+".weight"]
	b := tensors[path+".bias"]
	if w == nil || b == nil {
		return nil, fmt.Errorf("missing vision layer norm: %s", path)
	}
	// Eps 0 defers to nn.LayerNorm's 1e-5 default, matching the reference's
	// torch/mlx nn.LayerNorm defaults (distinct from the 1e-6 rms_norm_eps).
	return &nn.LayerNorm{Weight: w, Bias: b}, nil
}

// loadVisionWeights binds the vision tensors for whichever lineage the
// checkpoint ships. Called from LoadWeights when the config carries a
// vision_config.
func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	cfg := m.VisionCfg

	projPrefix, ok := resolveVisionPrefix(tensors, "embed_vision.embedding_projection.weight")
	if !ok {
		return fmt.Errorf("missing vision projection: embed_vision.embedding_projection.weight")
	}
	m.EmbedVisionProj = linears.Make(projPrefix + "embed_vision.embedding_projection")
	if m.EmbedVisionProj == nil {
		return fmt.Errorf("missing vision projection: %sembed_vision.embedding_projection.weight", projPrefix)
	}

	if cfg.IsUnified() {
		prefix, ok := resolveVisionPrefix(tensors, "vision_embedder.pos_embedding")
		if !ok {
			return fmt.Errorf("missing vision embedder: vision_embedder.pos_embedding")
		}
		p := prefix + "vision_embedder."

		e := &VisionEmbedder{}
		var err error
		if e.PatchLN1, err = visionLayerNorm(tensors, p+"patch_ln1"); err != nil {
			return err
		}
		if e.PatchDense = linears.Make(p + "patch_dense"); e.PatchDense == nil {
			return fmt.Errorf("missing vision embedder weight: %spatch_dense.weight", p)
		}
		if e.PatchLN2, err = visionLayerNorm(tensors, p+"patch_ln2"); err != nil {
			return err
		}
		if e.PosNorm, err = visionLayerNorm(tensors, p+"pos_norm"); err != nil {
			return err
		}
		pos := tensors[p+"pos_embedding"] // [MMPosembSize, 2, MMEmbedDim]
		if pos == nil {
			return fmt.Errorf("missing vision embedder weight: %spos_embedding", p)
		}
		e.PosEmbX = materialize(pos.Slice(mlx.Slice(), mlx.Slice(0, 1), mlx.Slice()).Squeeze(1))
		e.PosEmbY = materialize(pos.Slice(mlx.Slice(), mlx.Slice(1, 2), mlx.Slice()).Squeeze(1))
		m.VisionEmbedder = e
		return nil
	}

	prefix, ok := resolveVisionPrefix(tensors, "vision_tower.patch_embedder.input_proj.weight")
	if !ok {
		return fmt.Errorf("missing vision tower: vision_tower.patch_embedder.input_proj.weight")
	}
	p := prefix + "vision_tower."

	t := &VisionTower{}
	if t.InputProj = linears.Make(p + "patch_embedder.input_proj"); t.InputProj == nil {
		return fmt.Errorf("missing vision tower weight: %spatch_embedder.input_proj.weight", p)
	}
	table := tensors[p+"patch_embedder.position_embedding_table"] // [2, S, HiddenSize]
	if table == nil {
		return fmt.Errorf("missing vision tower weight: %spatch_embedder.position_embedding_table", p)
	}
	t.PosTableX = materialize(table.Slice(mlx.Slice(0, 1), mlx.Slice(), mlx.Slice()).Squeeze(0))
	t.PosTableY = materialize(table.Slice(mlx.Slice(1, 2), mlx.Slice(), mlx.Slice()).Squeeze(0))

	if cfg.Standardize {
		stdBias := tensors[p+"std_bias"]
		t.StdScale = tensors[p+"std_scale"]
		if stdBias == nil || t.StdScale == nil {
			return fmt.Errorf("missing vision tower standardization: %sstd_bias / %sstd_scale", p, p)
		}
		t.NegStdBias = materialize(mlx.MulScalar(stdBias, -1))
	}

	t.Layers = make([]*VisionLayer, cfg.NumHiddenLayers)
	for i := range t.Layers {
		lp := fmt.Sprintf("%sencoder.layers.%d.", p, i)
		l := &VisionLayer{Attn: &VisionAttention{}}

		for _, norm := range []struct {
			dst  **mlx.Array
			name string
		}{
			{&l.InputNorm, "input_layernorm"},
			{&l.PostAttnNorm, "post_attention_layernorm"},
			{&l.PreFFNorm, "pre_feedforward_layernorm"},
			{&l.PostFFNorm, "post_feedforward_layernorm"},
			{&l.Attn.QNormWeight, "self_attn.q_norm"},
			{&l.Attn.KNormWeight, "self_attn.k_norm"},
		} {
			w := tensors[lp+norm.name+".weight"]
			if w == nil {
				return fmt.Errorf("missing vision tower norm: %s%s.weight", lp, norm.name)
			}
			*norm.dst = w
		}

		// ClippableLinear wraps the projection as `.linear` in the checkpoint;
		// use_clipped_linears is false so no clip bounds ship alongside.
		for _, proj := range []struct {
			dst  *nn.LinearLayer
			name string
		}{
			{&l.Attn.QProj, "self_attn.q_proj.linear"},
			{&l.Attn.KProj, "self_attn.k_proj.linear"},
			{&l.Attn.VProj, "self_attn.v_proj.linear"},
			{&l.Attn.OProj, "self_attn.o_proj.linear"},
			{&l.GateProj, "mlp.gate_proj.linear"},
			{&l.UpProj, "mlp.up_proj.linear"},
			{&l.DownProj, "mlp.down_proj.linear"},
		} {
			if *proj.dst = linears.Make(lp + proj.name); *proj.dst == nil {
				return fmt.Errorf("missing vision tower weight: %s%s.weight", lp, proj.name)
			}
		}
		t.Layers[i] = l
	}
	m.VisionTower = t
	return nil
}
