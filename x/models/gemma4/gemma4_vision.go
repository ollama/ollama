// Vision encoder for Gemma 4 multimodal models.
package gemma4

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

// VisionConfig holds configuration for the vision encoder.
type VisionConfig struct {
	ModelType             string          `json:"model_type"`
	HiddenSize            int32           `json:"hidden_size"`             // 768
	IntermediateSize      int32           `json:"intermediate_size"`       // 3072
	NumHiddenLayers       int32           `json:"num_hidden_layers"`       // 16
	NumAttentionHeads     int32           `json:"num_attention_heads"`     // 12
	NumKeyValueHeads      int32           `json:"num_key_value_heads"`     // 12
	HeadDim               int32           `json:"head_dim"`                // 64
	PatchSize             int32           `json:"patch_size"`              // 16
	PositionEmbeddingSize int32           `json:"position_embedding_size"` // 10240
	ImageSeqLength        int32           `json:"image_seq_length"`        // 280
	DefaultOutputLength   int32           `json:"default_output_length"`
	PoolingKernelSize     int32           `json:"pooling_kernel_size"` // 3
	RMSNormEps            float32         `json:"rms_norm_eps"`        // 1e-6
	Standardize           bool            `json:"standardize"`
	RopeParameters        json.RawMessage `json:"rope_parameters"`
	LayerTypes            []string        `json:"layer_types"`

	// Unified vision uses a transformer-less patch embedder with different
	// names for the equivalent dimensions.
	MMEmbeddingDim          int32 `json:"mm_embed_dim"`
	MMPositionEmbeddingSize int32 `json:"mm_posemb_size"`
	ModelPatchSize          int32 `json:"model_patch_size"`
	NumSoftTokens           int32 `json:"num_soft_tokens"`

	// Computed fields.
	RopeTheta float32 `json:"-"` // from rope_parameters
	Scale     float32 `json:"-"` // 1/sqrt(head_dim)
}

// ClipBounds holds input/output clamp bounds for a clippable linear layer.
type ClipBounds struct {
	InputMin  *mlx.Array
	InputMax  *mlx.Array
	OutputMin *mlx.Array
	OutputMax *mlx.Array
}

// ClippableLinear wraps a linear layer with optional input/output clamping.
type ClippableLinear struct {
	Linear nn.LinearLayer
	Clip   *ClipBounds
}

func (cl *ClippableLinear) Forward(x *mlx.Array) *mlx.Array {
	if cl.Clip != nil && cl.Clip.InputMin != nil && cl.Clip.InputMax != nil {
		x = mlx.Clip(x, cl.Clip.InputMin, cl.Clip.InputMax)
	}
	x = cl.Linear.Forward(x)
	if cl.Clip != nil && cl.Clip.OutputMin != nil && cl.Clip.OutputMax != nil {
		x = mlx.Clip(x, cl.Clip.OutputMin, cl.Clip.OutputMax)
	}
	return x
}

// VisionAttention implements multi-head attention for the vision encoder.
type VisionAttention struct {
	QProj *ClippableLinear
	KProj *ClippableLinear
	VProj *ClippableLinear
	OProj *ClippableLinear

	QNorm *nn.RMSNorm
	KNorm *nn.RMSNorm

	// Norm weight for Q/K RMSNorm.
	QNormScaled *mlx.Array
	KNormScaled *mlx.Array
}

// VisionMLP is the feed-forward network for vision layers.
type VisionMLP struct {
	GateProj *ClippableLinear
	UpProj   *ClippableLinear
	DownProj *ClippableLinear
}

// VisionDecoderLayer is a single vision transformer block.
type VisionDecoderLayer struct {
	InputNorm    *nn.RMSNorm
	Attention    *VisionAttention
	PostAttnNorm *nn.RMSNorm
	PreFFNorm    *nn.RMSNorm
	MLP          *VisionMLP
	PostFFNorm   *nn.RMSNorm

	// Layer scalar (vision layers have this too).
	LayerScalar *mlx.Array

	// Norm weight for RMSNorm.
	InputNormScaled    *mlx.Array
	PostAttnNormScaled *mlx.Array
	PreFFNormScaled    *mlx.Array
	PostFFNormScaled   *mlx.Array
}

// VisionPatchEmbedder converts images to patch embeddings with 2D position info.
type VisionPatchEmbedder struct {
	InputProj              nn.LinearLayer // Linear(3*patch_size^2, hidden_size)
	PositionEmbeddingTable *mlx.Array     // [2, position_embedding_size, hidden_size]
}

// UnifiedVisionPatchEmbedder implements Gemma4's transformer-less vision path.
type UnifiedVisionPatchEmbedder struct {
	PatchNorm1        *nn.LayerNorm
	PatchDense        nn.LinearLayer
	PatchNorm2        *nn.LayerNorm
	PositionEmbedding *mlx.Array
	PositionNorm      *nn.LayerNorm
}

// VisionEncoder orchestrates the full vision encoding pipeline.
type VisionEncoder struct {
	PatchEmbedder        *VisionPatchEmbedder
	UnifiedPatchEmbedder *UnifiedVisionPatchEmbedder
	Layers               []*VisionDecoderLayer
	Cfg                  *VisionConfig
	StdBias              *mlx.Array
	StdScale             *mlx.Array
}

// MultimodalEmbedder projects vision tokens into language model space.
type MultimodalEmbedder struct {
	EmbeddingProjection nn.LinearLayer // Linear(multimodal_hidden, text_hidden)
	Eps                 float32        // RMSNorm epsilon
}

// Forward normalizes and projects multimodal embeddings.
func (me *MultimodalEmbedder) Forward(x *mlx.Array) *mlx.Array {
	x = mlx.RMSNormFn(x, nil, me.Eps)
	return me.EmbeddingProjection.Forward(x)
}

// makeClippableLinear creates a ClippableLinear from a linear layer and optional clip bounds.
// ClippableLinear wraps nn.Linear as self.linear, so the weight path is
// prefix.linear.weight (e.g., self_attn.q_proj.linear.weight).
func makeClippableLinear(linears model.LinearFactory, tensors map[string]*mlx.Array, prefix string) *ClippableLinear {
	linear := linears.Make(prefix + ".linear")
	if linear == nil {
		return nil
	}
	cl := &ClippableLinear{Linear: linear}

	// Load clip bounds if present.
	inputMin := tensors[prefix+".input_min"]
	inputMax := tensors[prefix+".input_max"]
	outputMin := tensors[prefix+".output_min"]
	outputMax := tensors[prefix+".output_max"]

	if inputMin != nil || outputMin != nil {
		cl.Clip = &ClipBounds{
			InputMin:  inputMin,
			InputMax:  inputMax,
			OutputMin: outputMin,
			OutputMax: outputMax,
		}
	}
	return cl
}

func parseVisionConfig(configData []byte) (*VisionConfig, error) {
	var wrapper struct {
		VisionConfig *VisionConfig `json:"vision_config"`
	}
	if err := json.Unmarshal(configData, &wrapper); err != nil {
		return nil, fmt.Errorf("parse vision config: %w", err)
	}
	if wrapper.VisionConfig == nil {
		return nil, nil // No vision config means a text-only model.
	}

	cfg := wrapper.VisionConfig
	switch cfg.ModelType {
	case "", "gemma4_vision":
	case "gemma4_unified_vision":
		if cfg.ModelPatchSize == 0 {
			cfg.ModelPatchSize = cfg.PatchSize * cfg.PoolingKernelSize
		}
		cfg.PatchSize = cfg.ModelPatchSize
		cfg.PoolingKernelSize = 1
		cfg.HiddenSize = cfg.MMEmbeddingDim
		cfg.PositionEmbeddingSize = cfg.MMPositionEmbeddingSize
		cfg.ImageSeqLength = cfg.NumSoftTokens
	default:
		// The manifest advertises vision whenever a vision_config is present
		// (x/create/client/create.go), so degrading to text-only here would
		// leave the capability claiming a tower we cannot run.
		return nil, fmt.Errorf("unsupported vision tower %q", cfg.ModelType)
	}

	// Defaults.
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 64
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.PatchSize == 0 {
		cfg.PatchSize = 16
	}
	if cfg.PoolingKernelSize == 0 {
		cfg.PoolingKernelSize = 3
	}
	if cfg.ImageSeqLength == 0 {
		cfg.ImageSeqLength = cfg.DefaultOutputLength
		if cfg.ImageSeqLength == 0 {
			cfg.ImageSeqLength = 280
		}
	}
	if cfg.PositionEmbeddingSize == 0 {
		cfg.PositionEmbeddingSize = 10240
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}

	cfg.Scale = 1.0 // Gemma 4 vision uses scale=1.0 (Q/K norms handle magnitude)

	// Extract RoPE theta from rope_parameters: {"rope_theta": 100.0, "rope_type": "default"}
	cfg.RopeTheta = 100.0
	if len(cfg.RopeParameters) > 0 {
		var rp struct {
			RopeTheta float32 `json:"rope_theta"`
		}
		if json.Unmarshal(cfg.RopeParameters, &rp) == nil && rp.RopeTheta > 0 {
			cfg.RopeTheta = rp.RopeTheta
		}
	}

	return cfg, nil
}

// loadVisionWeights loads vision encoder and multimodal embedder weights.
func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	if m.VisionConfig.ModelType == "gemma4_unified_vision" {
		return m.loadUnifiedVisionWeights(tensors, linears)
	}

	prefix := "model."

	// Resolve prefix: try "model.vision_tower." first, then "vision_tower.".
	visionPrefix := prefix + "vision_tower."
	if tensors[visionPrefix+"patch_embedder.input_proj.weight"] == nil {
		visionPrefix = "vision_tower."
		if tensors[visionPrefix+"patch_embedder.input_proj.weight"] == nil {
			return fmt.Errorf("missing vision tower weights")
		}
	}

	// Patch embedder.
	m.VisionEncoder.PatchEmbedder = &VisionPatchEmbedder{}
	m.VisionEncoder.PatchEmbedder.InputProj = linears.Make(visionPrefix + "patch_embedder.input_proj")
	if m.VisionEncoder.PatchEmbedder.InputProj == nil {
		return fmt.Errorf("missing vision patch embedder input_proj")
	}
	m.VisionEncoder.PatchEmbedder.PositionEmbeddingTable = tensors[visionPrefix+"patch_embedder.position_embedding_table"]
	if m.VisionEncoder.PatchEmbedder.PositionEmbeddingTable == nil {
		return fmt.Errorf("missing vision patch embedder position_embedding_table")
	}

	// Vision transformer layers.
	vcfg := m.VisionConfig
	m.VisionEncoder.Layers = make([]*VisionDecoderLayer, vcfg.NumHiddenLayers)
	for i := range vcfg.NumHiddenLayers {
		layerPrefix := fmt.Sprintf("%sencoder.layers.%d", visionPrefix, i)

		layer := &VisionDecoderLayer{
			Attention: &VisionAttention{},
			MLP:       &VisionMLP{},
		}

		// Norms.
		if w := tensors[layerPrefix+".input_layernorm.weight"]; w != nil {
			layer.InputNorm = nn.NewRMSNorm(w, vcfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_attention_layernorm.weight"]; w != nil {
			layer.PostAttnNorm = nn.NewRMSNorm(w, vcfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".pre_feedforward_layernorm.weight"]; w != nil {
			layer.PreFFNorm = nn.NewRMSNorm(w, vcfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".post_feedforward_layernorm.weight"]; w != nil {
			layer.PostFFNorm = nn.NewRMSNorm(w, vcfg.RMSNormEps)
		}

		// Attention projections (with clip bounds).
		layer.Attention.QProj = makeClippableLinear(linears, tensors, layerPrefix+".self_attn.q_proj")
		layer.Attention.KProj = makeClippableLinear(linears, tensors, layerPrefix+".self_attn.k_proj")
		layer.Attention.VProj = makeClippableLinear(linears, tensors, layerPrefix+".self_attn.v_proj")
		layer.Attention.OProj = makeClippableLinear(linears, tensors, layerPrefix+".self_attn.o_proj")

		if w := tensors[layerPrefix+".self_attn.q_norm.weight"]; w != nil {
			layer.Attention.QNorm = nn.NewRMSNorm(w, vcfg.RMSNormEps)
		}
		if w := tensors[layerPrefix+".self_attn.k_norm.weight"]; w != nil {
			layer.Attention.KNorm = nn.NewRMSNorm(w, vcfg.RMSNormEps)
		}

		// MLP (with clip bounds).
		layer.MLP.GateProj = makeClippableLinear(linears, tensors, layerPrefix+".mlp.gate_proj")
		layer.MLP.UpProj = makeClippableLinear(linears, tensors, layerPrefix+".mlp.up_proj")
		layer.MLP.DownProj = makeClippableLinear(linears, tensors, layerPrefix+".mlp.down_proj")

		// Layer scalar.
		if w := tensors[layerPrefix+".layer_scalar"]; w != nil {
			layer.LayerScalar = w
		}

		// Validation.
		if layer.InputNorm == nil || layer.PostAttnNorm == nil ||
			layer.PreFFNorm == nil || layer.PostFFNorm == nil {
			return fmt.Errorf("vision layer %d: missing norm weights", i)
		}
		if layer.Attention.QProj == nil || layer.Attention.KProj == nil ||
			layer.Attention.VProj == nil || layer.Attention.OProj == nil {
			return fmt.Errorf("vision layer %d: missing attention projections", i)
		}
		if layer.Attention.QNorm == nil || layer.Attention.KNorm == nil {
			return fmt.Errorf("vision layer %d: missing attention q/k norms", i)
		}
		if layer.MLP.GateProj == nil || layer.MLP.UpProj == nil || layer.MLP.DownProj == nil {
			return fmt.Errorf("vision layer %d: missing mlp projections", i)
		}
		if layer.Attention.QProj.Linear == nil || layer.Attention.KProj.Linear == nil ||
			layer.Attention.VProj.Linear == nil || layer.Attention.OProj.Linear == nil {
			return fmt.Errorf("vision layer %d: missing attention linear weights", i)
		}
		if layer.MLP.GateProj.Linear == nil || layer.MLP.UpProj.Linear == nil || layer.MLP.DownProj.Linear == nil {
			return fmt.Errorf("vision layer %d: missing mlp linear weights", i)
		}

		m.VisionEncoder.Layers[i] = layer
	}
	if vcfg.Standardize {
		m.VisionEncoder.StdBias = tensors[visionPrefix+"std_bias"]
		m.VisionEncoder.StdScale = tensors[visionPrefix+"std_scale"]
		if m.VisionEncoder.StdBias == nil || m.VisionEncoder.StdScale == nil {
			return fmt.Errorf("missing vision standardization weights")
		}
	}

	return m.loadVisionProjection(tensors, linears)
}

func (m *Model) loadUnifiedVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	const prefix = "model.vision_embedder."

	patchDense := linears.Make(prefix + "patch_dense")
	patchNorm1Weight := tensors[prefix+"patch_ln1.weight"]
	patchNorm1Bias := tensors[prefix+"patch_ln1.bias"]
	patchNorm2Weight := tensors[prefix+"patch_ln2.weight"]
	patchNorm2Bias := tensors[prefix+"patch_ln2.bias"]
	positionEmbedding := tensors[prefix+"pos_embedding"]
	positionNormWeight := tensors[prefix+"pos_norm.weight"]
	positionNormBias := tensors[prefix+"pos_norm.bias"]
	if patchDense == nil || patchNorm1Weight == nil || patchNorm1Bias == nil ||
		patchNorm2Weight == nil || patchNorm2Bias == nil || positionEmbedding == nil ||
		positionNormWeight == nil || positionNormBias == nil {
		return fmt.Errorf("missing unified vision embedder weights")
	}
	if dims := positionEmbedding.Dims(); len(dims) != 3 ||
		dims[0] != int(m.VisionConfig.PositionEmbeddingSize) ||
		dims[1] != 2 ||
		dims[2] != int(m.VisionConfig.HiddenSize) {
		return fmt.Errorf("unexpected unified vision position embedding shape %v", dims)
	}
	// Unified checkpoints store this as [position, axis, hidden], while the
	// regular vision tower and the shared lookup path use [axis, position, hidden].
	positionEmbedding = mlx.Transpose(positionEmbedding, 1, 0, 2)

	m.VisionEncoder.UnifiedPatchEmbedder = &UnifiedVisionPatchEmbedder{
		PatchNorm1: &nn.LayerNorm{
			Weight: patchNorm1Weight,
			Bias:   patchNorm1Bias,
		},
		PatchDense: patchDense,
		PatchNorm2: &nn.LayerNorm{
			Weight: patchNorm2Weight,
			Bias:   patchNorm2Bias,
		},
		PositionEmbedding: positionEmbedding,
		PositionNorm: &nn.LayerNorm{
			Weight: positionNormWeight,
			Bias:   positionNormBias,
		},
	}

	return m.loadVisionProjection(tensors, linears)
}

func (m *Model) loadVisionProjection(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	embedPrefix := "model.embed_vision."
	if tensors[embedPrefix+"embedding_projection.weight"] == nil {
		embedPrefix = "embed_vision."
	}
	m.MultimodalEmbed = &MultimodalEmbedder{
		EmbeddingProjection: linears.Make(embedPrefix + "embedding_projection"),
		Eps:                 m.VisionConfig.RMSNormEps,
	}
	if m.MultimodalEmbed.EmbeddingProjection == nil {
		return fmt.Errorf("missing multimodal embedder embedding_projection")
	}

	return nil
}

// precomputeVisionScaledWeights assigns raw norm weights to the *Scaled fields.
// Gemma 4 uses scale_shift=0.0 for all norms (no +1.0 adjustment).
func precomputeVisionScaledWeights(ve *VisionEncoder) {
	for _, layer := range ve.Layers {
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
		if layer.Attention.QNorm != nil {
			layer.Attention.QNormScaled = layer.Attention.QNorm.Weight
		}
		if layer.Attention.KNorm != nil {
			layer.Attention.KNormScaled = layer.Attention.KNorm.Weight
		}
	}
}

// patchify converts pixel values to patch embeddings.
// Input: pixelValues [B, 3, H, W] (channels-first, float32, rescaled to [0,1])
// Output: patches [B, numPatches, 3*patchSize^2]
func patchify(pixelValues *mlx.Array, patchSize int32) *mlx.Array {
	dims := pixelValues.Dims()
	B := int32(dims[0])
	H, W := int32(dims[2]), int32(dims[3])
	patchH := H / patchSize
	patchW := W / patchSize
	C := int32(3) // channels

	// Reshape: [B, C, pH, pS, pW, pS]
	x := mlx.Reshape(pixelValues, B, C, patchH, patchSize, patchW, patchSize)
	// Permute to: [B, pH, pW, pS, pS, C]
	x = mlx.Transpose(x, 0, 2, 4, 3, 5, 1)
	// Flatten to: [B, pH*pW, pS*pS*C]
	return mlx.Reshape(x, B, patchH*patchW, patchSize*patchSize*C)
}

// compute2DRoPE precomputes cos/sin arrays for 2D rotary position embeddings.
// positions is a flat slice of [x0, y0, x1, y1, ...] patch coordinates.
// Returns cos, sin arrays of shape [1, 1, numPatches, headDim] for broadcasting.
func compute2DRoPE(positions []int32, headDim int32, theta float32) (*mlx.Array, *mlx.Array) {
	numPatches := int32(len(positions) / 2)
	ndim := int32(2)
	dimPerSpatial := headDim / ndim // 32 for head_dim=64
	halfDim := dimPerSpatial / 2    // 16

	// Compute inverse frequencies for each spatial dimension.
	invFreq := make([]float64, halfDim)
	for i := range halfDim {
		invFreq[i] = 1.0 / math.Pow(float64(theta), 2.0*float64(i)/float64(dimPerSpatial))
	}

	cosVals := make([]float32, numPatches*headDim)
	sinVals := make([]float32, numPatches*headDim)

	for p := range numPatches {
		px := float64(positions[p*2])
		py := float64(positions[p*2+1])
		offset := p * headDim

		// X-dimension: dims [0, dimPerSpatial)
		// emb_x = [freqs_x, freqs_x] (duplicated for rotate_half convention)
		for i := range halfDim {
			freq := px * invFreq[i]
			c, s := float32(math.Cos(freq)), float32(math.Sin(freq))
			cosVals[offset+i] = c
			sinVals[offset+i] = s
			cosVals[offset+halfDim+i] = c
			sinVals[offset+halfDim+i] = s
		}

		// Y-dimension: dims [dimPerSpatial, headDim)
		for i := range halfDim {
			freq := py * invFreq[i]
			c, s := float32(math.Cos(freq)), float32(math.Sin(freq))
			cosVals[offset+dimPerSpatial+i] = c
			sinVals[offset+dimPerSpatial+i] = s
			cosVals[offset+dimPerSpatial+halfDim+i] = c
			sinVals[offset+dimPerSpatial+halfDim+i] = s
		}
	}

	// Shape: [1, 1, numPatches, headDim] for broadcasting with [B, numHeads, seq, headDim]
	cos := mlx.FromValues(cosVals, 1, 1, int(numPatches), int(headDim))
	sin := mlx.FromValues(sinVals, 1, 1, int(numPatches), int(headDim))
	return cos, sin
}

// applyRoPE2D applies 2D rotary position embeddings using precomputed cos/sin.
// x: [B, numHeads, seq, headDim], cos/sin: [1, 1, seq, headDim]
// The rotation is applied per spatial dimension (headDim/2 each), matching
// Python's apply_multidimensional_rope which splits into ndim=2 parts.
func applyRoPE2D(x, cos, sin *mlx.Array) *mlx.Array {
	dims := x.Dims()
	B, H, S := int32(dims[0]), int32(dims[1]), int32(dims[2])
	headDim := int32(dims[3])
	dimPerSpatial := headDim / 2 // 32 for headDim=64
	halfDim := dimPerSpatial / 2 // 16

	// Process each spatial dimension independently.
	var parts []*mlx.Array
	for d := range int32(2) {
		off := d * dimPerSpatial

		// Extract this spatial dimension's slice: x[..., off:off+dimPerSpatial]
		xPart := mlx.SliceStartStop(x,
			[]int32{0, 0, 0, off},
			[]int32{B, H, S, off + dimPerSpatial},
		)
		cosPart := mlx.SliceStartStop(cos,
			[]int32{0, 0, 0, off},
			[]int32{1, 1, S, off + dimPerSpatial},
		)
		sinPart := mlx.SliceStartStop(sin,
			[]int32{0, 0, 0, off},
			[]int32{1, 1, S, off + dimPerSpatial},
		)

		// rotate_half within this part: [-x2, x1]
		x1 := mlx.SliceStartStop(xPart,
			[]int32{0, 0, 0, 0},
			[]int32{B, H, S, halfDim},
		)
		x2 := mlx.SliceStartStop(xPart,
			[]int32{0, 0, 0, halfDim},
			[]int32{B, H, S, dimPerSpatial},
		)
		rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(x2), x1}, 3)

		// xPart * cos + rotated * sin
		parts = append(parts, mlx.Add(mlx.Mul(xPart, cosPart), mlx.Mul(rotated, sinPart)))
	}

	return mlx.Concatenate(parts, 3)
}

// buildPatchPositions creates patch coordinates in row-major image order.
func buildPatchPositions(patchH, patchW int32) []int32 {
	positions := make([]int32, patchH*patchW*2)
	idx := int32(0)
	for y := range patchH {
		for x := range patchW {
			positions[idx*2] = x
			positions[idx*2+1] = y
			idx++
		}
	}

	return positions
}

// compute2DPositionEmbeddings computes position embeddings from the embedding table.
// positions: [numPatches*2] flat (x0,y0,...)
// table: [2, posEmbSize, hiddenSize]
// Returns: [1, numPatches, hiddenSize]
func compute2DPositionEmbeddings(positions []int32, table *mlx.Array, posEmbSize, hiddenSize int32) *mlx.Array {
	numPatches := int32(len(positions) / 2)

	// Build index arrays for each spatial dimension.
	xIndices := make([]int32, numPatches)
	yIndices := make([]int32, numPatches)
	for p := range numPatches {
		xIndices[p] = positions[p*2]
		yIndices[p] = positions[p*2+1]
	}

	// table: [2, posEmbSize, hiddenSize]
	// Extract x-dim table (table[0]) and y-dim table (table[1]).
	xTable := mlx.SliceStartStop(table,
		[]int32{0, 0, 0},
		[]int32{1, posEmbSize, hiddenSize},
	)
	xTable = mlx.Squeeze(xTable, 0) // [posEmbSize, hiddenSize]

	yTable := mlx.SliceStartStop(table,
		[]int32{1, 0, 0},
		[]int32{2, posEmbSize, hiddenSize},
	)
	yTable = mlx.Squeeze(yTable, 0) // [posEmbSize, hiddenSize]

	// Gather: look up positions from each table.
	xIdx := mlx.NewArrayInt32(xIndices, []int32{numPatches})
	yIdx := mlx.NewArrayInt32(yIndices, []int32{numPatches})

	xEmb := mlx.Take(xTable, xIdx, 0) // [numPatches, hiddenSize]
	yEmb := mlx.Take(yTable, yIdx, 0) // [numPatches, hiddenSize]

	// Sum x and y embeddings.
	posEmb := mlx.Add(xEmb, yEmb) // [numPatches, hiddenSize]

	// Add batch dim: [1, numPatches, hiddenSize]
	return mlx.ExpandDims(posEmb, 0)
}

// VisionTokenCount computes how many soft tokens a given image size will produce,
// without running the vision encoder. This allows the pipeline to determine the
// token count before deciding whether vision encoding is needed.
func (m *Model) VisionTokenCount(height, width int) int {
	vcfg := m.VisionConfig
	if vcfg == nil {
		return 0
	}
	patchH := int32(height) / vcfg.PatchSize
	patchW := int32(width) / vcfg.PatchSize
	k := vcfg.PoolingKernelSize
	return int((patchH / k) * (patchW / k))
}

// ForwardVision encodes pixel values into soft tokens in text embedding space.
// pixelValues: [B, 3, H, W] float32 tensor (channels-first, values in [0, 1]).
// Returns: soft tokens [1, numRealTokens, textHiddenSize].
func (m *Model) ForwardVision(pixelValues *mlx.Array) *mlx.Array {
	ve := m.VisionEncoder
	vcfg := ve.Cfg
	if ve.UnifiedPatchEmbedder != nil {
		return m.forwardUnifiedVision(pixelValues)
	}

	dims := pixelValues.Dims()
	B := int32(dims[0])
	H, W := int32(dims[2]), int32(dims[3])
	patchH := H / vcfg.PatchSize
	patchW := W / vcfg.PatchSize

	positions := buildPatchPositions(patchH, patchW)

	// Patchify and project.
	patches := patchify(pixelValues, vcfg.PatchSize)
	patches = mlx.AddScalar(mlx.MulScalar(patches, 2.0), -1.0)
	h := ve.PatchEmbedder.InputProj.Forward(patches)

	// Add 2D position embeddings.
	posEmb := compute2DPositionEmbeddings(positions,
		ve.PatchEmbedder.PositionEmbeddingTable, vcfg.PositionEmbeddingSize, vcfg.HiddenSize)
	h = mlx.Add(h, posEmb)

	cos, sin := compute2DRoPE(positions, vcfg.HeadDim, vcfg.RopeTheta)
	materializeVisionState(h, cos, sin)

	// Run through vision transformer layers.
	for _, layer := range ve.Layers {
		h = visionLayerForward(layer, h, cos, sin, B, vcfg)
		materializeVisionState(h, cos, sin)
	}

	h = visionPool(h, patchH, patchW, vcfg.PoolingKernelSize)

	// Scale by sqrt(hidden_size).
	h = mlx.MulScalar(h, float32(math.Sqrt(float64(vcfg.HiddenSize))))

	if vcfg.Standardize {
		h = mlx.Mul(mlx.Sub(h, ve.StdBias), ve.StdScale)
	}

	// Project to text embedding space via multimodal embedder.
	h = m.MultimodalEmbed.Forward(h)

	return h
}

func (m *Model) forwardUnifiedVision(pixelValues *mlx.Array) *mlx.Array {
	ve := m.VisionEncoder
	cfg := ve.Cfg
	embedder := ve.UnifiedPatchEmbedder

	dims := pixelValues.Dims()
	patchH := int32(dims[2]) / cfg.PatchSize
	patchW := int32(dims[3]) / cfg.PatchSize
	positions := buildPatchPositions(patchH, patchW)

	h := embedder.PatchNorm1.Forward(patchify(pixelValues, cfg.PatchSize))
	h = embedder.PatchDense.Forward(h)
	h = embedder.PatchNorm2.Forward(h)
	h = mlx.Add(h, compute2DPositionEmbeddings(
		positions,
		embedder.PositionEmbedding,
		cfg.PositionEmbeddingSize,
		cfg.HiddenSize,
	))
	h = embedder.PositionNorm.Forward(h)
	return m.MultimodalEmbed.Forward(h)
}

func materializeVisionState(h, cos, sin *mlx.Array) {
	mlx.Eval(h, cos, sin)
	mlx.Pin(h, cos, sin)
	mlx.Sweep()
	mlx.Unpin(h, cos, sin)
}

func visionLayerForward(layer *VisionDecoderLayer, x, cos, sin *mlx.Array, B int32, cfg *VisionConfig) *mlx.Array {
	S := int32(x.Dim(1))

	// Pre-attention norm.
	normed := mlx.RMSNormFn(x, layer.InputNormScaled, cfg.RMSNormEps)

	// Self-attention.
	attnOut := visionAttentionForward(layer.Attention, normed, cos, sin, B, S, cfg)
	attnOut = mlx.RMSNormFn(attnOut, layer.PostAttnNormScaled, cfg.RMSNormEps)
	h := mlx.Add(x, attnOut)

	// MLP.
	normed = mlx.RMSNormFn(h, layer.PreFFNormScaled, cfg.RMSNormEps)
	mlpOut := visionMLPForward(layer.MLP, normed)
	mlpOut = mlx.RMSNormFn(mlpOut, layer.PostFFNormScaled, cfg.RMSNormEps)
	h = mlx.Add(h, mlpOut)

	// Layer scalar.
	if layer.LayerScalar != nil {
		h = mlx.Mul(h, layer.LayerScalar)
	}

	return h
}

func visionAttentionForward(a *VisionAttention, x, cos, sin *mlx.Array, B, S int32, cfg *VisionConfig) *mlx.Array {
	headDim := cfg.HeadDim
	numHeads := cfg.NumAttentionHeads
	numKVHeads := cfg.NumKeyValueHeads

	q := a.QProj.Forward(x)
	q = mlx.Reshape(q, B, S, numHeads, headDim)

	k := a.KProj.Forward(x)
	k = mlx.Reshape(k, B, S, numKVHeads, headDim)

	v := a.VProj.Forward(x)
	v = mlx.Reshape(v, B, S, numKVHeads, headDim)

	// Apply Q/K norms.
	q = mlx.RMSNormFn(q, a.QNormScaled, cfg.RMSNormEps)
	k = mlx.RMSNormFn(k, a.KNormScaled, cfg.RMSNormEps)

	// Apply V norm (no learnable weight).
	v = mlx.RMSNormFn(v, nil, cfg.RMSNormEps)

	// Apply 2D RoPE (after norms, before transpose).
	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)

	q = applyRoPE2D(q, cos, sin)
	k = applyRoPE2D(k, cos, sin)

	v = mlx.Transpose(v, 0, 2, 1, 3)

	// Bidirectional attention over the image's real patches.
	out := mlx.FastScaledDotProductAttention(q, k, v, cfg.Scale, "", nil)

	// Reshape and project output.
	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, S, numHeads*headDim)
	return a.OProj.Forward(out)
}

func visionMLPForward(mlp *VisionMLP, x *mlx.Array) *mlx.Array {
	gate := mlx.GELUApprox(mlp.GateProj.Forward(x))
	up := mlp.UpProj.Forward(x)
	return mlp.DownProj.Forward(mlx.Mul(gate, up))
}

// visionPool performs non-overlapping 2D spatial average pooling.
func visionPool(h *mlx.Array, patchH, patchW, kernel int32) *mlx.Array {
	B := int32(h.Dim(0))
	hidden := int32(h.Dim(2))
	pooledH := patchH / kernel
	pooledW := patchW / kernel

	h = mlx.Reshape(h, B, pooledH, kernel, pooledW, kernel, hidden)
	h = mlx.Mean(h, 4, false)
	h = mlx.Mean(h, 2, false)
	return mlx.Reshape(h, B, pooledH*pooledW, hidden)
}
