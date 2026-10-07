package nemotron_h

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const (
	nemotronVisionDefaultLayerNormEpsilon = float32(1e-6)
	nemotronVisionDefaultProjectorNormEps = float32(1e-5)
	nemotronVisionDefaultMaxModelLen      = 16384
	nemotronVisionReservedTokens          = 4
)

type radioModelSpec struct {
	patchSize         int32
	hiddenSize        int32
	numHiddenLayers   int32
	numAttentionHeads int32
}

// radioModelSpecs is the architecture set used by RADIO's timm model names.
var radioModelSpecs = map[string]radioModelSpec{
	"vit_small_patch16_224": {patchSize: 16, hiddenSize: 384, numHiddenLayers: 12, numAttentionHeads: 6},
	"vit_base_patch16_224":  {patchSize: 16, hiddenSize: 768, numHiddenLayers: 12, numAttentionHeads: 12},
	"vit_large_patch16_224": {patchSize: 16, hiddenSize: 1024, numHiddenLayers: 24, numAttentionHeads: 16},
	"vit_huge_patch16_224":  {patchSize: 16, hiddenSize: 1280, numHiddenLayers: 32, numAttentionHeads: 16},
}

type VisionConfig struct {
	ModelName         string
	PSVersion         string
	Version           string
	PatchSize         int32
	HiddenSize        int32
	NumHiddenLayers   int32
	NumAttentionHeads int32
	HeadDim           int32
	LayerNormEps      float32
	ProjectorNormEps  float32
	DownsampleFactor  int32
	MinNumPatches     int
	MaxNumPatches     int
	MaxModelLen       int
	Mean              [3]float32
	Std               [3]float32
	ImageTokenID      int32
	ImageToken        string
	ImageStartToken   string
	ImageEndToken     string

	configuredPatchSize       int32
	configuredHiddenSize      int32
	configuredNumHiddenLayers int32
}

type RadioVisionEncoder struct {
	PatchEmbed       nn.LinearLayer
	ClassToken       *mlx.Array
	Position         *mlx.Array
	PositionGridSize int32
	Layers           []*RadioVisionLayer
}

type RadioVisionLayer struct {
	Norm1 *nn.LayerNorm
	QKV   nn.LinearLayer
	Proj  nn.LinearLayer
	Norm2 *nn.LayerNorm
	FC1   nn.LinearLayer
	FC2   nn.LinearLayer
}

type VisionProjector struct {
	Norm *nn.RMSNorm
	FC1  nn.LinearLayer
	FC2  nn.LinearLayer
}

func parseVisionConfig(configData, preprocessorData []byte) (*VisionConfig, error) {
	var env struct {
		PSVersion    string `json:"ps_version"`
		VisionConfig *struct {
			Version         string `json:"version"`
			PatchSize       int32  `json:"patch_size"`
			MinNumPatches   int    `json:"min_num_patches"`
			MaxNumPatches   int    `json:"max_num_patches"`
			HiddenSize      int32  `json:"hidden_size"`
			NumHiddenLayers int32  `json:"num_hidden_layers"`
			NumHeads        int32  `json:"num_attention_heads"`
			Args            struct {
				Model         string `json:"model"`
				MinNumPatches int    `json:"min_num_patches"`
				MaxNumPatches int    `json:"max_num_patches"`
			} `json:"args"`
		} `json:"vision_config"`
		PatchSize         int32     `json:"patch_size"`
		DownsampleRatio   float32   `json:"downsample_ratio"`
		ImgContextTokenID int32     `json:"img_context_token_id"`
		ImgContextToken   string    `json:"img_context_token"`
		ImgStartToken     string    `json:"img_start_token"`
		ImgEndToken       string    `json:"img_end_token"`
		VitHiddenSize     int32     `json:"vit_hidden_size"`
		NormMean          []float32 `json:"norm_mean"`
		NormStd           []float32 `json:"norm_std"`
	}
	if err := json.Unmarshal(configData, &env); err != nil {
		return nil, fmt.Errorf("parse vision config: %w", err)
	}
	if env.VisionConfig == nil {
		return nil, nil
	}
	modelName := strings.TrimSpace(env.VisionConfig.Args.Model)
	spec, ok := radioModelSpecs[modelName]
	if !ok {
		return nil, fmt.Errorf("unsupported RADIO model %q", modelName)
	}
	psVersion := strings.TrimSpace(env.PSVersion)
	if psVersion != "v2" {
		return nil, fmt.Errorf("unsupported RADIO ps_version %q", psVersion)
	}
	if heads := env.VisionConfig.NumHeads; heads > 0 && heads != spec.numAttentionHeads {
		return nil, fmt.Errorf("RADIO model %q has num_attention_heads=%d, config has %d", modelName, spec.numAttentionHeads, heads)
	}

	cfg := &VisionConfig{
		ModelName:                 modelName,
		PSVersion:                 psVersion,
		Version:                   strings.TrimSpace(env.VisionConfig.Version),
		NumAttentionHeads:         spec.numAttentionHeads,
		LayerNormEps:              nemotronVisionDefaultLayerNormEpsilon,
		ProjectorNormEps:          nemotronVisionDefaultProjectorNormEps,
		MinNumPatches:             firstPositiveInt(env.VisionConfig.MinNumPatches, env.VisionConfig.Args.MinNumPatches),
		MaxNumPatches:             firstPositiveInt(env.VisionConfig.MaxNumPatches, env.VisionConfig.Args.MaxNumPatches),
		MaxModelLen:               nemotronVisionDefaultMaxModelLen,
		Mean:                      [3]float32{0.48145466, 0.4578275, 0.40821073},
		Std:                       [3]float32{0.26862954, 0.26130258, 0.27577711},
		ImageTokenID:              env.ImgContextTokenID,
		ImageToken:                firstNonEmpty(env.ImgContextToken, "<image>"),
		ImageStartToken:           firstNonEmpty(env.ImgStartToken, "<img>"),
		ImageEndToken:             firstNonEmpty(env.ImgEndToken, "</img>"),
		configuredPatchSize:       firstPositiveInt32(env.VisionConfig.PatchSize, env.PatchSize),
		configuredHiddenSize:      firstPositiveInt32(env.VisionConfig.HiddenSize, env.VitHiddenSize),
		configuredNumHiddenLayers: env.VisionConfig.NumHiddenLayers,
	}
	if env.DownsampleRatio > 0 {
		cfg.DownsampleFactor = int32(math.Round(float64(1 / env.DownsampleRatio)))
	}
	if cfg.DownsampleFactor <= 0 {
		cfg.DownsampleFactor = 2
	}
	if err := setTriplet(cfg.Mean[:], env.NormMean, "norm_mean"); err != nil {
		return nil, fmt.Errorf("vision config: %w", err)
	}
	if err := setTriplet(cfg.Std[:], env.NormStd, "norm_std"); err != nil {
		return nil, fmt.Errorf("vision config: %w", err)
	}

	if len(preprocessorData) > 0 {
		if err := cfg.applyPreprocessorConfig(preprocessorData); err != nil {
			return nil, err
		}
	}

	switch cfg.Version {
	case "", "radio_v2.5-h", "c-radio_v4-h":
	default:
		return nil, fmt.Errorf("unsupported RADIO version %q", cfg.Version)
	}
	if cfg.DownsampleFactor != 2 {
		return nil, fmt.Errorf("unsupported RADIO downsample factor=%d", cfg.DownsampleFactor)
	}
	if cfg.MinNumPatches <= 0 || cfg.MaxNumPatches <= 0 {
		return nil, fmt.Errorf("RADIO dynamic resolution requires min_num_patches and max_num_patches")
	}
	if cfg.MinNumPatches > cfg.MaxNumPatches {
		return nil, fmt.Errorf("RADIO min_num_patches (%d) exceeds max_num_patches (%d)", cfg.MinNumPatches, cfg.MaxNumPatches)
	}
	return cfg, nil
}

func (cfg *VisionConfig) applyPreprocessorConfig(data []byte) error {
	var pre struct {
		PatchSize       int32     `json:"patch_size"`
		DownsampleRatio float32   `json:"downsample_ratio"`
		NormMean        []float32 `json:"norm_mean"`
		NormStd         []float32 `json:"norm_std"`
		MinNumPatches   int       `json:"min_num_patches"`
		MaxNumPatches   int       `json:"max_num_patches"`
		MaxModelLen     int       `json:"max_model_len"`
	}
	if err := json.Unmarshal(data, &pre); err != nil {
		return fmt.Errorf("parse preprocessor_config.json: %w", err)
	}
	if pre.PatchSize > 0 {
		if cfg.configuredPatchSize > 0 && pre.PatchSize != cfg.configuredPatchSize {
			return fmt.Errorf("parse preprocessor_config.json: patch_size=%d, config has %d", pre.PatchSize, cfg.configuredPatchSize)
		}
		cfg.configuredPatchSize = pre.PatchSize
	}
	if pre.DownsampleRatio > 0 {
		cfg.DownsampleFactor = int32(math.Round(float64(1 / pre.DownsampleRatio)))
	}
	cfg.MinNumPatches = firstPositiveInt(pre.MinNumPatches, cfg.MinNumPatches)
	cfg.MaxNumPatches = firstPositiveInt(pre.MaxNumPatches, cfg.MaxNumPatches)
	cfg.MaxModelLen = firstPositiveInt(pre.MaxModelLen, cfg.MaxModelLen)
	if err := setTriplet(cfg.Mean[:], pre.NormMean, "norm_mean"); err != nil {
		return fmt.Errorf("parse preprocessor_config.json: %w", err)
	}
	if err := setTriplet(cfg.Std[:], pre.NormStd, "norm_std"); err != nil {
		return fmt.Errorf("parse preprocessor_config.json: %w", err)
	}
	return nil
}

func firstPositiveInt32(values ...int32) int32 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func setTriplet(dst []float32, src []float32, name string) error {
	if len(src) == 0 {
		return nil
	}
	if len(src) != len(dst) {
		return fmt.Errorf("%s must contain %d values, got %d", name, len(dst), len(src))
	}
	copy(dst, src)
	return nil
}

func (m *Model) configureVision(cfg *VisionConfig) error {
	imageStartTokenID, ok := m.tok.GetSpecialToken(cfg.ImageStartToken)
	if !ok {
		return fmt.Errorf("tokenizer is missing %s", cfg.ImageStartToken)
	}
	imageTokenID, ok := m.tok.GetSpecialToken(cfg.ImageToken)
	if !ok {
		return fmt.Errorf("tokenizer is missing %s", cfg.ImageToken)
	}
	if cfg.ImageTokenID > 0 && imageTokenID != cfg.ImageTokenID {
		return fmt.Errorf("tokenizer %s id = %d, config has %d", cfg.ImageToken, imageTokenID, cfg.ImageTokenID)
	}
	imageEndTokenID, ok := m.tok.GetSpecialToken(cfg.ImageEndToken)
	if !ok {
		return fmt.Errorf("tokenizer is missing %s", cfg.ImageEndToken)
	}

	m.VisionConfig = cfg
	m.VisionEncoder = &RadioVisionEncoder{}
	m.Projector = &VisionProjector{}
	m.imageStartTokenID = imageStartTokenID
	m.imageTokenID = imageTokenID
	m.imageEndTokenID = imageEndTokenID
	return nil
}

func (m *Model) loadVisionWeights(tensors map[string]*mlx.Array, linears model.LinearFactory) error {
	visionPrefix := resolveVisionPrefix(tensors)
	if visionPrefix == "" {
		return fmt.Errorf("missing RADIO vision weights")
	}

	ve := m.VisionEncoder
	ve.PatchEmbed = linears.Make(visionPrefix + "patch_generator.embedder")
	ve.ClassToken = tensors[visionPrefix+"patch_generator.cls_token.token"]
	ve.Position = tensors[visionPrefix+"patch_generator.pos_embed"]
	if ve.PatchEmbed == nil || ve.ClassToken == nil || ve.Position == nil {
		return fmt.Errorf("missing RADIO patch generator weights")
	}
	positionGridSize, err := radioPositionGridSize(ve.Position)
	if err != nil {
		return err
	}
	ve.PositionGridSize = positionGridSize

	cfg := m.VisionConfig
	if err := cfg.loadArchitecture(tensors, visionPrefix); err != nil {
		return err
	}
	ve.Layers = make([]*RadioVisionLayer, cfg.NumHiddenLayers)
	for i := range cfg.NumHiddenLayers {
		prefix := fmt.Sprintf("%sblocks.%d", visionPrefix, i)
		layer := &RadioVisionLayer{
			Norm1: newLayerNorm(tensors, prefix+".norm1", cfg.LayerNormEps),
			QKV:   linears.Make(prefix + ".attn.qkv"),
			Proj:  linears.Make(prefix + ".attn.proj"),
			Norm2: newLayerNorm(tensors, prefix+".norm2", cfg.LayerNormEps),
			FC1:   linears.Make(prefix + ".mlp.fc1"),
			FC2:   linears.Make(prefix + ".mlp.fc2"),
		}
		if layer.Norm1 == nil || layer.QKV == nil || layer.Proj == nil ||
			layer.Norm2 == nil || layer.FC1 == nil || layer.FC2 == nil {
			return fmt.Errorf("vision layer %d: missing RADIO transformer weights", i)
		}
		ve.Layers[i] = layer
	}

	projector := m.Projector
	normWeight := tensors["mlp1.0.weight"]
	projector.FC1 = linears.Make("mlp1.1")
	projector.FC2 = linears.Make("mlp1.3")
	if normWeight == nil || projector.FC1 == nil || projector.FC2 == nil {
		return fmt.Errorf("missing Nemotron vision projector weights")
	}
	projector.Norm = nn.NewRMSNorm(normWeight, cfg.ProjectorNormEps)
	return nil
}

func (cfg *VisionConfig) loadArchitecture(tensors map[string]*mlx.Array, visionPrefix string) error {
	weight := tensors[visionPrefix+"patch_generator.embedder.weight"]
	if weight == nil {
		return fmt.Errorf("missing RADIO patch embedder weight")
	}
	hiddenSize, patchSize, err := radioPatchDimensions(weight.Dims())
	if err != nil {
		return fmt.Errorf("RADIO patch embedder: %w", err)
	}
	numHiddenLayers, err := radioLayerCount(tensors, visionPrefix)
	if err != nil {
		return err
	}

	spec, ok := radioModelSpecs[cfg.ModelName]
	if !ok {
		return fmt.Errorf("unsupported RADIO model %q", cfg.ModelName)
	}
	cfg.NumAttentionHeads = spec.numAttentionHeads
	for _, check := range []struct {
		name string
		got  int32
		want int32
	}{
		{name: "patch_size", got: patchSize, want: spec.patchSize},
		{name: "hidden_size", got: hiddenSize, want: spec.hiddenSize},
		{name: "num_hidden_layers", got: numHiddenLayers, want: spec.numHiddenLayers},
	} {
		if check.got != check.want {
			return fmt.Errorf("RADIO model %q has %s=%d, weights have %d", cfg.ModelName, check.name, check.want, check.got)
		}
	}
	for _, check := range []struct {
		name       string
		configured int32
		derived    int32
	}{
		{name: "patch_size", configured: cfg.configuredPatchSize, derived: patchSize},
		{name: "hidden_size", configured: cfg.configuredHiddenSize, derived: hiddenSize},
		{name: "num_hidden_layers", configured: cfg.configuredNumHiddenLayers, derived: numHiddenLayers},
	} {
		if check.configured > 0 && check.configured != check.derived {
			return fmt.Errorf("RADIO %s=%d, weights have %d", check.name, check.configured, check.derived)
		}
	}
	if hiddenSize%cfg.NumAttentionHeads != 0 {
		return fmt.Errorf("vision hidden_size (%d) must be divisible by num_attention_heads (%d)", hiddenSize, cfg.NumAttentionHeads)
	}

	cfg.PatchSize = patchSize
	cfg.HiddenSize = hiddenSize
	cfg.NumHiddenLayers = numHiddenLayers
	cfg.HeadDim = hiddenSize / cfg.NumAttentionHeads
	return nil
}

func radioPatchDimensions(dims []int) (hiddenSize, patchSize int32, err error) {
	if len(dims) != 2 || dims[0] <= 0 || dims[1] <= 0 {
		return 0, 0, fmt.Errorf("weight shape %v, want [hidden_size, 3*patch_size*patch_size]", dims)
	}
	if dims[1]%3 != 0 {
		return 0, 0, fmt.Errorf("input dimension %d is not divisible by 3 RGB channels", dims[1])
	}
	patchArea := dims[1] / 3
	patch := int(math.Sqrt(float64(patchArea)))
	if patch*patch != patchArea {
		return 0, 0, fmt.Errorf("input dimension %d does not describe square RGB patches", dims[1])
	}
	return int32(dims[0]), int32(patch), nil
}

func radioLayerCount(tensors map[string]*mlx.Array, visionPrefix string) (int32, error) {
	blockPrefix := visionPrefix + "blocks."
	indices := make(map[int]struct{})
	for name := range tensors {
		rest, ok := strings.CutPrefix(name, blockPrefix)
		if !ok {
			continue
		}
		indexText, _, ok := strings.Cut(rest, ".")
		if !ok {
			continue
		}
		index, err := strconv.Atoi(indexText)
		if err != nil || index < 0 {
			continue
		}
		indices[index] = struct{}{}
	}
	if len(indices) == 0 {
		return 0, fmt.Errorf("missing RADIO transformer block weights")
	}
	for i := range len(indices) {
		if _, ok := indices[i]; !ok {
			return 0, fmt.Errorf("missing RADIO transformer block %d", i)
		}
	}
	return int32(len(indices)), nil
}

func resolveVisionPrefix(tensors map[string]*mlx.Array) string {
	for _, prefix := range []string{
		"vision_model.radio_model.model.",
		"model.vision_model.radio_model.model.",
	} {
		if tensors[prefix+"patch_generator.embedder.weight"] != nil {
			return prefix
		}
	}
	return ""
}

func newLayerNorm(tensors map[string]*mlx.Array, prefix string, eps float32) *nn.LayerNorm {
	weight := tensors[prefix+".weight"]
	bias := tensors[prefix+".bias"]
	if weight == nil || bias == nil {
		return nil
	}
	return &nn.LayerNorm{Weight: weight, Bias: bias, Eps: eps}
}

func (m *Model) visionTokenCount(height, width int) int {
	cfg := m.VisionConfig
	if cfg == nil || height <= 0 || width <= 0 {
		return 0
	}
	patchH := height / int(cfg.PatchSize)
	patchW := width / int(cfg.PatchSize)
	return (patchH / int(cfg.DownsampleFactor)) * (patchW / int(cfg.DownsampleFactor))
}

func (m *Model) forwardVision(pixelValues *mlx.Array) *mlx.Array {
	cfg := m.VisionConfig
	ve := m.VisionEncoder
	dims := pixelValues.Dims()
	B := int32(dims[0])
	H, W := int32(dims[2]), int32(dims[3])
	patchH := H / cfg.PatchSize
	patchW := W / cfg.PatchSize

	h := ve.PatchEmbed.Forward(radioPatchify(pixelValues, cfg.PatchSize))
	h = mlx.Add(h, radioPositionEmbedding(ve.Position, ve.PositionGridSize, patchH, patchW))
	cls := mlx.ExpandDims(ve.ClassToken, 0)
	if B > 1 {
		cls = mlx.Tile(cls, []int32{B, 1, 1})
	}
	h = mlx.Concatenate([]*mlx.Array{cls, h}, 1)

	for _, layer := range ve.Layers {
		h = radioVisionLayerForward(layer, h, B, cfg)
	}

	clsTokens := int32(ve.ClassToken.Dim(0))
	h = mlx.SliceStartStop(
		h,
		[]int32{0, clsTokens, 0},
		[]int32{B, clsTokens + patchH*patchW, cfg.HiddenSize},
	)
	h = mlx.Reshape(h, B, patchH, patchW, cfg.HiddenSize)
	h = radioPixelShuffle(h, cfg.DownsampleFactor)
	h = mlx.Reshape(h, B, (patchH/cfg.DownsampleFactor)*(patchW/cfg.DownsampleFactor), cfg.HiddenSize*cfg.DownsampleFactor*cfg.DownsampleFactor)
	return m.Projector.Forward(h)
}

func (p *VisionProjector) Forward(x *mlx.Array) *mlx.Array {
	x = p.Norm.Forward(x, 0)
	x = p.FC1.Forward(x)
	x = mlx.ReLUSquared(x)
	return p.FC2.Forward(x)
}

func radioVisionLayerForward(layer *RadioVisionLayer, x *mlx.Array, B int32, cfg *VisionConfig) *mlx.Array {
	S := int32(x.Dim(1))
	normed := layer.Norm1.Forward(x)
	attn := radioAttentionForward(layer, normed, B, S, cfg)
	h := mlx.Add(x, attn)
	mlp := layer.FC1.Forward(layer.Norm2.Forward(h))
	mlpDType := mlp.DType()
	mlp = mlx.GELU(mlp.AsType(mlx.DTypeFloat32)).AsType(mlpDType)
	mlp = layer.FC2.Forward(mlp)
	return mlx.Add(h, mlp)
}

func radioAttentionForward(layer *RadioVisionLayer, x *mlx.Array, B, S int32, cfg *VisionConfig) *mlx.Array {
	qkv := layer.QKV.Forward(x)
	q := sliceAxis(qkv, 2, 0, cfg.HiddenSize)
	k := sliceAxis(qkv, 2, cfg.HiddenSize, 2*cfg.HiddenSize)
	v := sliceAxis(qkv, 2, 2*cfg.HiddenSize, 3*cfg.HiddenSize)

	q = mlx.Transpose(mlx.Reshape(q, B, S, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)
	k = mlx.Transpose(mlx.Reshape(k, B, S, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)
	v = mlx.Transpose(mlx.Reshape(v, B, S, cfg.NumAttentionHeads, cfg.HeadDim), 0, 2, 1, 3)

	out := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(cfg.HeadDim))), "", nil)
	out = mlx.Transpose(out, 0, 2, 1, 3)
	out = mlx.Reshape(out, B, S, cfg.HiddenSize)
	return layer.Proj.Forward(out)
}

func radioPatchify(pixelValues *mlx.Array, patchSize int32) *mlx.Array {
	dims := pixelValues.Dims()
	B := int32(dims[0])
	C := int32(dims[1])
	H, W := int32(dims[2]), int32(dims[3])
	patchH := H / patchSize
	patchW := W / patchSize

	x := mlx.Reshape(pixelValues, B, C, patchH, patchSize, patchW, patchSize)
	x = mlx.Transpose(x, 0, 2, 4, 1, 3, 5)
	return mlx.Reshape(x, B, patchH*patchW, C*patchSize*patchSize)
}

func radioPixelShuffle(x *mlx.Array, factor int32) *mlx.Array {
	dims := x.Dims()
	B := int32(dims[0])
	H := int32(dims[1])
	W := int32(dims[2])
	C := int32(dims[3])

	x = mlx.Reshape(x, B, H, W/factor, factor*C)
	x = mlx.Transpose(x, 0, 2, 1, 3)
	x = mlx.Reshape(x, B, W/factor, H/factor, factor*factor*C)
	return mlx.Transpose(x, 0, 2, 1, 3)
}

func radioPositionGridSize(pos *mlx.Array) (int32, error) {
	dims := pos.Dims()
	if len(dims) < 2 {
		return 0, fmt.Errorf("RADIO position embedding has %d dimensions, want at least 2", len(dims))
	}
	numPositions := dims[len(dims)-2]
	if numPositions <= 0 {
		return 0, fmt.Errorf("RADIO position embedding has no positions")
	}
	src := int32(math.Round(math.Sqrt(float64(numPositions))))
	if int(src*src) != numPositions {
		return 0, fmt.Errorf("RADIO position embedding has %d positions, want a square grid", numPositions)
	}
	return src, nil
}

func radioPositionEmbedding(pos *mlx.Array, src, patchH, patchW int32) *mlx.Array {
	dims := pos.Dims()
	hidden := int32(dims[len(dims)-1])
	grid := mlx.Reshape(pos, 1, src, src, hidden)
	target := max(patchH, patchW)
	if target != src {
		grid = resizeGrid2D(grid, target, target)
	}
	if patchH != target || patchW != target {
		grid = mlx.SliceStartStop(
			grid,
			[]int32{0, 0, 0, 0},
			[]int32{1, patchH, patchW, hidden},
		)
	}
	return mlx.Reshape(grid, 1, patchH*patchW, hidden)
}

func resizeGrid2D(x *mlx.Array, outH, outW int32) *mlx.Array {
	if int32(x.Dim(1)) == outH && int32(x.Dim(2)) == outW {
		return x
	}
	dtype := x.DType()
	x = x.AsType(mlx.DTypeFloat32)
	if inH := int32(x.Dim(1)); inH != outH {
		lo, hi, loWeight, hiWeight := linearResizeWeights(inH, outH)
		loX := mlx.Take(x, mlx.NewArrayInt32(lo, []int32{outH}), 1)
		hiX := mlx.Take(x, mlx.NewArrayInt32(hi, []int32{outH}), 1)
		loW := mlx.FromValues(loWeight, 1, int(outH), 1, 1)
		hiW := mlx.FromValues(hiWeight, 1, int(outH), 1, 1)
		x = mlx.Add(mlx.Mul(loX, loW), mlx.Mul(hiX, hiW))
	}
	if inW := int32(x.Dim(2)); inW != outW {
		lo, hi, loWeight, hiWeight := linearResizeWeights(inW, outW)
		loX := mlx.Take(x, mlx.NewArrayInt32(lo, []int32{outW}), 2)
		hiX := mlx.Take(x, mlx.NewArrayInt32(hi, []int32{outW}), 2)
		loW := mlx.FromValues(loWeight, 1, 1, int(outW), 1)
		hiW := mlx.FromValues(hiWeight, 1, 1, int(outW), 1)
		x = mlx.Add(mlx.Mul(loX, loW), mlx.Mul(hiX, hiW))
	}
	return x.AsType(dtype)
}

func linearResizeWeights(in, out int32) ([]int32, []int32, []float32, []float32) {
	lo := make([]int32, out)
	hi := make([]int32, out)
	loWeight := make([]float32, out)
	hiWeight := make([]float32, out)
	scale := float64(in) / float64(out)
	for i := range out {
		src := (float64(i)+0.5)*scale - 0.5
		floor := math.Floor(src)
		l := int32(floor)
		h := l + 1
		wHigh := float32(src - floor)
		if l < 0 {
			l = 0
		}
		if h < 0 {
			h = 0
		}
		if l >= in {
			l = in - 1
		}
		if h >= in {
			h = in - 1
		}
		lo[i] = l
		hi[i] = h
		loWeight[i] = 1 - wHigh
		hiWeight[i] = wHigh
	}
	return lo, hi, loWeight, hiWeight
}
