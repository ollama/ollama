package qwen3_5

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"github.com/ollama/ollama/model/imageproc"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	mlxmodel "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

var errNoVisionModel = errors.New("qwen3_5: no vision model")

// VisionConfig mirrors Qwen3.5/Qwen3-Next vision_config.
type VisionConfig struct {
	Depth                 int32   `json:"depth"`
	HiddenSize            int32   `json:"hidden_size"`
	NumHeads              int32   `json:"num_heads"`
	InChannels            int32   `json:"in_channels"`
	PatchSize             int32   `json:"patch_size"`
	SpatialMergeSize      int32   `json:"spatial_merge_size"`
	LayerNormEpsilon      float32 `json:"layer_norm_epsilon"`
	RopeTheta             float32 `json:"rope_theta"`
	TemporalPatchSize     int32   `json:"temporal_patch_size"`
	NumPositionEmbeddings int32   `json:"num_position_embeddings"`

	Size struct {
		ShortestEdge int32 `json:"shortest_edge"`
		LongestEdge  int32 `json:"longest_edge"`
	} `json:"size"`

	ImageMean []float32 `json:"image_mean"`
	ImageStd  []float32 `json:"image_std"`

	GridPerSide int32 `json:"-"`
}

func (v *VisionConfig) applyDefaults() {
	if v == nil {
		return
	}
	if v.HiddenSize <= 0 {
		v.HiddenSize = 1280
	}
	if v.NumHeads <= 0 {
		v.NumHeads = 16
	}
	if v.InChannels <= 0 {
		v.InChannels = 3
	}
	if v.PatchSize <= 0 {
		v.PatchSize = 14
	}
	if v.SpatialMergeSize <= 0 {
		v.SpatialMergeSize = 2
	}
	if v.LayerNormEpsilon == 0 {
		v.LayerNormEpsilon = 1e-6
	}
	if v.RopeTheta == 0 {
		v.RopeTheta = 10000
	}
	if v.TemporalPatchSize <= 0 {
		v.TemporalPatchSize = 2
	}
	if v.NumPositionEmbeddings <= 0 {
		v.NumPositionEmbeddings = 2304
	}
	if len(v.ImageMean) < 3 {
		v.ImageMean = []float32{0.5, 0.5, 0.5}
	}
	if len(v.ImageStd) < 3 {
		v.ImageStd = []float32{0.5, 0.5, 0.5}
	}
	if v.Size.ShortestEdge <= 0 {
		v.Size.ShortestEdge = 64 << 10
	}
	if v.Size.LongestEdge <= 0 {
		v.Size.LongestEdge = 2 << 20
	}

	grid := int32(math.Sqrt(float64(v.NumPositionEmbeddings)))
	if grid <= 0 {
		grid = 48
	}
	v.GridPerSide = grid
}

func (v *VisionConfig) applyPreprocessorConfig(data []byte) {
	if v == nil || len(data) == 0 {
		return
	}

	var pre struct {
		Size struct {
			ShortestEdge int32 `json:"shortest_edge"`
			LongestEdge  int32 `json:"longest_edge"`
		} `json:"size"`
		PatchSize         int32     `json:"patch_size"`
		TemporalPatchSize int32     `json:"temporal_patch_size"`
		MergeSize         int32     `json:"merge_size"`
		ImageMean         []float32 `json:"image_mean"`
		ImageStd          []float32 `json:"image_std"`
	}
	if err := json.Unmarshal(data, &pre); err != nil {
		return
	}

	if pre.PatchSize > 0 {
		v.PatchSize = pre.PatchSize
	}
	if pre.TemporalPatchSize > 0 {
		v.TemporalPatchSize = pre.TemporalPatchSize
	}
	if pre.MergeSize > 0 {
		v.SpatialMergeSize = pre.MergeSize
	}
	if pre.Size.ShortestEdge > 0 {
		v.Size.ShortestEdge = pre.Size.ShortestEdge
	}
	if pre.Size.LongestEdge > 0 {
		v.Size.LongestEdge = pre.Size.LongestEdge
	}
	if len(pre.ImageMean) >= 3 {
		v.ImageMean = pre.ImageMean
	}
	if len(pre.ImageStd) >= 3 {
		v.ImageStd = pre.ImageStd
	}
	v.applyDefaults()
}

// VisionGrid tracks patch-grid dimensions for an image.
type VisionGrid struct {
	Height   int32
	Width    int32
	Temporal int32
}

// VisionImageProcessor reproduces qwen3vl image preprocessing.
type VisionImageProcessor struct {
	numChannels       int32
	patchSize         int32
	temporalPatchSize int32
	mergeSize         int32
	shortestEdge      int32
	longestEdge       int32
	factor            int32
	imageMean         [3]float32
	imageStd          [3]float32
}

func newVisionImageProcessor(cfg *VisionConfig) *VisionImageProcessor {
	if cfg == nil {
		return nil
	}

	return &VisionImageProcessor{
		numChannels:       cfg.InChannels,
		patchSize:         cfg.PatchSize,
		temporalPatchSize: cfg.TemporalPatchSize,
		mergeSize:         cfg.SpatialMergeSize,
		shortestEdge:      cfg.Size.ShortestEdge,
		longestEdge:       cfg.Size.LongestEdge,
		factor:            cfg.PatchSize * cfg.SpatialMergeSize,
		imageMean:         [3]float32{cfg.ImageMean[0], cfg.ImageMean[1], cfg.ImageMean[2]},
		imageStd:          [3]float32{cfg.ImageStd[0], cfg.ImageStd[1], cfg.ImageStd[2]},
	}
}

func (p *VisionImageProcessor) smartResize(height, width int) (int, int, error) {
	factor := int(p.factor)
	if factor <= 0 {
		return 0, 0, fmt.Errorf("invalid factor: %d", factor)
	}

	if height < factor || width < factor {
		return 0, 0, fmt.Errorf("height (%d) or width (%d) must be >= factor (%d)", height, width, factor)
	}
	if min(height, width) == 0 {
		return 0, 0, fmt.Errorf("invalid dimensions: %dx%d", width, height)
	}
	if max(height, width)/min(height, width) > 200 {
		return 0, 0, fmt.Errorf("aspect ratio too large: %dx%d", width, height)
	}

	roundEven := func(x float64) int { return int(math.RoundToEven(x)) }

	hBar := roundEven(float64(height)/float64(factor)) * factor
	wBar := roundEven(float64(width)/float64(factor)) * factor

	if hBar*wBar > int(p.longestEdge) {
		beta := math.Sqrt(float64(height*width) / float64(p.longestEdge))
		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < int(p.shortestEdge) {
		beta := math.Sqrt(float64(p.shortestEdge) / float64(height*width))
		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}

	return hBar, wBar, nil
}

func (p *VisionImageProcessor) ProcessImage(img image.Image) (*mlx.Array, *VisionGrid, error) {
	if p == nil {
		return nil, nil, errNoVisionModel
	}

	img = imageproc.Composite(img)
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	resizedH, resizedW, err := p.smartResize(origH, origW)
	if err != nil {
		return nil, nil, err
	}

	resized := imageproc.Resize(
		img,
		image.Point{X: resizedW, Y: resizedH},
		imageproc.ResizeBilinear,
	)
	pixels := imageproc.Normalize(resized, p.imageMean, p.imageStd, true, true)

	grid := &VisionGrid{
		Height:   int32(resizedH / int(p.patchSize)),
		Width:    int32(resizedW / int(p.patchSize)),
		Temporal: 1,
	}

	patches := p.createPatches(pixels, resizedH, resizedW, grid)

	patchDim := int(p.numChannels * p.temporalPatchSize * p.patchSize * p.patchSize)
	numPatches := int(grid.Height * grid.Width)
	pixelValues := mlx.FromValues(patches, numPatches, patchDim).ExpandDims(0)
	return pixelValues, grid, nil
}

func (p *VisionImageProcessor) createPatches(pixels []float32, height, width int, grid *VisionGrid) []float32 {
	channels := int(p.numChannels)
	patchSize := int(p.patchSize)
	mergeSize := int(p.mergeSize)
	temporalPatchSize := int(p.temporalPatchSize)

	// Temporal is always 1 for static images; only spatial patches are created.
	numPatches := int(grid.Height * grid.Width)
	patchDim := channels * temporalPatchSize * patchSize * patchSize
	result := make([]float32, numPatches*patchDim)

	patchIndex := 0
	for h := 0; h < int(grid.Height); h += mergeSize {
		for w := 0; w < int(grid.Width); w += mergeSize {
			for mh := 0; mh < mergeSize; mh++ {
				for mw := 0; mw < mergeSize; mw++ {
					baseOffset := patchIndex * patchDim

					for c := 0; c < channels; c++ {
						channelOffset := baseOffset + c*temporalPatchSize*patchSize*patchSize
						for py := 0; py < patchSize; py++ {
							for px := 0; px < patchSize; px++ {
								y := (h+mh)*patchSize + py
								x := (w+mw)*patchSize + px
								srcIdx := c*height*width + y*width + x
								dstIdx := channelOffset + py*patchSize + px
								if srcIdx < len(pixels) && dstIdx < len(result) {
									result[dstIdx] = pixels[srcIdx]
								}
							}
						}
					}

					if temporalPatchSize > 1 {
						for c := 0; c < channels; c++ {
							channelOffset := baseOffset + c*temporalPatchSize*patchSize*patchSize
							frameSize := patchSize * patchSize
							for tp := 1; tp < temporalPatchSize; tp++ {
								cur := channelOffset + tp*frameSize
								copy(result[cur:cur+frameSize], result[channelOffset:channelOffset+frameSize])
							}
						}
					}

					patchIndex++
				}
			}
		}
	}

	return result
}

// VisionAttention runs one self-attention block inside the vision encoder.
type VisionAttention struct {
	QKV    nn.LinearLayer
	Query  nn.LinearLayer
	Key    nn.LinearLayer
	Value  nn.LinearLayer
	Output nn.LinearLayer
}

func applyVisionRoPE(x, cos, sin *mlx.Array) *mlx.Array {
	return mlx.Add(mlx.Mul(x, cos), mlx.Mul(rotateHalf(x), sin))
}

func (a *VisionAttention) Forward(x, cos, sin *mlx.Array, cfg *VisionConfig) (*mlx.Array, error) {
	shape := x.Dims()
	if len(shape) != 3 {
		return nil, fmt.Errorf("vision attention expects [B,L,D], got %v", shape)
	}
	B, L, hidden := int32(shape[0]), int32(shape[1]), int32(shape[2])
	headDim := cfg.HiddenSize / cfg.NumHeads
	if headDim <= 0 {
		return nil, fmt.Errorf("invalid vision head dim: %d", headDim)
	}

	var q, k, v *mlx.Array
	if a.QKV != nil {
		qkv := a.QKV.Forward(x)
		qkv = mlx.Reshape(qkv, B, L, 3, cfg.NumHeads, headDim)
		q = mlx.Squeeze(mlx.SliceStartStop(qkv, []int32{0, 0, 0, 0, 0}, []int32{B, L, 1, cfg.NumHeads, headDim}), 2)
		k = mlx.Squeeze(mlx.SliceStartStop(qkv, []int32{0, 0, 1, 0, 0}, []int32{B, L, 2, cfg.NumHeads, headDim}), 2)
		v = mlx.Squeeze(mlx.SliceStartStop(qkv, []int32{0, 0, 2, 0, 0}, []int32{B, L, 3, cfg.NumHeads, headDim}), 2)
	} else {
		if a.Query == nil || a.Key == nil || a.Value == nil {
			return nil, errors.New("vision attention is missing q/k/v projections")
		}
		q = mlx.Reshape(a.Query.Forward(x), B, L, cfg.NumHeads, headDim)
		k = mlx.Reshape(a.Key.Forward(x), B, L, cfg.NumHeads, headDim)
		v = mlx.Reshape(a.Value.Forward(x), B, L, cfg.NumHeads, headDim)
	}

	q = applyVisionRoPE(q, cos, sin)
	k = applyVisionRoPE(k, cos, sin)

	q = mlx.Transpose(q, 0, 2, 1, 3)
	k = mlx.Transpose(k, 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attn := mlx.ScaledDotProductAttentionCausal(q, k, v, scale, false)
	attn = mlx.Reshape(mlx.Transpose(attn, 0, 2, 1, 3), B, L, hidden)
	if a.Output == nil {
		return nil, errors.New("vision attention is missing output projection")
	}
	return a.Output.Forward(attn), nil
}

// VisionMLP is the vision feed-forward block.
type VisionMLP struct {
	FC1 nn.LinearLayer
	FC2 nn.LinearLayer
}

func (m *VisionMLP) Forward(x *mlx.Array) (*mlx.Array, error) {
	if m.FC1 == nil || m.FC2 == nil {
		return nil, errors.New("vision mlp is missing fc1/fc2")
	}
	return m.FC2.Forward(mlx.GELUApprox(m.FC1.Forward(x))), nil
}

// VisionEncoderLayer is one transformer block in the vision encoder.
type VisionEncoderLayer struct {
	Norm1 *nn.LayerNorm
	Attn  *VisionAttention
	Norm2 *nn.LayerNorm
	MLP   *VisionMLP
}

func (l *VisionEncoderLayer) Forward(x, cos, sin *mlx.Array, cfg *VisionConfig) (*mlx.Array, error) {
	if l.Norm1 == nil || l.Norm2 == nil || l.Attn == nil || l.MLP == nil {
		return nil, errors.New("vision layer is incomplete")
	}

	r := x
	a, err := l.Attn.Forward(l.Norm1.Forward(x), cos, sin, cfg)
	if err != nil {
		return nil, err
	}
	x = mlx.Add(r, a)

	r = x
	m, err := l.MLP.Forward(l.Norm2.Forward(x))
	if err != nil {
		return nil, err
	}
	return mlx.Add(r, m), nil
}

// VisionPatchMerger projects merged spatial groups into language embedding space.
type VisionPatchMerger struct {
	Norm *nn.LayerNorm
	FC1  nn.LinearLayer
	FC2  nn.LinearLayer
}

func groupMergedTokens(x *mlx.Array, merge int32) (*mlx.Array, error) {
	shape := x.Dims()
	if len(shape) != 3 {
		return nil, fmt.Errorf("expected [B,L,D], got %v", shape)
	}
	if merge <= 0 {
		merge = 1
	}
	B, L, D := int32(shape[0]), int32(shape[1]), int32(shape[2])
	group := merge * merge
	if group <= 0 || L%group != 0 {
		return nil, fmt.Errorf("invalid merge layout: L=%d merge=%d", L, merge)
	}

	x = mlx.Reshape(x, B, L/group, group, D)
	x = mlx.Reshape(x, B, L/group, group*D)
	return x, nil
}

func (m *VisionPatchMerger) Forward(x *mlx.Array, cfg *VisionConfig) (*mlx.Array, error) {
	if m == nil || m.Norm == nil || m.FC1 == nil || m.FC2 == nil {
		return nil, errors.New("vision patch merger is incomplete")
	}

	x = m.Norm.Forward(x)

	var err error
	x, err = groupMergedTokens(x, cfg.SpatialMergeSize)
	if err != nil {
		return nil, err
	}

	x = m.FC2.Forward(mlx.GELUApprox(m.FC1.Forward(x)))
	return x, nil
}

// VisionModel contains the full Qwen vision tower.
type VisionModel struct {
	PatchProjection nn.LinearLayer
	PositionEmbed   *nn.Embedding
	Layers          []*VisionEncoderLayer
	PatchMerger     *VisionPatchMerger

	cfg *VisionConfig
}

func mergedPatchCoordinates(grid *VisionGrid, merge int32) [][2]int32 {
	if merge <= 0 {
		merge = 1
	}
	// Temporal is always 1 for static images; only spatial coordinates are generated.
	coords := make([][2]int32, 0, grid.Height*grid.Width)
	for h := int32(0); h < grid.Height; h += merge {
		for w := int32(0); w < grid.Width; w += merge {
			for mh := int32(0); mh < merge; mh++ {
				for mw := int32(0); mw < merge; mw++ {
					coords = append(coords, [2]int32{h + mh, w + mw})
				}
			}
		}
	}
	return coords
}

func (m *VisionModel) addPositionEmbedding(x *mlx.Array, grid *VisionGrid) (*mlx.Array, error) {
	if m.PositionEmbed == nil {
		return x, nil
	}
	shape := x.Dims()
	if len(shape) != 3 {
		return nil, fmt.Errorf("vision embeddings expect [B,L,D], got %v", shape)
	}
	B, D := int32(shape[0]), int32(shape[2])
	coords := mergedPatchCoordinates(grid, m.cfg.SpatialMergeSize)
	L := int32(len(coords))
	if L != int32(shape[1]) {
		return nil, fmt.Errorf("vision sequence mismatch: hidden L=%d coords=%d", shape[1], L)
	}

	stepH := float32(0)
	if grid.Height > 1 {
		stepH = float32(m.cfg.GridPerSide-1) / float32(grid.Height-1)
	}
	stepW := float32(0)
	if grid.Width > 1 {
		stepW = float32(m.cfg.GridPerSide-1) / float32(grid.Width-1)
	}

	indices := make([]int32, 0, L*4)
	weights := make([]float32, 0, L*4)
	for _, c := range coords {
		y := float32(c[0]) * stepH
		x0 := float32(c[1]) * stepW

		fy := int32(y)
		fx := int32(x0)
		cy := min(fy+1, m.cfg.GridPerSide-1)
		cx := min(fx+1, m.cfg.GridPerSide-1)

		indices = append(indices,
			fy*m.cfg.GridPerSide+fx,
			fy*m.cfg.GridPerSide+cx,
			cy*m.cfg.GridPerSide+fx,
			cy*m.cfg.GridPerSide+cx,
		)

		dy := y - float32(fy)
		dx := x0 - float32(fx)
		weights = append(weights,
			(1-dy)*(1-dx),
			(1-dy)*dx,
			dy*(1-dx),
			dy*dx,
		)
	}

	idxArr := mlx.FromValues(indices, int(L), 4)
	wArr := mlx.FromValues(weights, int(L), 4, 1)

	pos := m.PositionEmbed.Forward(idxArr)
	wArr = wArr.AsType(pos.DType())
	pos = mlx.Sum(mlx.Mul(pos, wArr), 1, false)
	if D != int32(pos.Dim(1)) {
		return nil, fmt.Errorf("position embedding dim mismatch: hidden=%d pos=%d", D, pos.Dim(1))
	}

	pos = mlx.ExpandDims(pos, 0)
	if B > 1 {
		pos = mlx.Tile(pos, []int32{B, 1, 1})
	}

	return mlx.Add(x, pos), nil
}

func (m *VisionModel) rotaryEmbeddings(grid *VisionGrid) (*mlx.Array, *mlx.Array, error) {
	headDim := m.cfg.HiddenSize / m.cfg.NumHeads
	if headDim <= 0 {
		return nil, nil, fmt.Errorf("invalid vision head dim: %d", headDim)
	}

	coords := mergedPatchCoordinates(grid, m.cfg.SpatialMergeSize)
	L := int32(len(coords))
	half := headDim / 2
	quarter := half / 2
	if quarter <= 0 {
		return nil, nil, fmt.Errorf("invalid vision rotary layout: head_dim=%d", headDim)
	}

	angles := make([]float32, L*headDim)
	for i, c := range coords {
		base := int32(i) * headDim
		for j := int32(0); j < quarter; j++ {
			freq := 1.0 / math.Pow(float64(m.cfg.RopeTheta), float64(2*j)/float64(half))
			angles[base+j] = float32(float64(c[0]) * freq)
			angles[base+quarter+j] = float32(float64(c[1]) * freq)
		}
		for j := int32(0); j < half; j++ {
			angles[base+half+j] = angles[base+j]
		}
	}

	arr := mlx.FromValues(angles, int(L), int(headDim))
	cos := mlx.ExpandDims(mlx.ExpandDims(mlx.Cos(arr), 0), 2)
	sin := mlx.ExpandDims(mlx.ExpandDims(mlx.Sin(arr), 0), 2)
	return cos, sin, nil
}

func (m *VisionModel) Forward(pixelValues *mlx.Array, grid *VisionGrid) (*mlx.Array, error) {
	if m == nil || pixelValues == nil || grid == nil {
		return nil, errNoVisionModel
	}
	if m.PatchProjection == nil || m.PatchMerger == nil {
		return nil, errors.New("vision model is missing required projections")
	}

	x := m.PatchProjection.Forward(pixelValues)
	var err error
	x, err = m.addPositionEmbedding(x, grid)
	if err != nil {
		return nil, err
	}

	cos, sin, err := m.rotaryEmbeddings(grid)
	if err != nil {
		return nil, err
	}

	for i, layer := range m.Layers {
		x, err = layer.Forward(x, cos, sin, m.cfg)
		if err != nil {
			return nil, fmt.Errorf("vision layer %d: %w", i, err)
		}
	}

	main, err := m.PatchMerger.Forward(x, m.cfg)
	if err != nil {
		return nil, fmt.Errorf("vision patch merger: %w", err)
	}
	return main, nil
}

type VisionEmbeddings struct {
	Main *mlx.Array
	Grid *VisionGrid
}

func (m *Model) EncodeVisionImage(multimodalData []byte) (*VisionEmbeddings, error) {
	if m == nil || m.Vision == nil || m.ImageProcessor == nil {
		return nil, errNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	pixelValues, grid, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, err
	}

	main, err := m.Vision.Forward(pixelValues, grid)
	if err != nil {
		return nil, err
	}

	return &VisionEmbeddings{Main: main, Grid: grid}, nil
}

func resolveVisionPrefix(tensors map[string]*mlx.Array, weightPrefix string) string {
	candidates := []string{
		"vision_tower",
		weightPrefix + "vision_tower",
		"model.visual",
		"visual",
		weightPrefix + "model.visual",
		weightPrefix + "visual",
	}

	hasTensor := func(prefix string) bool {
		for _, suffix := range []string{
			".patch_embed.proj.weight",
			".patch_embed.weight",
			".pos_embed.weight",
			".blocks.0.attn.qkv.weight",
			".merger.linear_fc1.weight",
			".merger.mlp.0.weight",
		} {
			if tensors[prefix+suffix] != nil {
				return true
			}
		}
		return false
	}

	for _, prefix := range candidates {
		if hasTensor(prefix) {
			return prefix
		}
	}

	return ""
}

func firstLinear(linears mlxmodel.LinearFactory, paths ...string) nn.LinearLayer {
	for _, p := range paths {
		if l := linears.Make(p); l != nil {
			return l
		}
	}
	return nil
}

func loadLayerNorm(tensors map[string]*mlx.Array, eps float32, bases ...string) *nn.LayerNorm {
	for _, base := range bases {
		if w := tensors[base+".weight"]; w != nil {
			return &nn.LayerNorm{Weight: w, Bias: tensors[base+".bias"], Eps: eps}
		}
		if w := tensors[base]; w != nil {
			return &nn.LayerNorm{Weight: w, Bias: tensors[base+"_bias"], Eps: eps}
		}
	}
	return nil
}

func loadVisionPatchMerger(
	tensors map[string]*mlx.Array,
	linears mlxmodel.LinearFactory,
	eps float32,
	bases ...string,
) *VisionPatchMerger {
	for _, base := range bases {
		norm := loadLayerNorm(tensors, eps, base+".norm", base+".ln_q")
		fc1 := firstLinear(linears, base+".linear_fc1", base+".mlp.0")
		fc2 := firstLinear(linears, base+".linear_fc2", base+".mlp.2")
		if norm != nil && fc1 != nil && fc2 != nil {
			return &VisionPatchMerger{Norm: norm, FC1: fc1, FC2: fc2}
		}
	}
	return nil
}

func flattenPatchEmbeddingWeight(w *mlx.Array) (*mlx.Array, error) {
	if w == nil || !w.Valid() {
		return nil, errors.New("missing patch embedding weight")
	}
	if w.NumDims() < 2 {
		return nil, fmt.Errorf("patch embedding weight must be >=2D, got %dD", w.NumDims())
	}
	if w.NumDims() == 2 {
		return w, nil
	}

	out := int32(w.Dim(0))
	in := int32(w.Size() / w.Dim(0))
	return mlx.Reshape(w, out, in), nil
}

func loadVisionComponents(
	tensors map[string]*mlx.Array,
	linears mlxmodel.LinearFactory,
	cfg *Config,
	weightPrefix string,
) (*VisionModel, *VisionImageProcessor, error) {
	if cfg == nil || cfg.Vision == nil || cfg.Vision.Depth <= 0 {
		return nil, nil, nil
	}
	cfg.Vision.applyDefaults()

	visionPrefix := resolveVisionPrefix(tensors, weightPrefix)
	if visionPrefix == "" {
		return nil, nil, errors.New("vision enabled in config but vision tensors were not found")
	}

	patchW, _ := tensorAny(
		tensors,
		visionPrefix+".patch_embed.proj.weight",
		visionPrefix+".patch_embed.weight",
	)
	if patchW == nil {
		return nil, nil, fmt.Errorf("missing vision patch embedding weight under %s", visionPrefix)
	}
	patchW, err := flattenPatchEmbeddingWeight(patchW)
	if err != nil {
		return nil, nil, err
	}
	patchB, _ := tensorAny(
		tensors,
		visionPrefix+".patch_embed.proj.bias",
		visionPrefix+".patch_embed.bias",
	)

	patchProj := nn.NewLinear(patchW, patchB)
	if got := int32(patchW.Dim(1)); got != cfg.Vision.InChannels*cfg.Vision.TemporalPatchSize*cfg.Vision.PatchSize*cfg.Vision.PatchSize {
		return nil, nil, fmt.Errorf(
			"vision patch embedding input dim mismatch: got %d expected %d",
			got,
			cfg.Vision.InChannels*cfg.Vision.TemporalPatchSize*cfg.Vision.PatchSize*cfg.Vision.PatchSize,
		)
	}

	posW, _ := tensorAny(
		tensors,
		visionPrefix+".pos_embed.weight",
		visionPrefix+".position_embedding.weight",
	)
	if posW == nil {
		return nil, nil, fmt.Errorf("missing vision position embedding under %s", visionPrefix)
	}
	cfg.Vision.NumPositionEmbeddings = int32(posW.Dim(0))
	cfg.Vision.applyDefaults()

	vm := &VisionModel{
		PatchProjection: patchProj,
		PositionEmbed:   nn.NewEmbedding(posW),
		Layers:          make([]*VisionEncoderLayer, cfg.Vision.Depth),
		cfg:             cfg.Vision,
	}

	for i := int32(0); i < cfg.Vision.Depth; i++ {
		layerPrefix := fmt.Sprintf("%s.blocks.%d", visionPrefix, i)
		layer := &VisionEncoderLayer{
			Norm1: loadLayerNorm(tensors, cfg.Vision.LayerNormEpsilon, layerPrefix+".norm1"),
			Norm2: loadLayerNorm(tensors, cfg.Vision.LayerNormEpsilon, layerPrefix+".norm2"),
			Attn: &VisionAttention{
				QKV: firstLinear(
					linears,
					layerPrefix+".attn.qkv",
					layerPrefix+".attn_qkv",
				),
				Query: firstLinear(
					linears,
					layerPrefix+".attn.q_proj",
					layerPrefix+".attn_q",
				),
				Key: firstLinear(
					linears,
					layerPrefix+".attn.k_proj",
					layerPrefix+".attn_k",
				),
				Value: firstLinear(
					linears,
					layerPrefix+".attn.v_proj",
					layerPrefix+".attn_v",
				),
				Output: firstLinear(
					linears,
					layerPrefix+".attn.proj",
					layerPrefix+".attn_out",
					layerPrefix+".attn.o_proj",
				),
			},
			MLP: &VisionMLP{
				FC1: firstLinear(
					linears,
					layerPrefix+".mlp.fc1",
					layerPrefix+".mlp.linear_fc1",
				),
				FC2: firstLinear(
					linears,
					layerPrefix+".mlp.fc2",
					layerPrefix+".mlp.linear_fc2",
				),
			},
		}

		if layer.Norm1 == nil || layer.Norm2 == nil {
			return nil, nil, fmt.Errorf("vision layer %d: missing norm1/norm2", i)
		}
		if layer.Attn.Output == nil || (layer.Attn.QKV == nil && (layer.Attn.Query == nil || layer.Attn.Key == nil || layer.Attn.Value == nil)) {
			return nil, nil, fmt.Errorf("vision layer %d: missing attention projections", i)
		}
		if layer.MLP.FC1 == nil || layer.MLP.FC2 == nil {
			return nil, nil, fmt.Errorf("vision layer %d: missing mlp projections", i)
		}

		vm.Layers[i] = layer
	}

	vm.PatchMerger = loadVisionPatchMerger(
		tensors,
		linears,
		cfg.Vision.LayerNormEpsilon,
		visionPrefix+".merger",
	)
	if vm.PatchMerger == nil {
		return nil, nil, fmt.Errorf("missing vision patch merger under %s", visionPrefix)
	}

	return vm, newVisionImageProcessor(cfg.Vision), nil
}
