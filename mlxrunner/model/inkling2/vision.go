package inkling2

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"math"
	"slices"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type visionConfig struct {
	Patch     int     `json:"native_patch_size"`
	Merge     int     `json:"adapter_merge_size"`
	Temporal  int     `json:"temporal_patch_size"`
	Projector int     `json:"projection_hidden_dim"`
	Width     int     `json:"hidden_size"`
	Layers    int     `json:"num_hidden_layers"`
	Heads     int     `json:"num_attention_heads"`
	MLPRatio  float64 `json:"mlp_ratio"`
	Grid      int     `json:"posemb_grid_size"`
	Epsilon   float32 `json:"layer_norm_eps"`
	ProjBias  bool    `json:"projector_bias"`
}

func (c visionConfig) validate() error {
	if c.Patch <= 0 || c.Merge <= 0 || c.Temporal <= 0 || c.Projector <= 0 || c.Width <= 0 || c.Layers <= 0 ||
		c.Heads <= 0 || c.Width%c.Heads != 0 || (c.Width/c.Heads)%4 != 0 || c.MLPRatio <= 0 || c.Grid <= 0 || c.Epsilon <= 0 {
		return fmt.Errorf("invalid Inkling2 vision dimensions")
	}
	if c.Layers > maxLayerCount || c.Patch > math.MaxInt32/c.Merge {
		return fmt.Errorf("Inkling2 vision layer count %d or patch dimensions %d × %d exceed their limit", c.Layers, c.Patch, c.Merge)
	}
	return nil
}

type imageGrid struct{ rows, cols int }

type visionEncoder struct {
	config      visionConfig
	imageToken  int32
	Patch       *nn.Linear
	Class       *mlx.Array
	ClassOffset *mlx.Array
	Positions   *mlx.Array
	PreNorm     *nn.LayerNorm
	Layers      []visionLayer
	Project     *nn.Linear
	Out         *nn.Linear
}

type visionLayer struct {
	Norm, FfNorm *nn.LayerNorm
	Q, K, V, O   *nn.Linear
	Up, Down     *nn.Linear
	AttnScale    *mlx.Array
	FfScale      *mlx.Array
}

// Inkling uses CLIP normalization before rounding the pixels to the vision
// tower's input dtype. The patch encoder converts these back to [-1,1].
var (
	imageMean = []float32{0.48145466, 0.4578275, 0.40821073}
	imageStd  = []float32{0.26862954, 0.26130258, 0.27577711}
)

func (m *targetModel) PrepareMedia(segments []model.Segment) (*model.PreparedRequest, error) {
	if m.Vision == nil {
		return nil, fmt.Errorf("Inkling2 checkpoint does not have a vision encoder")
	}
	side := m.Vision.config.Patch * m.Vision.config.Merge
	remaining := m.config.Context - 1 // leave a token for generation
	for _, segment := range segments {
		remaining -= len(segment.Tokens)
	}
	var prepared model.PreparedRequest
	for i, segment := range segments {
		if segment.Data == nil {
			prepared.Tokens = append(prepared.Tokens, segment.Tokens...)
			continue
		}
		if segment.Kind != "image" {
			if segment.Kind == "" {
				return nil, fmt.Errorf("unrecognized Inkling2 media at input segment %d", i)
			}
			return nil, fmt.Errorf("Inkling2 does not support %s inputs", segment.Kind)
		}
		info, _, err := image.DecodeConfig(bytes.NewReader(segment.Data))
		if err != nil {
			return nil, fmt.Errorf("read Inkling2 image dimensions: %w", err)
		}
		if info.Width <= 0 || info.Height <= 0 {
			return nil, fmt.Errorf("invalid Inkling2 image dimensions %d × %d", info.Width, info.Height)
		}
		grid := imageGrid{rows: (info.Height-1)/side + 1, cols: (info.Width-1)/side + 1}
		if grid.rows > remaining/grid.cols {
			return nil, fmt.Errorf("Inkling2 image %d × %d exceeds the remaining context", info.Width, info.Height)
		}
		if grid.rows > math.MaxInt32/side || grid.cols > math.MaxInt32/side ||
			grid.rows > math.MaxInt/3/side/side/grid.cols {
			return nil, fmt.Errorf("Inkling2 image %d × %d exceeds representable dimensions", info.Width, info.Height)
		}
		tokens := grid.rows * grid.cols
		remaining -= tokens
		img, _, err := image.Decode(bytes.NewReader(segment.Data))
		if err != nil {
			return nil, fmt.Errorf("decode Inkling2 image: %w", err)
		}
		if bounds := img.Bounds(); bounds.Dx() != info.Width || bounds.Dy() != info.Height {
			return nil, fmt.Errorf("Inkling2 image dimensions changed during decoding")
		}
		pixels := imagePixels(img, side, grid)
		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, slices.Repeat([]int32{m.Vision.imageToken}, tokens)...)
		prepared.Items = append(prepared.Items, model.PreparedItem{
			Range: [2]int{start, len(prepared.Tokens)}, Source: i, Causal: true,
			MediaData: pixels, Dims: []int{grid.rows * side, grid.cols * side, 3}, Opaque: grid,
		})
	}
	return &prepared, nil
}

func imagePixels(img image.Image, tile int, grid imageGrid) []float32 {
	b := img.Bounds()
	height, width := grid.rows*tile, grid.cols*tile
	pixels := make([]float32, height*width*3)
	for y := range height {
		for x := range width {
			rgb := [3]int{-1, -1, -1}
			if x < b.Dx() && y < b.Dy() {
				p := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				bg := 204
				if ((x/8)+(y/8))%2 == 1 {
					bg = 153
				}
				alpha := int(p.A)
				for channel, value := range [3]uint8{p.R, p.G, p.B} {
					rgb[channel] = (int(value)*alpha + bg*(255-alpha) + 127) / 255
				}
			}
			for channel, value := range rgb {
				pixels[3*(y*width+x)+channel] = (float32(value)/255 - imageMean[channel]) / imageStd[channel]
			}
		}
	}
	return pixels
}

func (m *targetModel) EncodeMedia(item *model.PreparedItem, data *mlx.Array) *mlx.Array {
	return m.Vision.encode(data, item.Opaque.(imageGrid))
}

func (m *targetModel) imageEmbeddings(b *batch.Batch, embeddings *mlx.Array) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		offset := int(b.SeqOffsets[item.Seq])
		start := max(item.Pos, offset)
		end := min(item.Pos+item.Features.Dim(0), offset+int(b.SeqQueryLens[item.Seq]))
		if start >= end {
			continue
		}
		feature := item.Features.Slice(mlx.Slice(start-item.Pos, end-item.Pos), mlx.Slice()).ExpandDims(0)
		embeddings = embeddings.SliceUpdate(feature, mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(start-offset, end-offset), mlx.Slice())
	}
	return embeddings
}

func (v *visionEncoder) encode(data *mlx.Array, grid imageGrid) *mlx.Array {
	c := v.config
	patchRows, patchCols := grid.rows*c.Merge, grid.cols*c.Merge
	data = data.AsType(mlx.DTypeBFloat16)
	mean := mlx.FromValues(imageMean, 3).AsType(mlx.DTypeBFloat16)
	std := mlx.FromValues(imageStd, 3).AsType(mlx.DTypeBFloat16)
	data = mlx.AddScalar(mlx.MulScalar(mlx.Add(mlx.Mul(data, std), mean), 2), -1)
	// A still image has identical temporal frames, so their mean is the
	// single preprocessed frame.
	patches := data.Reshape(patchRows, c.Patch, patchCols, c.Patch, 3).Transpose(0, 2, 1, 3, 4)
	h := v.Patch.Forward(patches.Reshape(patchRows*patchCols, c.Patch*c.Patch*3))
	h = mlx.Add(h, interpolatePositions(v.Positions, c.Grid, patchRows, patchCols))
	h = mlx.Concatenate([]*mlx.Array{mlx.Add(v.Class, v.ClassOffset).ExpandDims(0), h}, 0)
	h = v.PreNorm.Forward(h)
	cos, sin := imagePhases(patchRows, patchCols, c.Width/c.Heads)
	for _, layer := range v.Layers {
		h = layer.forward(h, cos, sin, c.Heads)
	}
	patchOutput := h.Slice(mlx.Slice(1, h.Dim(0)), mlx.Slice())
	patchOutput = patchOutput.Reshape(grid.rows, c.Merge, grid.cols, c.Merge, c.Width).Transpose(0, 2, 1, 3, 4)
	patchOutput = patchOutput.Reshape(grid.rows*grid.cols, c.Merge*c.Merge*c.Width)
	return v.Out.Forward(mlx.GELU(v.Project.Forward(patchOutput)))
}

func interpolatePositions(table *mlx.Array, size, rows, cols int) *mlx.Array {
	indices := make([]int32, 0, 4*rows*cols)
	weights := make([]float32, 0, 4*rows*cols)
	for row := range rows {
		for col := range cols {
			y := min(float64(size-1), max(0, (float64(row)+0.5)*float64(size)/float64(rows)-0.5))
			x := min(float64(size-1), max(0, (float64(col)+0.5)*float64(size)/float64(cols)-0.5))
			y0, x0 := int(y), int(x)
			y1, x1 := min(y0+1, size-1), min(x0+1, size-1)
			dy, dx := float32(y-float64(y0)), float32(x-float64(x0))
			indices = append(indices, int32(y0*size+x0), int32(y0*size+x1), int32(y1*size+x0), int32(y1*size+x1))
			weights = append(weights, (1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx)
		}
	}
	selected := table.TakeAxis(mlx.FromValues(indices, rows*cols, 4), 0).AsType(mlx.DTypeFloat32)
	return mlx.Sum(mlx.Mul(selected, mlx.FromValues(weights, rows*cols, 4, 1)), 1, false).AsType(table.DType())
}

func imagePhases(rows, cols, headDim int) (cos, sin *mlx.Array) {
	count := rows*cols + 1
	x, y := make([]float32, count), make([]float32, count)
	for i := range count - 1 {
		x[i+1], y[i+1] = float32(i%cols+1), float32(i/cols+1)
	}
	frequencies := make([]float32, headDim/4)
	for i := range frequencies {
		frequencies[i] = float32(math.Pow(10000, -4*float64(i)/float64(headDim)))
	}
	freq := mlx.FromValues(frequencies, 1, len(frequencies))
	angles := mlx.Concatenate([]*mlx.Array{mlx.Mul(mlx.FromValues(x, count, 1), freq), mlx.Mul(mlx.FromValues(y, count, 1), freq)}, -1)
	return mlx.Cos(angles).ExpandDims(1), mlx.Sin(angles).ExpandDims(1)
}

func applyImageRotary(x, cos, sin *mlx.Array) *mlx.Array {
	count, heads, dim := x.Dim(0), x.Dim(1), x.Dim(2)
	values := x.AsType(mlx.DTypeFloat32)
	even := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, dim, 2))
	odd := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(1, dim, 2))
	real := mlx.Sub(mlx.Mul(even, cos), mlx.Mul(odd, sin))
	imag := mlx.Add(mlx.Mul(even, sin), mlx.Mul(odd, cos))
	return mlx.Stack([]*mlx.Array{real, imag}, -1).Reshape(count, heads, dim).AsType(x.DType())
}

func (l visionLayer) forward(h, cos, sin *mlx.Array, heads int) *mlx.Array {
	count, width := h.Dim(0), h.Dim(1)
	dim := width / heads
	n := l.Norm.Forward(h)
	q := applyImageRotary(l.Q.Forward(n).Reshape(count, heads, dim), cos, sin).Transpose(1, 0, 2).ExpandDims(0)
	k := applyImageRotary(l.K.Forward(n).Reshape(count, heads, dim), cos, sin).Transpose(1, 0, 2).ExpandDims(0)
	v := l.V.Forward(n).Reshape(count, heads, dim).Transpose(1, 0, 2).ExpandDims(0)
	a := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(dim))), mlx.FastSDPANoWindow, "", nil)
	a = a.Squeeze(0).Transpose(1, 0, 2).Reshape(count, width)
	h = mlx.Add(h, mlx.Mul(l.O.Forward(a), l.AttnScale))
	f := l.Down.Forward(mlx.GELU(l.Up.Forward(l.FfNorm.Forward(h))))
	return mlx.Add(h, mlx.Mul(f, l.FfScale))
}
