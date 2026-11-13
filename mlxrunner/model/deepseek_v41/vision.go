package deepseek_v41

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type visionBlock struct {
	norm1, norm2 *nn.RMSNorm
	qkv, output  nn.LinearLayer
	gateUp, down nn.LinearLayer
}

type visionTower struct {
	config                visionConfig
	patch, align1, align2 nn.LinearLayer
	blocks                []visionBlock
	norm                  *nn.RMSNorm
	start, end, newline   *mlx.Array
}

func loadVision(tensors map[string]*mlx.Array, cfg *Config) (*visionTower, error) {
	if cfg.Vision.ModelType == "" {
		if tensors["vision.patch_embed.proj.weight"] != nil {
			return nil, fmt.Errorf("DeepSeek vision weights require vision_config")
		}
		return nil, nil
	}
	required := func(name string) (*mlx.Array, error) { return requireTensor(tensors, name) }
	linear := func(name string, rows, cols int, bias bool) (nn.LinearLayer, error) {
		w, err := required(name + ".weight")
		if err != nil {
			return nil, err
		}
		if w.NumDims() != 2 || w.Dim(0) != rows || w.Dim(1) != cols ||
			(w.DType() != mlx.DTypeBFloat16 && w.DType() != mlx.DTypeFloat32) {
			return nil, fmt.Errorf("invalid DeepSeek vision projection %s: shape %v, dtype %v", name, w.Dims(), w.DType())
		}
		var b *mlx.Array
		if bias {
			b, err = required(name + ".bias")
			if err != nil {
				return nil, err
			}
			if b.NumDims() != 1 || b.Size() != rows {
				return nil, fmt.Errorf("invalid DeepSeek vision bias %s: shape %v", name, b.Dims())
			}
		}
		return nn.NewLinear(w, b), nil
	}
	norm := func(name string, width int) (*nn.RMSNorm, error) {
		w, err := required(name + ".weight")
		if err != nil {
			return nil, err
		}
		if w.NumDims() != 1 || w.Size() != width {
			return nil, fmt.Errorf("invalid DeepSeek vision norm %s: shape %v", name, w.Dims())
		}
		return nn.NewRMSNorm(w, 1e-6), nil
	}
	v := &visionTower{config: cfg.Vision, blocks: make([]visionBlock, cfg.Vision.NumHiddenLayers)}
	width := cfg.Vision.HiddenSize
	inner := cfg.Vision.IntermediateSize
	patchWidth := 3 * cfg.Vision.PatchSize * cfg.Vision.PatchSize
	alignWidth := width * cfg.Vision.DownsampleRatio * cfg.Vision.DownsampleRatio
	var err error
	if v.patch, err = linear("vision.patch_embed.proj", width, patchWidth, true); err != nil {
		return nil, err
	}
	if v.norm, err = norm("vision.norm", width); err != nil {
		return nil, err
	}
	for i := range v.blocks {
		prefix := fmt.Sprintf("vision.blocks.%d", i)
		b := &v.blocks[i]
		if b.norm1, err = norm(prefix+".norm1", width); err != nil {
			return nil, err
		}
		if b.norm2, err = norm(prefix+".norm2", width); err != nil {
			return nil, err
		}
		if b.qkv, err = linear(prefix+".attn.wqkv", 3*width, width, true); err != nil {
			return nil, err
		}
		if b.output, err = linear(prefix+".attn.wo", width, width, true); err != nil {
			return nil, err
		}
		if b.gateUp, err = linear(prefix+".mlp.w1", 2*inner, width, false); err != nil {
			return nil, err
		}
		if b.down, err = linear(prefix+".mlp.w2", width, inner, false); err != nil {
			return nil, err
		}
	}
	if v.align1, err = linear("aligner.w1", cfg.HiddenSize, alignWidth, true); err != nil {
		return nil, err
	}
	if v.align2, err = linear("aligner.w2", cfg.HiddenSize, cfg.HiddenSize, true); err != nil {
		return nil, err
	}
	if v.start, err = required("image_start"); err != nil {
		return nil, err
	}
	if v.end, err = required("image_end"); err != nil {
		return nil, err
	}
	if v.newline, err = required("image_newline"); err != nil {
		return nil, err
	}
	if v.start.Size() != cfg.HiddenSize || v.end.Size() != cfg.HiddenSize || v.newline.Size() != cfg.HiddenSize {
		return nil, fmt.Errorf("DeepSeek image delimiter widths do not match text hidden size")
	}
	return v, nil
}

func visionRotary(g imageGeometry, headDim int, theta float64) (*mlx.Array, *mlx.Array) {
	half := headDim / 2
	frequencyCount := half / 2
	positions := make([]float32, g.patchRows*g.patchCols*headDim)
	for row := range g.patchRows {
		for col := range g.patchCols {
			dst := positions[(row*g.patchCols+col)*headDim:]
			for j := range frequencyCount {
				freq := float32(1 / math.Pow(theta, float64(2*j)/float64(half)))
				dst[j] = float32(row) * freq
				dst[frequencyCount+j] = float32(col) * freq
			}
			copy(dst[half:], dst[:half])
		}
	}
	angles := mlx.FromValues(positions, 1, g.patchRows*g.patchCols, 1, headDim)
	return mlx.Transpose(mlx.Cos(angles), 0, 2, 1, 3), mlx.Transpose(mlx.Sin(angles), 0, 2, 1, 3)
}

func rotateVision(x, cos, sin *mlx.Array) *mlx.Array {
	dtype := x.DType()
	x = x.AsType(mlx.DTypeFloat32)
	width := x.Dim(3)
	left := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, width/2))
	right := x.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(width/2, width))
	rotated := mlx.Concatenate([]*mlx.Array{mlx.Neg(right), left}, -1)
	return mlx.Add(mlx.Mul(x, cos), mlx.Mul(rotated, sin)).AsType(dtype)
}

func (v *visionTower) forward(patches *mlx.Array, g imageGeometry) *mlx.Array {
	n := g.patchRows * g.patchCols
	width := v.config.HiddenSize
	heads := v.config.NumAttentionHeads
	dim := width / heads
	x := mlx.Reshape(v.patch.Forward(patches.AsType(v.start.DType())), 1, int32(n), int32(width))
	cos, sin := visionRotary(g, dim, v.config.RoPETheta)
	for _, block := range v.blocks {
		normalized := block.norm1.Forward(x, 0)
		qkv := mlx.Reshape(block.qkv.Forward(normalized), 1, int32(n), 3, int32(heads), int32(dim))
		part := func(i int) *mlx.Array {
			t := qkv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice())
			return mlx.Transpose(mlx.Squeeze(t, 2), 0, 2, 1, 3)
		}
		q, k, value := rotateVision(part(0), cos, sin), rotateVision(part(1), cos, sin), part(2)
		attended := mlx.FastScaledDotProductAttention(q, k, value, float32(1/math.Sqrt(float64(dim))), mlx.FastSDPANoWindow, "", nil)
		attended = mlx.Reshape(mlx.Transpose(attended, 0, 2, 1, 3), 1, int32(n), int32(width))
		x = mlx.Add(x, block.output.Forward(attended))
		gateUp := block.gateUp.Forward(block.norm2.Forward(x, 0))
		gate := gateUp.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, v.config.IntermediateSize))
		up := gateUp.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(v.config.IntermediateSize, 2*v.config.IntermediateSize))
		x = mlx.Add(x, block.down.Forward(mlx.Mul(mlx.Mul(gate, mlx.Sigmoid(gate)), up)))
	}
	x = mlx.Reshape(v.norm.Forward(x, 0), int32(g.patchRows), int32(g.patchCols), int32(width))
	ratio := v.config.DownsampleRatio
	if pad := g.textRows*ratio - g.patchRows; pad > 0 {
		x = mlx.PadConstant(x, []int{0}, []int{0}, []int{pad})
	}
	if pad := g.textCols*ratio - g.patchCols; pad > 0 {
		x = mlx.PadConstant(x, []int{1}, []int{0}, []int{pad})
	}
	x = mlx.Reshape(x, int32(g.textRows), int32(ratio), int32(g.textCols), int32(ratio), int32(width))
	x = mlx.Reshape(mlx.Transpose(x, 0, 2, 4, 1, 3), int32(g.textRows*g.textCols), int32(width*ratio*ratio))
	return v.align2.Forward(mlx.GELU(v.align1.Forward(x).AsType(mlx.DTypeFloat32)).AsType(x.DType()))
}
