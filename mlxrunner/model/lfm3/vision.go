package lfm3

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type VisionModel struct {
	Config              visionConfig
	ImageTokenID        int32
	DownsampleFactor    int32
	ProjectorHiddenSize int32
	ProjectorNormEps    float32
	ProjectorAct        string
	MinPixels           int32
	MaxPixels           int32
	Patchifier          nn.LinearLayer
	EmbeddingNorm       *nn.RMSNorm
	Layers              []*visionLayer
	ProjectorNorm       *nn.LayerNorm
	ProjectorLinear1    nn.LinearLayer
	ProjectorLinear2    nn.LinearLayer
	InvFreq             *mlx.Array
}

type visionLayer struct {
	OperatorNorm *nn.RMSNorm
	FFNNorm      *nn.RMSNorm
	MLP          *MLP
	Conv         *visionConv
	Attention    *visionAttention
}

type visionConv struct {
	InProj  nn.LinearLayer
	OutProj nn.LinearLayer
	Weight  *mlx.Array
}

type visionAttention struct {
	QProj   nn.LinearLayer
	KProj   nn.LinearLayer
	VProj   nn.LinearLayer
	OutProj nn.LinearLayer
	QNorm   *nn.RMSNorm
	KNorm   *nn.RMSNorm
}

func (v *VisionModel) LoadWeights(tensors map[string]*mlx.Array) error {
	c := &v.Config
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || len(c.LayerTypes) != int(c.NumHiddenLayers) ||
		c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 ||
		c.HiddenSize%c.NumAttentionHeads != 0 || c.IntermediateSize <= 0 || c.NormEps <= 0 ||
		c.PatchSize <= 0 || c.SpatialMergeSize <= 0 || c.ConvKernelSize <= 0 || c.RopeParameters.Theta <= 0 {
		return fmt.Errorf("invalid LFM3 VL vision dimensions")
	}
	if v.DownsampleFactor != c.SpatialMergeSize || v.ProjectorAct != "gelu" ||
		v.ProjectorHiddenSize <= 0 || v.ProjectorNormEps <= 0 {
		return fmt.Errorf("unsupported LFM3 VL projector configuration")
	}
	required := func(name string) (*mlx.Array, error) {
		w := tensors[name]
		if w == nil {
			return nil, fmt.Errorf("missing VL weight %s", name)
		}
		return w, nil
	}
	linear := func(prefix string) (nn.LinearLayer, error) {
		w, err := required(prefix + ".weight")
		if err != nil {
			return nil, err
		}
		return nn.NewLinear(w, tensors[prefix+".bias"]), nil
	}
	var err error
	if v.Patchifier, err = linear("model.vision_tower.patchifier"); err != nil {
		return err
	}
	w, err := required("model.vision_tower.embedding_norm.weight")
	if err != nil {
		return err
	}
	v.EmbeddingNorm = nn.NewRMSNorm(w, c.NormEps)
	v.Layers = make([]*visionLayer, c.NumHiddenLayers)
	for i, kind := range c.LayerTypes {
		prefix := fmt.Sprintf("model.vision_tower.layers.%d.", i)
		layer := &visionLayer{MLP: &MLP{}}
		if w, err = required(prefix + "operator_norm.weight"); err != nil {
			return err
		}
		layer.OperatorNorm = nn.NewRMSNorm(w, c.NormEps)
		if w, err = required(prefix + "ffn_norm.weight"); err != nil {
			return err
		}
		layer.FFNNorm = nn.NewRMSNorm(w, c.NormEps)
		if layer.MLP.W1, err = linear(prefix + "feed_forward.w1"); err != nil {
			return err
		}
		if layer.MLP.W3, err = linear(prefix + "feed_forward.w3"); err != nil {
			return err
		}
		if layer.MLP.W2, err = linear(prefix + "feed_forward.w2"); err != nil {
			return err
		}
		switch kind {
		case "conv":
			layer.Conv = &visionConv{}
			if layer.Conv.InProj, err = linear(prefix + "conv.in_proj"); err != nil {
				return err
			}
			if layer.Conv.OutProj, err = linear(prefix + "conv.out_proj"); err != nil {
				return err
			}
			if w, err = required(prefix + "conv.spatial_conv.weight"); err != nil {
				return err
			}
			layer.Conv.Weight = mlx.Transpose(w, 0, 2, 3, 1)
		case "full_attention":
			layer.Attention = &visionAttention{}
			a := layer.Attention
			if a.QProj, err = linear(prefix + "self_attn.q_proj"); err != nil {
				return err
			}
			if a.KProj, err = linear(prefix + "self_attn.k_proj"); err != nil {
				return err
			}
			if a.VProj, err = linear(prefix + "self_attn.v_proj"); err != nil {
				return err
			}
			if a.OutProj, err = linear(prefix + "self_attn.out_proj"); err != nil {
				return err
			}
			if w, err = required(prefix + "self_attn.q_layernorm.weight"); err != nil {
				return err
			}
			a.QNorm = nn.NewRMSNorm(w, c.NormEps)
			if w, err = required(prefix + "self_attn.k_layernorm.weight"); err != nil {
				return err
			}
			a.KNorm = nn.NewRMSNorm(w, c.NormEps)
		default:
			return fmt.Errorf("unsupported VL vision layer %q at %d", kind, i)
		}
		v.Layers[i] = layer
	}
	projector := "model.multi_modal_projector."
	w, err = required(projector + "layer_norm.weight")
	if err != nil {
		return err
	}
	bias, err := required(projector + "layer_norm.bias")
	if err != nil {
		return err
	}
	v.ProjectorNorm = &nn.LayerNorm{Weight: w, Bias: bias, Eps: v.ProjectorNormEps}
	if v.ProjectorLinear1, err = linear(projector + "linear_1"); err != nil {
		return err
	}
	if v.ProjectorLinear2, err = linear(projector + "linear_2"); err != nil {
		return err
	}
	spatialDim := c.HiddenSize / c.NumAttentionHeads / 2
	frequencies := make([]float32, spatialDim/2)
	for i := range frequencies {
		frequencies[i] = float32(math.Pow(float64(c.RopeParameters.Theta), -float64(2*i)/float64(spatialDim)))
	}
	v.InvFreq = mlx.FromValues(frequencies, 1, len(frequencies))
	return nil
}

func (v *VisionModel) PositionIDs(height, width int32) *mlx.Array {
	merge := v.Config.SpatialMergeSize
	coords := make([]float32, 0, 2*height*width)
	for blockH := int32(0); blockH < height; blockH += merge {
		for blockW := int32(0); blockW < width; blockW += merge {
			for dh := range merge {
				for dw := range merge {
					coords = append(coords, float32(blockH+dh), float32(blockW+dw))
				}
			}
		}
	}
	return mlx.FromValues(coords, int(height*width), 2)
}

func (v *VisionModel) rotaryAngles(positionIDs *mlx.Array) (cos, sin *mlx.Array) {
	n := int32(positionIDs.Dim(0))
	h := mlx.SliceStartStop(positionIDs, []int32{0, 0}, []int32{n, 1})
	w := mlx.SliceStartStop(positionIDs, []int32{0, 1}, []int32{n, 2})
	angles := mlx.Concatenate([]*mlx.Array{mlx.Mul(h, v.InvFreq), mlx.Mul(w, v.InvFreq)}, 1)
	angles = mlx.Concatenate([]*mlx.Array{angles, angles}, 1)
	return mlx.Cos(angles), mlx.Sin(angles)
}

func rotateVision(x, cos, sin *mlx.Array) *mlx.Array {
	dims := int32(x.Dim(2))
	n := int32(x.Dim(0))
	heads := int32(x.Dim(1))
	left := mlx.SliceStartStop(x, []int32{0, 0, 0}, []int32{n, heads, dims / 2})
	right := mlx.SliceStartStop(x, []int32{0, 0, dims / 2}, []int32{n, heads, dims})
	rotated := mlx.Concatenate([]*mlx.Array{right.Negative(), left}, 2).AsType(mlx.DTypeFloat32)
	floatX := x.AsType(mlx.DTypeFloat32)
	cos = mlx.ExpandDims(cos, 1)
	sin = mlx.ExpandDims(sin, 1)
	return mlx.Add(mlx.Mul(floatX, cos), mlx.Mul(rotated, sin)).AsType(x.DType())
}

func (v *VisionModel) Embed(pixels *mlx.Array) *mlx.Array {
	h := v.Patchifier.Forward(pixels.AsType(mlx.DTypeBFloat16))
	return v.EmbeddingNorm.Forward(h, v.Config.NormEps)
}

func (l *visionLayer) Forward(x *mlx.Array, height, width int32, cos, sin *mlx.Array, cfg *visionConfig) *mlx.Array {
	normalized := l.OperatorNorm.Forward(x, cfg.NormEps)
	var op *mlx.Array
	if l.Conv != nil {
		op = l.Conv.Forward(normalized, height, width, cfg)
	} else {
		op = l.Attention.Forward(normalized, cos, sin, cfg)
	}
	h := mlx.Add(x, op)
	return mlx.Add(h, l.MLP.Forward(l.FFNNorm.Forward(h, cfg.NormEps)))
}

func (c *visionConv) Forward(x *mlx.Array, height, width int32, cfg *visionConfig) *mlx.Array {
	n := height * width
	d := cfg.HiddenSize
	projection := c.InProj.Forward(x)
	gateIn := mlx.SliceStartStop(projection, []int32{0, 0}, []int32{n, d})
	gateOut := mlx.SliceStartStop(projection, []int32{0, d}, []int32{n, 2 * d})
	content := mlx.SliceStartStop(projection, []int32{0, 2 * d}, []int32{n, 3 * d})
	merged := cfg.SpatialMergeSize
	image := mlx.Reshape(mlx.Mul(gateIn, content), height/merged, width/merged, merged, merged, d)
	image = mlx.Reshape(mlx.Transpose(image, 0, 2, 1, 3, 4), 1, height, width, d)
	padding := cfg.ConvKernelSize / 2
	image = mlx.Conv2d(image, c.Weight, 1, 1, padding, padding, 1, 1, d)
	image = mlx.Reshape(image, height/merged, merged, width/merged, merged, d)
	image = mlx.Reshape(mlx.Transpose(image, 0, 2, 1, 3, 4), n, d)
	return c.OutProj.Forward(mlx.Mul(gateOut, image))
}

func (a *visionAttention) Forward(x, cos, sin *mlx.Array, cfg *visionConfig) *mlx.Array {
	n := int32(x.Dim(0))
	width := cfg.HiddenSize / cfg.NumAttentionHeads
	q := a.QNorm.Forward(mlx.Reshape(a.QProj.Forward(x), n, cfg.NumAttentionHeads, width), cfg.NormEps)
	k := a.KNorm.Forward(mlx.Reshape(a.KProj.Forward(x), n, cfg.NumKeyValueHeads, width), cfg.NormEps)
	v := mlx.Reshape(a.VProj.Forward(x), n, cfg.NumKeyValueHeads, width)
	q = mlx.ExpandDims(mlx.Transpose(rotateVision(q, cos, sin), 1, 0, 2), 0)
	k = mlx.ExpandDims(mlx.Transpose(rotateVision(k, cos, sin), 1, 0, 2), 0)
	v = mlx.ExpandDims(mlx.Transpose(v, 1, 0, 2), 0)
	out := mlx.FastScaledDotProductAttention(q, k, v, float32(1/math.Sqrt(float64(width))), mlx.FastSDPANoWindow, "", nil)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), n, cfg.HiddenSize)
	return a.OutProj.Forward(out)
}

func (v *VisionModel) Project(x *mlx.Array) *mlx.Array {
	merged := v.ProjectorNorm.Forward(x)
	merged = mlx.Reshape(merged, int32(merged.Dim(0))/v.DownsampleFactor/v.DownsampleFactor,
		v.Config.HiddenSize*v.DownsampleFactor*v.DownsampleFactor)
	return v.ProjectorLinear2.Forward(mlx.GELU(v.ProjectorLinear1.Forward(merged)))
}

func (v *VisionModel) Forward(pixels *mlx.Array, height, width int32) *mlx.Array {
	h := v.Embed(pixels)
	cos, sin := v.rotaryAngles(v.PositionIDs(height, width))
	for _, layer := range v.Layers {
		h = layer.Forward(h, height, width, cos, sin, &v.Config)
	}
	return v.Project(h)
}
