package hy4_flash

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

type visionConfig struct {
	Width      int     `json:"hidden_size"`
	Heads      int     `json:"num_attention_heads"`
	Layers     int     `json:"num_hidden_layers"`
	Channels   int     `json:"num_channels"`
	Patch      int     `json:"patch_size"`
	Merge      int     `json:"spatial_merge_size"`
	TableImage int     `json:"max_image_size"`
	TextWidth  int     `json:"out_hidden_size"`
	Eps        float32 `json:"rms_norm_eps"`
	Activation string  `json:"hidden_act"`
	Interp     string  `json:"interpolate_mode"`
}

type processorConfig struct {
	Image struct {
		Mean  []float32 `json:"image_mean"`
		Std   []float32 `json:"image_std"`
		Scale float32   `json:"rescale_factor"`
		Size  struct {
			Min int32 `json:"shortest_edge"`
			Max int32 `json:"longest_edge"`
		} `json:"size"`
	} `json:"image_processor"`
}

func (v *visionConfig) validate(textWidth int) error {
	if v.Width <= 0 || v.Heads <= 0 || v.Width%v.Heads != 0 || v.Layers <= 0 ||
		v.Channels != 3 || v.Patch <= 0 || v.Merge <= 1 || v.TableImage < v.Patch ||
		v.TableImage%v.Patch != 0 || v.TextWidth != textWidth || v.Eps <= 0 ||
		v.Activation != "gelu" || v.Interp != "bilinear" {
		return fmt.Errorf("unsupported Hy4 Flash vision tower configuration")
	}
	return nil
}

type visionTower struct {
	Block                 []visionBlock
	Patch, Merge, Project visionConv
	Positions, Newline    *mlx.Array
	MergeNorm             *nn.RMSNorm
}

type visionConv struct {
	Weight, Bias *mlx.Array
	stride       int32
}

func (c visionConv) forward(x *mlx.Array) *mlx.Array {
	return mlx.Add(mlx.Conv2d(x, c.Weight, c.stride, c.stride, 0, 0, 1, 1, 1), c.Bias)
}

type visionBlock struct {
	Q, K, V, Out             nn.LinearLayer
	FC1, FC2                 nn.LinearLayer
	AttentionNorm, DenseNorm *nn.LayerNorm
}

func loadVision(w weights, v *visionConfig) *visionTower {
	layerNorm := func(prefix string) *nn.LayerNorm {
		return &nn.LayerNorm{Weight: w.tensor(prefix + ".weight"), Bias: w.tensor(prefix + ".bias"), Eps: v.Eps}
	}
	conv := func(prefix string, stride int) visionConv {
		weight := w.tensor(prefix + ".weight")
		if *w.err == nil {
			weight = weight.Transpose(0, 2, 3, 1)
		}
		return visionConv{Weight: weight, Bias: w.tensor(prefix + ".bias"), stride: int32(stride)}
	}
	tower := &visionTower{
		Block:     make([]visionBlock, v.Layers),
		Patch:     conv("vit.embeddings.patch_embedding", v.Patch),
		Merge:     conv("vit.perceive.proj.0", v.Merge),
		Project:   conv("vit.perceive.proj.2", 1),
		Positions: w.tensor("vit.embeddings.position_embedding.weight"),
		MergeNorm: nn.NewRMSNorm(w.tensor("vit.perceive.norm.weight"), v.Eps),
		Newline:   w.tensor("vit.perceive.image_newline"),
	}
	for n := range tower.Block {
		p := fmt.Sprintf("vit.layers.%d", n)
		a := p + ".self_attn."
		tower.Block[n] = visionBlock{
			Q: w.linear(a + "q_proj"), K: w.linear(a + "k_proj"), V: w.linear(a + "v_proj"), Out: w.linear(a + "o_proj"),
			FC1: w.linear(p + ".mlp.dense_h_to_4h"), FC2: w.linear(p + ".mlp.dense_4h_to_h"),
			AttentionNorm: layerNorm(p + ".input_layernorm"), DenseNorm: layerNorm(p + ".post_attention_layernorm"),
		}
	}
	return tower
}

func (m *Model) EncodeMedia(item *model.PreparedItem, input *mlx.Array) *mlx.Array {
	tower, cfg := m.Vision, m.cfg.Vision
	prep := item.Opaque.(imageGrid)
	n := prep.rows * prep.cols
	x := tower.Patch.forward(input.AsType(tower.Patch.Weight.DType())).Reshape(1, n, cfg.Width)
	x = mlx.Add(x, tower.interpolate(prep, cfg.TableImage/cfg.Patch, cfg.Width).AsType(x.DType()))
	headDim := cfg.Width / cfg.Heads
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for _, layer := range tower.Block {
		normed := layer.AttentionNorm.Forward(x)
		project := func(proj nn.LinearLayer) *mlx.Array {
			return proj.Forward(normed).Reshape(1, n, cfg.Heads, headDim).Transpose(0, 2, 1, 3)
		}
		out := mlx.FastScaledDotProductAttention(project(layer.Q), project(layer.K), project(layer.V), scale, mlx.FastSDPANoWindow, "", nil)
		x = mlx.Add(x, layer.Out.Forward(out.Transpose(0, 2, 1, 3).Reshape(1, n, cfg.Width)))
		x = mlx.Add(x, layer.FC2.Forward(mlx.GELU(layer.FC1.Forward(layer.DenseNorm.Forward(x)))))
	}
	x = tower.MergeNorm.Forward(x, 0).Reshape(1, prep.rows, prep.cols, cfg.Width)
	x = tower.Project.forward(mlx.GELU(tower.Merge.forward(x)))
	rows, cols := prep.rows/cfg.Merge, prep.cols/cfg.Merge
	newline := mlx.BroadcastTo(tower.Newline.AsType(x.DType()), 1, int32(rows), 1, int32(cfg.TextWidth))
	x = mlx.Concatenate([]*mlx.Array{x, newline}, 2)
	return x.Reshape(rows*(cols+1), cfg.TextWidth)
}

func (v *visionTower) interpolate(grid imageGrid, side, width int) *mlx.Array {
	n := grid.rows * grid.cols
	indices, weights := make([]int32, 4*n), make([]float32, 4*n)
	hScale, wScale := float64(side)/(float64(grid.rows)+0.1), float64(side)/(float64(grid.cols)+0.1)
	for r := range grid.rows {
		y := max(0, (float64(r)+0.5)*hScale-0.5)
		y0, y1 := int(y), min(int(y)+1, side-1)
		fy := float32(y - float64(y0))
		for c := range grid.cols {
			x := max(0, (float64(c)+0.5)*wScale-0.5)
			x0, x1 := int(x), min(int(x)+1, side-1)
			fx := float32(x - float64(x0))
			i := 4 * (r*grid.cols + c)
			// Row zero of the learned table belongs to an unused class token.
			indices[i], indices[i+1] = int32(1+y0*side+x0), int32(1+y0*side+x1)
			indices[i+2], indices[i+3] = int32(1+y1*side+x0), int32(1+y1*side+x1)
			weights[i], weights[i+1] = (1-fy)*(1-fx), (1-fy)*fx
			weights[i+2], weights[i+3] = fy*(1-fx), fy*fx
		}
	}
	corners := mlx.Take(v.Positions, mlx.FromValues(indices, n, 4), 0).AsType(mlx.DTypeFloat32)
	values := mlx.Mul(corners, mlx.FromValues(weights, n, 4, 1))
	return mlx.Sum(values, 1, false).Reshape(1, n, width)
}
