package phonon

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

// Layer is a pre-norm transformer layer, shared by the flow LM and the Mimi
// decoder. Scale1 and Scale2 are Mimi's layer scales.
type Layer struct {
	Norm1   *nn.LayerNorm
	Norm2   *nn.LayerNorm
	InProj  *nn.Linear
	OutProj *nn.Linear
	Linear1 *nn.Linear
	Linear2 *nn.Linear
	Scale1  *mlx.Array
	Scale2  *mlx.Array
}

type Transformer struct {
	Layers    []*Layer
	NumHeads  int32
	MaxPeriod float32
	GELU      string
}

func loadTransformer(tensors map[string]*mlx.Array, prefix string, layers, heads int32, period float32, gelu string) (*Transformer, error) {
	t := &Transformer{NumHeads: heads, MaxPeriod: period, GELU: gelu}
	for i := range layers {
		p := fmt.Sprintf("%s.layers.%d.", prefix, i)
		l := &Layer{
			Norm1:   layerNorm(tensors, p+"norm1", 1e-5),
			Norm2:   layerNorm(tensors, p+"norm2", 1e-5),
			InProj:  linear(tensors, p+"self_attn.in_proj"),
			OutProj: linear(tensors, p+"self_attn.out_proj"),
			Linear1: linear(tensors, p+"linear1"),
			Linear2: linear(tensors, p+"linear2"),
			// The runner loads "<name>.scale" tensors as "<name>_scale".
			Scale1: tensors[p+"layer_scale_1_scale"],
			Scale2: tensors[p+"layer_scale_2_scale"],
		}
		if l.Norm1 == nil || l.Norm2 == nil || l.InProj == nil || l.OutProj == nil || l.Linear1 == nil || l.Linear2 == nil {
			return nil, fmt.Errorf("missing weights for %s", p)
		}
		if heads <= 0 || l.Norm1.Weight.Dim(0)%int(heads) != 0 {
			return nil, fmt.Errorf("invalid attention heads for %s", p)
		}
		t.Layers = append(t.Layers, l)
	}
	return t, nil
}

// forward runs x [1, L, D] at absolute position offset.
func (t *Transformer) forward(x *mlx.Array, offset int, caches []cache.Attention) *mlx.Array {
	L, D := int32(x.Dim(1)), int32(x.Dim(2))
	// Attention uses the batch shape and positions for both text and audio frames.
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, int(L)),
		SeqOffsets:   []int32{int32(offset)},
		SeqQueryLens: []int32{L},
	}
	H, hd := t.NumHeads, D/t.NumHeads
	scale := float32(1 / math.Sqrt(float64(hd)))
	pos := mlx.FromValues([]int32{int32(offset)}, 1)
	split := func(qkv *mlx.Array, i int32) *mlx.Array {
		s := qkv.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(int(i*D), int((i+1)*D)))
		return mlx.Transpose(mlx.Reshape(s, 1, L, H, hd), 0, 2, 1, 3)
	}
	for i, l := range t.Layers {
		qkv := l.InProj.Forward(l.Norm1.Forward(x))
		q := mlx.RoPEWithBase(split(qkv, 0), int(hd), true, t.MaxPeriod, 1, pos)
		k := mlx.RoPEWithBase(split(qkv, 1), int(hd), true, t.MaxPeriod, 1, pos)
		history := caches[i].Update(b, k, split(qkv, 2))
		a := nn.ScaledDotProductAttention(b, q, scale, nn.WithKVHistory(history), nn.WithMask(nn.CausalMask()))
		a = mlx.Reshape(mlx.Transpose(a, 0, 2, 1, 3), 1, L, D)
		x = residual(x, l.OutProj.Forward(a), l.Scale1)
		h := l.Linear1.Forward(l.Norm2.Forward(x))
		if t.GELU == "erf" {
			h = mlx.GELU(h)
		} else {
			h = mlx.GELUApprox(h)
		}
		h = l.Linear2.Forward(h)
		x = residual(x, h, l.Scale2)
	}
	return x
}

func residual(x, update, scale *mlx.Array) *mlx.Array {
	if scale != nil {
		update = mlx.Mul(update, scale)
	}
	return mlx.Add(x, update)
}

// FlowHead is the AdaLN MLP that turns a transformer state into a latent with
// one Lagrangian self-distillation step from s=0 to t=1.
type FlowHead struct {
	CondEmbed *nn.Linear
	InputProj *nn.Linear
	Blocks    []*FlowBlock
	Final     *nn.Linear
	// Modulation stacks every block's adaLN projection and the final
	// layer's so one matmul produces all shifts, scales and gates.
	Modulation *nn.Linear
	// Time is the averaged embedding of the constant s=0, t=1 times.
	Time *mlx.Array
}

type FlowBlock struct {
	Norm *nn.LayerNorm
	MLP0 *nn.Linear
	MLP2 *nn.Linear
}

func loadFlowHead(tensors map[string]*mlx.Array, prefix string, depth int32, norm string) (*FlowHead, error) {
	f := &FlowHead{
		CondEmbed: linear(tensors, prefix+".cond_embed"),
		InputProj: linear(tensors, prefix+".input_proj"),
		Final:     linear(tensors, prefix+".final_layer.linear"),
	}
	var mw, mb []*mlx.Array
	for i := range depth {
		p := fmt.Sprintf("%s.res_blocks.%d.", prefix, i)
		b := &FlowBlock{Norm: layerNorm(tensors, p+"in_ln", 1e-6), MLP0: linear(tensors, p+"mlp.0"), MLP2: linear(tensors, p+"mlp.2")}
		ada := linear(tensors, p+"adaLN_modulation.1")
		if b.Norm == nil || b.MLP0 == nil || b.MLP2 == nil || ada == nil {
			return nil, fmt.Errorf("missing weights for %s", p)
		}
		f.Blocks = append(f.Blocks, b)
		mw, mb = append(mw, ada.Weight), append(mb, ada.Bias)
	}
	ada := linear(tensors, prefix+".final_layer.adaLN_modulation.1")
	if f.CondEmbed == nil || f.InputProj == nil || f.Final == nil || ada == nil {
		return nil, fmt.Errorf("missing weights for %s", prefix)
	}
	f.Modulation = nn.NewLinear(mlx.Concatenate(append(mw, ada.Weight), 0), mlx.Concatenate(append(mb, ada.Bias), 0))

	// LSD conditions on a start and a target time; with one step they are
	// always 0 and 1.
	t0, t1 := timeEmbed(tensors, prefix+".time_embed.0", 0, norm), timeEmbed(tensors, prefix+".time_embed.1", 1, norm)
	if t0 == nil || t1 == nil {
		return nil, fmt.Errorf("missing time embedding for %s", prefix)
	}
	f.Time = mlx.MulScalar(mlx.Add(t0, t1), 0.5)
	return f, nil
}

// timeEmbed evaluates a TimestepEmbedder at t with the checkpoint's norm.
func timeEmbed(tensors map[string]*mlx.Array, prefix string, t float32, norm string) *mlx.Array {
	freqs := tensors[prefix+".freqs"]
	l0, l2, alpha := linear(tensors, prefix+".mlp.0"), linear(tensors, prefix+".mlp.2"), tensors[prefix+".mlp.3.alpha"]
	if freqs == nil || l0 == nil || l2 == nil || alpha == nil {
		return nil
	}
	args := mlx.MulScalar(freqs, t)
	x := mlx.Concatenate([]*mlx.Array{mlx.Cos(args), mlx.Sin(args)}, 0)
	x = l2.Forward(mlx.SiLU(l0.Forward(x)))
	return timeNorm(x, alpha, norm)
}

func timeNorm(x, alpha *mlx.Array, norm string) *mlx.Array {
	if norm == "rms" {
		return mlx.RMSNormFn(x, alpha, 1e-5)
	}
	// Legacy checkpoints divide by the unbiased standard deviation, without centering x.
	n := float32(x.Dim(0))
	centered := mlx.Sub(x, mlx.Mean(x, -1, true))
	variance := mlx.DivScalar(mlx.Sum(mlx.Mul(centered, centered), -1, true), n-1)
	return mlx.Mul(mlx.Mul(x, alpha), mlx.RSqrt(mlx.AddScalar(variance, 1e-5)))
}

// sample returns the latent for state c [1, D] from noise [1, latent].
func (f *FlowHead) sample(c, noise *mlx.Array) *mlx.Array {
	y := mlx.Add(f.CondEmbed.Forward(c), f.Time)
	mod := f.Modulation.Forward(mlx.SiLU(y))
	dim := y.Dim(1)
	chunk := func(i int) *mlx.Array { return mod.Slice(mlx.Slice(), mlx.Slice(i*dim, (i+1)*dim)) }

	x := f.InputProj.Forward(noise)
	for i, b := range f.Blocks {
		h := modulate(b.Norm.Forward(x), chunk(3*i), chunk(3*i+1))
		h = b.MLP2.Forward(mlx.SiLU(b.MLP0.Forward(h)))
		x = mlx.Add(x, mlx.Mul(chunk(3*i+2), h))
	}
	n := 3 * len(f.Blocks)
	x = modulate(mlx.LayerNormFn(x, nil, nil, 1e-6), chunk(n), chunk(n+1))
	return mlx.Add(noise, f.Final.Forward(x))
}

func modulate(x, shift, scale *mlx.Array) *mlx.Array {
	return mlx.Add(mlx.Mul(x, mlx.AddScalar(scale, 1)), shift)
}

func linear(tensors map[string]*mlx.Array, prefix string) *nn.Linear {
	w := tensors[prefix+".weight"]
	if w == nil {
		return nil
	}
	return nn.NewLinear(w, tensors[prefix+".bias"])
}

func layerNorm(tensors map[string]*mlx.Array, prefix string, eps float32) *nn.LayerNorm {
	w := tensors[prefix+".weight"]
	if w == nil {
		return nil
	}
	return &nn.LayerNorm{Weight: w, Bias: tensors[prefix+".bias"], Eps: eps}
}
