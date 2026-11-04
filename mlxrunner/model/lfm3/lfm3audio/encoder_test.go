package lfm3audio

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func tinyConfig() Config {
	cfg := defaultConfig()
	cfg.NumMelBins = 16
	cfg.ConvChannels = 4
	cfg.HiddenSize = 16
	cfg.IntermediateSize = 24
	cfg.NumAttentionHeads = 2
	cfg.NumKeyValueHeads = 1
	cfg.HeadDim = 8
	cfg.ConvKernel = 3
	cfg.NormEps = 1e-5
	cfg.RopeTheta = 1e6
	cfg.SlidingWindow = 4
	cfg.LayerTypes = []string{"conv", "full_attention", "conv", "full_attention"}
	cfg.OutputSize = 12
	cfg.ProjectorNormEps = 1e-5
	return cfg
}

// tinyTensors builds random float32 weights under layout's names.
func tinyTensors(cfg Config, layout Layout, seed uint64) map[string]*mlx.Array {
	rng := rand.New(rand.NewPCG(seed, 1))
	tensors := map[string]*mlx.Array{}
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		v := make([]float32, n)
		scale := 1 / math.Sqrt(float64(shape[len(shape)-1]))
		if len(shape) > 1 {
			scale = 1 / math.Sqrt(float64(n/shape[0]))
		}
		for i := range v {
			v[i] = float32(rng.NormFloat64() * scale)
		}
		tensors[name] = mlx.FromValues(v, shape...)
	}
	ch, d, ff := int(cfg.ConvChannels), int(cfg.HiddenSize), int(cfg.IntermediateSize)
	q, kv := int(cfg.NumAttentionHeads*cfg.HeadDim), int(cfg.NumKeyValueHeads*cfg.HeadDim)
	ds := layout.Downsample
	add(ds+"conv_in.weight", ch, 1, 3, 3)
	add(ds+"conv_in.bias", ch)
	for _, name := range []string{"conv_mid", "conv_end"} {
		add(ds+name+".weight", ch, 1, 3, 3)
		add(ds+name+".bias", ch)
	}
	for _, name := range []string{"lin_mid", "lin_end"} {
		add(ds+name+".weight", ch, ch, 1, 1)
		add(ds+name+".bias", ch)
	}
	add(ds+"lin_out.weight", d, ch*cfg.NumMelBins/8, 1)
	add(ds+"lin_out.bias", d)
	for i, kind := range cfg.LayerTypes {
		p := fmt.Sprintf("%slayers.%d.", layout.Encoder, i)
		add(p+"operator_norm.weight", d)
		add(p+"ffn_norm.weight", d)
		add(p+"feed_forward.w1.weight", ff, d)
		add(p+"feed_forward.w3.weight", ff, d)
		add(p+"feed_forward.w2.weight", d, ff)
		if kind == "conv" {
			add(p+"conv.in_proj.weight", 3*d, d)
			add(p+"conv.out_proj.weight", d, d)
			add(p+"conv.conv.weight", d, 1, int(cfg.ConvKernel))
		} else {
			add(p+"self_attn.q_proj.weight", q, d)
			add(p+"self_attn.k_proj.weight", kv, d)
			add(p+"self_attn.v_proj.weight", kv, d)
			add(p+"self_attn.out_proj.weight", d, q)
			add(p+"self_attn.q_layernorm.weight", int(cfg.HeadDim))
			add(p+"self_attn.k_layernorm.weight", int(cfg.HeadDim))
		}
	}
	add(layout.Encoder+"embedding_norm.weight", d)
	add(layout.Projector[0]+".weight", d)
	add(layout.Projector[0]+".bias", d)
	add(layout.Projector[1]+".weight", int(cfg.OutputSize), d)
	add(layout.Projector[1]+".bias", int(cfg.OutputSize))
	add(layout.Projector[2]+".weight", int(cfg.OutputSize), int(cfg.OutputSize))
	add(layout.Projector[2]+".bias", int(cfg.OutputSize))
	return tensors
}

func loadTiny(t *mlxtest.T, cfg Config, tensors map[string]*mlx.Array) *Encoder {
	t.Helper()
	enc, err := LoadEncoder(tensors, cfg, model.NewLinearFactory(tensors, 0, 0, "", nil))
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

func tinyFeatures(cfg Config, frames int) *mlx.Array {
	rng := rand.New(rand.NewPCG(7, 7))
	v := make([]float32, cfg.NumMelBins*frames)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return mlx.FromValues(v, cfg.NumMelBins, frames)
}

func maxAbsDiff(a, b []float32) float64 {
	var m float64
	for i := range a {
		m = max(m, math.Abs(float64(a[i]-b[i])))
	}
	return m
}

func evalFloats(a *mlx.Array) []float32 {
	a = a.AsType(mlx.DTypeFloat32)
	mlx.Eval(a)
	return a.Floats()
}

func TestEncoderLayoutsAndShapes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			cfg := tinyConfig()
			omni := loadTiny(t, cfg, tinyTensors(cfg, OmniLayout, 1))
			audio := loadTiny(t, cfg, tinyTensors(cfg, AudioLayout, 1))
			features := tinyFeatures(cfg, 75)
			a, b := omni.Forward(features), audio.Forward(features)
			if got := a.Dims(); len(got) != 2 || got[0] != 10 || got[1] != 12 {
				t.Fatalf("output shape %v, want [10 12]", got)
			}
			if d := maxAbsDiff(evalFloats(a), evalFloats(b)); d != 0 {
				t.Fatalf("layouts disagree by %g", d)
			}
		})
	})
}

// TestEncoderBlocksMatchWhole checks that blocked downsampling and attention
// compute the same result as one pass over the clip.
func TestEncoderBlocksMatchWhole(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			cfg := tinyConfig()
			enc := loadTiny(t, cfg, tinyTensors(cfg, OmniLayout, 2))
			defer func(d, a int32) { downsampleBlockTokens, attentionBlockTokens = d, a }(downsampleBlockTokens, attentionBlockTokens)
			for _, frames := range []int{1, 9, 75, 203} {
				features := tinyFeatures(cfg, frames)
				downsampleBlockTokens, attentionBlockTokens = 1<<20, 1<<20
				whole := evalFloats(enc.Forward(features))
				for _, blocks := range [][2]int32{{1, 1}, {2, 3}, {5, 7}} {
					downsampleBlockTokens, attentionBlockTokens = blocks[0], blocks[1]
					got := evalFloats(enc.Forward(features))
					if len(got) != len(whole) {
						t.Fatalf("frames %d blocks %v: %d values, want %d", frames, blocks, len(got), len(whole))
					}
					if d := maxAbsDiff(got, whole); d > 1e-5 {
						t.Errorf("frames %d blocks %v: max diff %g", frames, blocks, d)
					}
				}
			}
		})
	})
}

func TestWindowMask(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			const window = 4
			m := evalFloats(windowMask(6, 10, 3, window, mlx.DTypeFloat32))
			for q := 6; q < 10; q++ {
				for k := 3; k < 10; k++ {
					open := k <= q && k > q-window
					if got := m[(q-6)*7+k-3]; (got == 0) != open {
						t.Errorf("q=%d k=%d: mask %v, open=%v", q, k, got, open)
					}
				}
			}
		})
	})
}

func TestLoadEncoderErrors(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			cfg := tinyConfig()
			if _, err := LoadEncoder(map[string]*mlx.Array{}, cfg, model.LinearFactory{}); err == nil || !strings.Contains(err.Error(), "not found") {
				t.Errorf("empty tensors: err = %v", err)
			}
			tensors := tinyTensors(cfg, OmniLayout, 3)
			delete(tensors, OmniLayout.Encoder+"layers.1.self_attn.k_layernorm.weight")
			if _, err := LoadEncoder(tensors, cfg, model.NewLinearFactory(tensors, 0, 0, "", nil)); err == nil || !strings.Contains(err.Error(), "k_layernorm") {
				t.Errorf("missing tensor: err = %v", err)
			}
			wide := cfg
			wide.IntermediateSize = 32
			tensors = tinyTensors(cfg, OmniLayout, 3)
			if _, err := LoadEncoder(tensors, wide, model.NewLinearFactory(tensors, 0, 0, "", nil)); err == nil || !strings.Contains(err.Error(), "outputs") {
				t.Errorf("wrong feed-forward width: err = %v", err)
			}
		})
	})
}
