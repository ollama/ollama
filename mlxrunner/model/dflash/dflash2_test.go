package dflash

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	"github.com/ollama/ollama/mlxrunner/sample"
)

func rounded(x float32, bf16 bool) float32 {
	if !bf16 {
		return x
	}
	bits := math.Float32bits(x)
	return math.Float32frombits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

// Independently indexed scalar convolution covers every batch, row, channel,
// and tap, including one-token blocks shorter than the trained two-tap kernel.
func TestConvolutionOracle(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		for _, length := range []int{1, 2, 5, 7, 8, 15, 16, 32} {
			mlxtest.RunSubtest(t, fmt.Sprintf("%v length %d", dtype, length), func(t *mlxtest.T) {
				mlx.EnableCompile()
				const b, width, group, taps = 2, 8, 2, 2
				groups := width / group
				round := func(x float32) float32 { return rounded(x, dtype == mlx.DTypeBFloat16) }
				values := func(n, modulus int) []float32 {
					x := make([]float32, n)
					for i := range x {
						x[i] = round(float32((i*7)%modulus-modulus/2) / 9)
					}
					return x
				}
				h := values(b*length*width, 23)
				d := values(b*length*taps*groups, 17)
				s := values(taps*width, 13)
				want := convolutionOracle(h, d, s, b, length, width, taps, groups, round)
				got := convolve(mlx.FromValues(h, b, length, width).AsType(dtype), mlx.FromValues(d, b, length, taps, groups).AsType(dtype), mlx.FromValues(s, taps, width).AsType(dtype)).AsType(mlx.DTypeFloat32)
				mlx.Eval(got)
				for i, v := range got.Floats() {
					if math.Abs(float64(v-want[i])) > 1e-6 {
						t.Fatalf("dtype=%v length=%d index=%d: got %g want %g", dtype, length, i, v, want[i])
					}
				}
			})
		}
	}
}

// convolutionOracle is convolve one scalar at a time, rounding every product
// and sum: h is [b, length, width], d [b, length, taps, groups], s [taps, width].
func convolutionOracle(h, d, s []float32, b, length, width, taps, groups int, round func(float32) float32) []float32 {
	group := width / groups
	want := make([]float32, len(h))
	for seq := range b {
		for row := range length {
			for channel := range width {
				idx := (seq*length+row)*width + channel
				for tap := range taps {
					if row < tap {
						continue
					}
					x := h[(seq*length+row-tap)*width+channel]
					want[idx] = round(want[idx] + round(s[tap*width+channel]*x))
					weight := d[((seq*length+row)*taps+tap)*groups+channel/group]
					want[idx] = round(want[idx] + round(weight*x))
				}
			}
		}
	}
	return want
}

// A DFlash2 layer convolves each sublayer's normed input with half 0 of the
// taps projected from it, and the sublayer's output with half 1. Zero queries
// make attention a mean, so scalar steps around the oracle compose the layer.
func TestDFlash2LayerConvolution(t *testing.T) {
	const width, group, taps = 8, 2, 2
	const groups = width / group
	values := func(n, salt, modulus int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*salt)%modulus-modulus/2) / float32(modulus)
		}
		return v
	}
	identity := make([]float32, width*width)
	for i := range width {
		identity[i*width+i] = 1
	}
	for _, length := range []int{1, 3, 8} {
		mlxtest.RunSubtest(t, fmt.Sprintf("length %d", length), func(t *mlxtest.T) {
			x := values(length*width, 7, 23)
			norms := [2][]float32{values(width, 3, 7), values(width, 5, 11)}
			for _, w := range norms {
				for i := range w {
					w[i]++
				}
			}
			bases := [2][]float32{values(2*taps*width, 3, 13), values(2*taps*width, 11, 19)}
			projections := [2][]float32{values(2*taps*groups*width, 5, 17), values(2*taps*groups*width, 13, 29)}
			conv := func(i int) *convolution {
				return &convolution{Base: mlx.FromValues(bases[i], 2, taps, width), Projection: nn.NewLinear(mlx.FromValues(projections[i], 2*taps*groups, width), nil), Group: group, Kernel: taps}
			}
			cfg, attention := meanAttention(width, 0)
			layer := &Layer{
				InputNorm:     nn.NewRMSNorm(mlx.FromValues(norms[0], width), cfg.RMSNormEps),
				PostAttnNorm:  nn.NewRMSNorm(mlx.FromValues(norms[1], width), cfg.RMSNormEps),
				Attention:     attention,
				MLP:           &MLP{GateUpProj: nn.NewLinear(mlx.FromValues(slices.Concat(identity, identity), 2*width, width), nil), DownProj: nn.NewLinear(mlx.FromValues(identity, width, width), nil)},
				AttentionConv: conv(0),
				MLPConv:       conv(1),
			}
			kv := cache.NewKVCache()
			defer kv.Free()
			bb := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, length), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(length)}}
			got := layer.Forward(mlx.FromValues(x, 1, length, width), nil, nil, kv, bb, mlx.FromValues([]int32{0}, 1), nn.AttentionMask{}, 1, int32(length), cfg)
			mlx.Eval(got)

			norm := func(h, w []float32) []float32 {
				out := make([]float32, len(h))
				for r := range length {
					sum := 0.0
					for c := range width {
						sum += float64(h[r*width+c]) * float64(h[r*width+c])
					}
					inv := 1 / math.Sqrt(sum/width+float64(cfg.RMSNormEps))
					for c := range width {
						out[r*width+c] = float32(float64(h[r*width+c]) * inv * float64(w[c]))
					}
				}
				return out
			}
			// half convolves h with one half of the taps convolution i projects from p.
			half := func(i, which int, p, h []float32) []float32 {
				d := make([]float32, length*taps*groups)
				for r := range length {
					for j := range taps * groups {
						sum := 0.0
						for c := range width {
							sum += float64(p[r*width+c]) * float64(projections[i][(which*taps*groups+j)*width+c])
						}
						d[r*taps*groups+j] = float32(sum)
					}
				}
				return convolutionOracle(h, d, bases[i][which*taps*width:(which+1)*taps*width], 1, length, width, taps, groups, func(x float32) float32 { return x })
			}
			add := func(a, b []float32) []float32 {
				out := make([]float32, len(a))
				for i := range a {
					out[i] = a[i] + b[i]
				}
				return out
			}
			inner := norm(x, norms[0])
			convolved := half(0, 0, inner, inner)
			mean := make([]float32, len(convolved))
			for c := range width {
				sum := 0.0
				for r := range length {
					sum += float64(convolved[r*width+c])
				}
				for r := range length {
					mean[r*width+c] = float32(sum / float64(length))
				}
			}
			x = add(x, half(0, 1, inner, mean))
			inner = norm(x, norms[1])
			mlp := half(1, 0, inner, inner)
			for i, v := range mlp {
				mlp[i] = float32(float64(v) * float64(v) / (1 + math.Exp(-float64(v))))
			}
			want := add(x, half(1, 1, inner, mlp))
			// 2% absorbs MLX's default TF32 float32 matmuls on M5-class GPUs:
			// ~9e-3 worst there, ~2e-6 with MLX_ENABLE_TF32=0.
			for i, v := range got.Floats() {
				if math.Abs(float64(v-want[i])) > 2e-2*math.Max(math.Abs(float64(want[i])), 1) {
					t.Fatalf("length %d index %d: got %g want %g", length, i, v, want[i])
				}
			}
		})
	}
}

// The edge term must use the last sampled candidate, not the original anchor.
func TestSelectorConditionalProbabilities(t *testing.T) {
	for _, temperature := range []float32{0, 1} {
		mlxtest.RunSubtest(t, fmt.Sprintf("temperature %g", temperature), func(t *mlxtest.T) {
			m := &model2{Selector: &candidateSelector{
				Projection:  nn.NewLinear(mlx.FromValues([]float32{1}, 1, 1), nil),
				Predecessor: mlx.FromValues([]float32{3, 3, -3}, 3, 1),
				Successor:   mlx.FromValues([]float32{100, 0, 1}, 3, 1), TopK: 2,
			}}
			sampler := sample.New(16)
			defer sampler.Free()
			sampler.Add(0, sample.Options{Temperature: temperature, UseSeed: true, Seed: 42}, nil)
			tokens, ids, probs := m.Select(mlx.FromValues([]float32{1, 1, 1}, 1, 3, 1), mlx.FromValues([]float32{0, 4, 3, 0, 4, 3, 0, 4, 3}, 1, 3, 3), mlx.FromValues([]int32{0}, 1), func(ids, scores *mlx.Array) (*mlx.Array, *mlx.Array) {
				tokens, d := sampler.SampleCandidates(0, ids, scores)
				return tokens, d.Probs
			})
			mlx.Eval(tokens, ids, probs)
			path, support, q := tokens.Ints(), ids.Ints(), probs.Floats()
			if temperature == 0 && !slices.Equal(path, []int32{2, 1, 2}) {
				t.Fatalf("greedy path %v, want [2 1 2]", path)
			}
			previous := int32(0)
			for row := range 3 {
				if path[row] != support[row*2] && path[row] != support[row*2+1] {
					t.Fatalf("selected token %d outside support %v", path[row], support[row*2:row*2+2])
				}
				scores := []float64{4, 3 + float64([]float32{3, 3, -3}[previous])}
				for j := range 2 {
					id := support[row*2+j]
					if id != 1 && id != 2 {
						t.Fatalf("candidate %d outside unary top-k", id)
					}
					want := float64(0)
					if temperature == 0 {
						if id == path[row] {
							want = 1
						}
					} else {
						want = math.Exp(scores[id-1]) / (math.Exp(scores[0]) + math.Exp(scores[1]))
					}
					if math.Abs(float64(q[row*2+j])-want) > 1e-6 {
						t.Fatalf("temp=%g row=%d token=%d probability=%g want=%g", temperature, row, id, q[row*2+j], want)
					}
				}
				previous = path[row]
			}
		})
	}
}

func TestDFlash2ConfigValidation(t *testing.T) {
	valid := Config{HiddenSize: 5120, ConvGroupSize: 16, ConvKernelSize: 2, BlockSize: 8, SelectorRank: 256, SelectorTopK: 16, VocabSize: 248320}
	for _, tt := range []struct {
		name   string
		change func(*Config)
		valid  bool
	}{
		{"valid", func(*Config) {}, true},
		{"zero group", func(c *Config) { c.ConvGroupSize = 0 }, false},
		{"fractional group", func(c *Config) { c.ConvGroupSize = 17 }, false},
		{"negative kernel", func(c *Config) { c.ConvKernelSize = -1 }, false},
		{"oversized kernel", func(c *Config) { c.ConvKernelSize = 9 }, false},
		{"oversized convolution projection", func(c *Config) { c.HiddenSize, c.ConvGroupSize = 1<<30, 1 }, false},
		{"empty selector", func(c *Config) { c.SelectorTopK = 0 }, false},
		{"selector above vocab", func(c *Config) { c.VocabSize, c.SelectorTopK = 32, 33 }, false},
		{"selector at vocab", func(c *Config) { c.VocabSize, c.SelectorTopK = 32, 32 }, true},
		{"selector at cap", func(c *Config) { c.SelectorTopK = maxSelectorTopK }, true},
		{"selector over cap", func(c *Config) { c.SelectorTopK = maxSelectorTopK + 1 }, false},
		{"negative rank", func(c *Config) { c.SelectorRank = -1 }, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := valid
			tt.change(&cfg)
			if err := cfg.validateDFlash2(); (err == nil) != tt.valid {
				t.Fatalf("validateDFlash2() = %v, valid=%v", err, tt.valid)
			}
		})
	}
}

func TestProjectionRejectsMalformedWeights(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		nvfp4, nvfp4Scales, _ := mlx.Quantize(mlx.Zeros(mlx.DTypeBFloat16, 8, 16), 16, 4, "nvfp4")
		for _, tt := range []struct {
			name           string
			weight, scales *mlx.Array
			valid          bool
		}{
			{"dense", mlx.Zeros(mlx.DTypeBFloat16, 8, 16), nil, true},
			{"nvfp4", nvfp4, nvfp4Scales, true},
			{"missing", nil, nil, false},
			{"rank", mlx.Zeros(mlx.DTypeBFloat16, 8), nil, false},
			{"input width", mlx.Zeros(mlx.DTypeBFloat16, 8, 15), nil, false},
			{"output width", mlx.Zeros(mlx.DTypeBFloat16, 7, 16), nil, false},
			{"scale rank", mlx.Zeros(mlx.DTypeUint32, 8, 2), mlx.Zeros(mlx.DTypeFloat32, 8), false},
			{"scale groups", mlx.Zeros(mlx.DTypeUint32, 8, 2), mlx.Zeros(mlx.DTypeFloat32, 8, 2), false},
		} {
			tensors := map[string]*mlx.Array{"p.weight": tt.weight, "p.weight_scale": tt.scales}
			factory := model.NewLinearFactory(tensors, 16, 4, "nvfp4", nil)
			if _, err := loadProjection(factory, tensors, "p", 16, 8); (err == nil) != tt.valid {
				t.Errorf("%s: loadProjection() error = %v, valid=%v", tt.name, err, tt.valid)
			}
		}
	})
}

// LoadWeights takes every DFlash2 convolution and selector shape from the
// config, so a tensor of any other shape must fail the load.
func TestDFlash2LoadWeights(t *testing.T) {
	const layer, selector = "draft.layers.0.", "draft.candidate_selector."
	w := func(dims ...int) *mlx.Array { return mlx.Zeros(mlx.DTypeBFloat16, dims...) }
	valid := func() map[string]*mlx.Array {
		tensors := map[string]*mlx.Array{
			"draft.fc.weight": w(8, 8), "draft.hidden_norm.weight": w(8), "draft.norm.weight": w(8),
			layer + "mlp.gate_proj.weight": w(16, 8), layer + "mlp.up_proj.weight": w(16, 8), layer + "mlp.down_proj.weight": w(8, 16),
			selector + "predecessor_codebook": w(16, 4), selector + "successor_codebook": w(16, 4), selector + "hidden_projection.weight": w(4, 8),
		}
		for _, name := range []string{"q_proj", "k_proj", "v_proj", "o_proj"} {
			tensors[layer+"self_attn."+name+".weight"] = w(8, 8)
		}
		for _, name := range []string{"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"} {
			tensors[layer+name+".weight"] = w(8)
		}
		for _, name := range []string{"attention_conv", "mlp_conv"} {
			tensors[layer+name+".base_kernel"] = w(2, 2, 8)
			tensors[layer+name+".kernel_projection.weight"] = w(16, 8)
		}
		return tensors
	}
	load := func(tensors map[string]*mlx.Array) (*model2, error) {
		m := &model2{Model: &Model{
			Config:       &Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1, HeadDim: 8, RMSNormEps: 1e-6, LayerTypes: []string{"full_attention"}, BlockSize: 4, VocabSize: 16, TargetLayerIDs: []int{0}, ConvGroupSize: 2, ConvKernelSize: 2, SelectorRank: 4, SelectorTopK: 2},
			tensorPrefix: "draft.",
			Layers:       make([]*Layer, 1),
		}}
		return m, m.LoadWeights(tensors)
	}
	mlxtest.RunSubtest(t, "valid", func(t *mlxtest.T) {
		tensors := valid()
		m, err := load(tensors)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range []*convolution{m.Layers[0].AttentionConv, m.Layers[0].MLPConv} {
			if c == nil || !slices.Equal(c.Base.Dims(), []int{2, 2, 8}) || !slices.Equal(c.Projection.(*nn.Linear).Weight.Dims(), []int{16, 8}) || c.Group != 2 || c.Kernel != 2 {
				t.Fatalf("convolution %+v", c)
			}
		}
		s := m.Selector
		if s == nil || !slices.Equal(s.Predecessor.Dims(), []int{16, 4}) || !slices.Equal(s.Successor.Dims(), []int{16, 4}) || !slices.Equal(s.Projection.(*nn.Linear).Weight.Dims(), []int{4, 8}) || s.TopK != 2 {
			t.Fatalf("selector %+v", s)
		}
		// The loader keeps only the arrays mlx.Collect reaches.
		collected := mlx.Collect(m)
		for _, name := range []string{
			selector + "predecessor_codebook", selector + "successor_codebook", selector + "hidden_projection.weight",
			layer + "attention_conv.base_kernel", layer + "mlp_conv.base_kernel",
			layer + "attention_conv.kernel_projection.weight", layer + "mlp_conv.kernel_projection.weight",
		} {
			if !slices.Contains(collected, tensors[name]) {
				t.Fatalf("mlx.Collect misses %s, so loading would release it", name)
			}
		}
	})
	for _, tt := range []struct {
		name, key string
		dims      []int // nil removes the tensor
	}{
		{"missing base kernel", layer + "attention_conv.base_kernel", nil},
		{"base kernel halves", layer + "attention_conv.base_kernel", []int{1, 2, 8}},
		{"base kernel taps", layer + "mlp_conv.base_kernel", []int{2, 3, 8}},
		{"base kernel width", layer + "mlp_conv.base_kernel", []int{2, 2, 7}},
		{"kernel projection rows", layer + "attention_conv.kernel_projection.weight", []int{15, 8}},
		{"kernel projection width", layer + "mlp_conv.kernel_projection.weight", []int{16, 7}},
		{"missing codebook", selector + "successor_codebook", nil},
		{"codebook vocab", selector + "predecessor_codebook", []int{15, 4}},
		{"codebook rank", selector + "successor_codebook", []int{16, 5}},
		{"selector projection rows", selector + "hidden_projection.weight", []int{5, 8}},
		{"selector projection width", selector + "hidden_projection.weight", []int{4, 9}},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			tensors := valid()
			if tt.dims == nil {
				delete(tensors, tt.key)
			} else {
				tensors[tt.key] = w(tt.dims...)
			}
			if _, err := load(tensors); err == nil {
				t.Fatalf("LoadWeights accepted %s with shape %v", tt.key, tt.dims)
			}
		})
	}
}

// Both paths must preserve the BF16 rounding between static and dynamic taps.
func TestConvolutionCompiledMatchesEager(t *testing.T) {
	for _, length := range []int{1, 7, 8, 15, 16, 32} {
		mlxtest.RunSubtest(t, fmt.Sprintf("length%d", length), func(t *mlxtest.T) {
			t.Cleanup(mlx.EnableCompile)
			hidden, dynamic, base := convolutionInputs(length)
			want := convolveGraph(hidden, dynamic, base).AsType(mlx.DTypeFloat32)
			mlx.Eval(want)
			for _, enabled := range []bool{true, false} {
				if enabled {
					mlx.EnableCompile()
				} else {
					mlx.DisableCompile()
				}
				got := convolve(hidden, dynamic, base).AsType(mlx.DTypeFloat32)
				mlx.Eval(got)
				if !slices.Equal(got.Floats(), want.Floats()) {
					t.Fatalf("length%d compile=%v: compiled convolution differs from eager", length, enabled)
				}
			}
		})
	}
}

func convolutionInputs(length int) (hidden, dynamic, base *mlx.Array) {
	const width, groups, taps = 5120, 320, 2
	values := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*17)%73-36) / 64
		}
		return v
	}
	return mlx.FromValues(values(length*width), 1, length, width).AsType(mlx.DTypeBFloat16),
		mlx.FromValues(values(length*taps*groups), 1, length, taps, groups).AsType(mlx.DTypeBFloat16),
		mlx.FromValues(values(taps*width), taps, width).AsType(mlx.DTypeBFloat16)
}

func BenchmarkConvolutionEager(b *testing.B)    { benchmarkConvolution(b, convolveGraph) }
func BenchmarkConvolutionCompiled(b *testing.B) { benchmarkConvolution(b, convolve) }

func benchmarkConvolution(b *testing.B, fn func(*mlx.Array, *mlx.Array, *mlx.Array) *mlx.Array) {
	worker := mlxtest.Worker(b)
	for _, length := range []int{8, 16} {
		b.Run(fmt.Sprintf("length%d", length), func(b *testing.B) {
			if err := worker.Do(b.Context(), func() error {
				mlx.EnableCompile()
				defer mlx.ClearCache()
				mlx.Scoped(func() {
					hidden, dynamic, base := convolutionInputs(length)
					mlx.Eval(hidden, dynamic, base)
					mlx.Scoped(func() { mlx.Eval(fn(hidden, dynamic, base)) })
					b.ResetTimer()
					for range b.N {
						mlx.Scoped(func() { mlx.Eval(fn(hidden, dynamic, base)) })
					}
					b.StopTimer()
				})
				return nil
			}); err != nil {
				b.Fatal(err)
			}
		})
	}
}

func selectorScoreInputs(dtype mlx.DType, batch, rank int) []*mlx.Array {
	const vocab, k, length = 128, 16, 3
	values := make([]float32, vocab*rank)
	for i := range values {
		values[i] = float32((i*17)%43-21) / 32
	}
	ids := make([]int32, batch*length*k)
	previous := make([]int32, batch*length*k)
	for i := range ids {
		ids[i] = int32((i*5 + 3) % vocab)
		previous[i] = int32(i * 7 % vocab)
	}
	pred := mlx.FromValues(values, vocab, rank).AsType(dtype)
	succ := pred.Multiply(mlx.FromValue(float32(0.5))).AsType(dtype)
	gates := mlx.FromValues(values[:batch*length*rank], batch, length, rank).AsType(dtype)
	unary := mlx.FromValues(values[:batch*length*k], batch, length, k).AsType(dtype)
	return []*mlx.Array{pred, succ, mlx.FromValues(previous, batch, length, k), mlx.FromValues(ids, batch, length, k), gates, unary}
}

// The batched table scores each (predecessor, candidate) pair exactly as one
// row's reference expression does.
func TestSelectorScoresMatchPerRow(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		mlxtest.RunSubtest(t, fmt.Sprintf("%v", dtype), func(t *mlxtest.T) {
			in := selectorScoreInputs(dtype, 2, 256)
			predecessor, successor, previous, ids, gates, unary := in[0], in[1], in[2], in[3], in[4], in[5]
			got := selectorScores(in...)[0]
			batch, length, k := ids.Dim(0), ids.Dim(1), ids.Dim(2)
			for pos := range length {
				at := mlx.Slice(pos, pos+1)
				candidates := ids.Slice(mlx.Slice(), at, mlx.Slice()).Reshape(batch, k)
				gate := gates.Slice(mlx.Slice(), at, mlx.Slice())
				row := unary.Slice(mlx.Slice(), at, mlx.Slice()).Reshape(batch, k)
				for p := range k {
					prev := previous.Slice(mlx.Slice(), at, mlx.Slice(p, p+1)).Reshape(batch)
					pred := predecessor.TakeAxis(prev, 0).ExpandDims(1)
					succ := successor.TakeAxis(candidates, 0)
					want := row.Add(pred.Multiply(gate).Multiply(succ).SumAxis(-1, false)).AsType(mlx.DTypeFloat32)
					// Floats reads raw memory, so make the strided slice contiguous first.
					have := mlx.Contiguous(got.Slice(mlx.Slice(), at, mlx.Slice(p, p+1), mlx.Slice()), false).Reshape(batch, k).AsType(mlx.DTypeFloat32)
					mlx.Eval(want, have)
					if !slices.Equal(have.Floats(), want.Floats()) {
						t.Fatalf("pos %d predecessor slot %d: got %v want %v", pos, p, have.Floats(), want.Floats())
					}
				}
			}
		})
	}
}

func TestSelectorScoresCompiledMatchesEager(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		for _, batch := range []int{1, 2} {
			mlxtest.RunSubtest(t, fmt.Sprintf("%v-batch%d", dtype, batch), func(t *mlxtest.T) {
				t.Cleanup(mlx.EnableCompile)
				inputs := selectorScoreInputs(dtype, batch, 256)
				want := selectorScoresGraph(inputs...)[0].AsType(mlx.DTypeFloat32)
				mlx.Eval(want)
				for _, enabled := range []bool{true, false} {
					if enabled {
						mlx.EnableCompile()
					} else {
						mlx.DisableCompile()
					}
					got := selectorScores(inputs...)[0].AsType(mlx.DTypeFloat32)
					mlx.Eval(got)
					expected := want.Floats()
					for i, v := range got.Floats() {
						if v != expected[i] {
							t.Fatalf("score[%d] dtype=%v compile=%v: got %g want %g", i, dtype, enabled, v, expected[i])
						}
					}
				}
			})
		}
	}
}

func BenchmarkSelectorScoresEager(b *testing.B)    { benchmarkSelectorScores(b, selectorScoresGraph) }
func BenchmarkSelectorScoresCompiled(b *testing.B) { benchmarkSelectorScores(b, selectorScores) }

func benchmarkSelectorScores(b *testing.B, fn mlx.CompileFunc) {
	if err := mlxtest.Worker(b).Do(b.Context(), func() error {
		mlx.EnableCompile()
		defer mlx.ClearCache()
		mlx.Scoped(func() {
			inputs := selectorScoreInputs(mlx.DTypeBFloat16, 1, 256)
			mlx.Eval(inputs...)
			mlx.Scoped(func() { mlx.Eval(fn(inputs...)...) })
			b.ResetTimer()
			for range b.N {
				mlx.Scoped(func() { mlx.Eval(fn(inputs...)...) })
			}
			b.StopTimer()
		})
		return nil
	}); err != nil {
		b.Fatal(err)
	}
}
