package kolibri1

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

const tinyConfig = `{"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"num_experts":3,"num_experts_per_tok":2,"moe_intermediate_size":32,"shared_expert_intermediate_size":32,"vocab_size":17,"max_position_embeddings":128,"layer_types":["sliding_attention","full_attention"],"sliding_window":3}`

func TestConfig(t *testing.T) {
	cfg, err := parseConfig([]byte(tinyConfig))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HeadDim != 8 || cfg.RopeTheta != 10000 {
		t.Fatal(cfg)
	}
	for _, change := range []string{`"num_experts_per_tok":4`, `"num_key_value_heads":3`, `"layer_types":[]`, `"sliding_window":0`, `"hidden_act":"relu"`} {
		var fields map[string]json.RawMessage
		json.Unmarshal([]byte(tinyConfig), &fields)
		var patch map[string]json.RawMessage
		json.Unmarshal([]byte("{"+change+"}"), &patch)
		for k, v := range patch {
			fields[k] = v
		}
		b, _ := json.Marshal(fields)
		if _, err := parseConfig(b); err == nil {
			t.Errorf("accepted %s", change)
		}
	}
}

func TestUnembedHeadDType(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		for _, headDType := range []string{"", "float32"} {
			mlxtest.RunSubtest(t, fmt.Sprintf("quantized=%t/head_dtype=%s", quantized, headDType), func(t *mlxtest.T) {
				cfg, err := parseConfig([]byte(strings.Replace(tinyConfig, "{", fmt.Sprintf(`{"head_dtype":%q,`, headDType), 1)))
				if err != nil {
					t.Fatal(err)
				}
				weights := make([]float32, 32*32)
				for i := range weights {
					weights[i] = 1
				}
				w := mlx.FromValues(weights, 32, 32).AsType(mlx.DTypeBFloat16)
				m := &Model{Config: &cfg, LMHead: nn.NewLinear(w, nil)}
				if quantized {
					// The NVFP4 import policy uses MXFP8 for the output head.
					m.LMHead = nn.NewQuantizedLinear(w, nil, 32, 8, "mxfp8")
				}
				values := make([]float32, 32)
				values[0], values[1] = 1, 1.0/512
				for _, length := range []int32{1, 64} {
					x := mlx.BroadcastTo(mlx.FromValues(values, 1, 1, 32).AsType(mlx.DTypeBFloat16), 1, length, 32)
					logits := m.Unembed(x)
					wantDType, want := mlx.DTypeBFloat16, float32(1)
					if headDType == "float32" {
						// BF16 projection followed by an FP32 cast loses this term.
						wantDType, want = mlx.DTypeFloat32, 1+1.0/512
					}
					if logits.DType() != wantDType {
						t.Fatalf("length %d: dtype %v, want %v", length, logits.DType(), wantDType)
					}
					logits = logits.AsType(mlx.DTypeFloat32)
					mlx.Eval(logits)
					for i, got := range logits.Floats() {
						if got != want {
							t.Fatalf("length %d: logit[%d] = %g, want %g", length, i, got, want)
						}
					}
				}
			})
		}
	}
}

func tinyValues(seed, n int, norm bool) []float32 {
	out := make([]float32, n)
	for i := range out {
		v := math.Sin(float64(i)*0.17+float64(seed)) * 0.1
		if norm {
			v++
		}
		out[i] = float32(v)
	}
	return out
}

func tinyModel(t *mlxtest.T, quantized bool) *Model {
	t.Helper()
	cfg, err := parseConfig([]byte(tinyConfig))
	if err != nil {
		t.Fatal(err)
	}
	weights := map[string]*mlx.Array{}
	put := func(name string, seed int, norm bool, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		weights[name] = mlx.FromValues(tinyValues(seed, n, norm), shape...)
	}
	put("model.embed_tokens.weight", 1, false, 17, 32)
	put("model.norm.weight", 2, true, 32)
	put("lm_head.weight", 3, false, 17, 32)
	cfg.TensorQuant = map[string]*model.TensorQuantInfo{}
	for l := range 2 {
		p := fmt.Sprintf("model.layers.%d", l)
		s := 10 + l*30
		for name, seed := range map[string]int{"input_layernorm": s, "post_attn_norm": s + 7, "post_attention_layernorm": s + 8, "post_ffn_norm": s + 10} {
			put(p+"."+name+".weight", seed, true, 32)
		}
		put(p+".self_attn.q_proj.weight", s+1, false, 16, 32)
		put(p+".self_attn.k_proj.weight", s+2, false, 8, 32)
		put(p+".self_attn.v_proj.weight", s+3, false, 8, 32)
		put(p+".self_attn.o_proj.weight", s+4, false, 32, 16)
		put(p+".self_attn.q_norm.weight", s+5, true, 8)
		put(p+".self_attn.k_norm.weight", s+6, true, 8)
		put(p+".mlp.gate.weight", s+9, false, 3, 32)
		weights[p+".moe.router.expert_bias"] = mlx.FromValues([]float32{0.4, -0.3, 0.2}, 3)
		for j, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			put(p+".mlp.shared_experts."+proj+".weight", s+20+j, false, 32, 32)
			var vals []float32
			for e := range 3 {
				vals = append(vals, tinyValues(s+11+3*e+j, 32*32, false)...)
			}
			name := p + ".mlp.experts." + proj + ".weight"
			w := mlx.FromValues(vals, 3, 32, 32)
			if quantized {
				globalScale := mlx.FromValue(float32(j+1) * 0.1)
				wq, sc, bias := mlx.QuantizeWithGlobalScale(w, 16, 4, "nvfp4", globalScale)
				weights[name+".global_scale"] = mlx.DivScalar(globalScale, mlx.Nvfp4MaxProduct)
				weights[name] = wq
				weights[name+"_scale"] = sc
				if bias != nil {
					weights[name+"_qbias"] = bias
				}
				cfg.QuantGroupSize = 16
				cfg.QuantBits = 4
				cfg.QuantMode = "nvfp4"
			} else {
				weights[name] = w
			}
		}
	}
	m := &Model{Config: &cfg, Layers: make([]*Layer, 2)}
	if err := m.LoadWeights(weights); err != nil {
		t.Fatal(err)
	}
	return m
}

func assertClose(t *mlxtest.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > tol {
			t.Fatalf("[%d] got %g want %g (tol %g)", i, v, want[i], tol)
		}
	}
}

func TestReferenceAndCache(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := tinyModel(t, false)
		tokens := []int32{1, 4, 2, 7, 3}
		b := &batch.Batch{InputIDs: mlx.FromValues(tokens, 1, 5), SeqOffsets: []int32{0}, SeqQueryLens: []int32{5}}
		h, _ := m.Forward(b, nil)
		out := m.Unembed(h)
		mlx.Eval(out)
		// Final-token logits from an independent scalar implementation.
		reference := []float32{
			-0.009084007, -0.265772628, -0.344440112, -0.192393017, 0.088523720,
			0.310145062, 0.324023504, 0.120862919, -0.163254663, -0.338020200,
			-0.286371374, -0.042903858, 0.229301732, 0.347915302, 0.233486523,
			-0.037337354, -0.283151740,
		}
		want := out.Floats()
		if len(want) != len(tokens)*len(reference) {
			t.Fatalf("unexpected logits length: %d", len(want))
		}
		assertClose(t, want[len(want)-len(reference):], reference, 2e-4)
		for _, chunks := range [][]int{{1, 1, 1, 1, 1}, {2, 2, 1}, {4, 1}} {
			caches := m.NewCaches()
			if _, ok := caches[0].(*cache.RotatingKVCache); !ok {
				t.Fatal("missing rotating cache")
			}
			var got []float32
			offset := 0
			for _, n := range chunks {
				b := &batch.Batch{InputIDs: mlx.FromValues(tokens[offset:offset+n], 1, n), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(n)}}
				h, _ := m.Forward(b, caches)
				out := m.Unembed(h)
				mlx.Eval(out)
				got = append(got, out.Floats()...)
				offset += n
			}
			assertClose(t, got, want, 2e-4)
		}
	})
}

func TestRouterBiasSelectsWithoutReweighting(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := &MoE{Router: mlx.FromValues([]float32{2, 1, -1}, 3, 1), ExpertBias: mlx.FromValues([]float32{-5, 0, 10}, 3)}
		cfg := &Config{NumExpertsPerTok: 2}
		ids, scores := m.route(mlx.FromValues([]float32{1}, 1, 1, 1), cfg)
		mlx.Eval(ids, scores)
		indices := ids.AsType(mlx.DTypeInt32).Ints()
		values := scores.Floats()
		for j, id := range indices {
			if id == 0 {
				t.Fatal("routing ignored bias")
			}
			logit := float64(1)
			if id == 2 {
				logit = -1
			}
			if math.Abs(float64(values[j])-1/(1+math.Exp(-logit))) > 1e-6 {
				t.Fatal("bias changed mixture weight")
			}
		}
	})
}

func TestNVFP4ExpertsFusedAndSorted(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		m := tinyModel(t, true).Layers[0].MLP
		cfg := &Config{HiddenSize: 32, MoEIntermediateSize: 32, NumExperts: 3, NumExpertsPerTok: 2}
		if m.GateScale == nil || m.UpScale == nil || m.GateUp == nil || m.GateUp.Mode != "nvfp4" || m.Down.Mode != "nvfp4" {
			t.Fatal("experts lost quantization or fusion")
		}
		x := mlx.FromValues(tinyValues(7, 64*32, false), 1, 64, 32)
		sorted := m.Forward(x, cfg)
		mlx.Eval(sorted)
		var want []float32
		for i := range 64 {
			y := m.Forward(x.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()), cfg)
			mlx.Eval(y)
			want = append(want, y.Floats()...)
		}
		assertClose(t, sorted.Floats(), want, 2e-5)
		// Different projection scales must survive fusion, including when
		// decoding with BF16 activations and no sorting.
		unfused := *m
		unfused.GateUp = nil
		split := func(start int, scale *mlx.Array) *ExpertLinear {
			e := *m.GateUp
			e.Weight = e.Weight.Slice(mlx.Slice(), mlx.Slice(start, start+32), mlx.Slice())
			e.Scales = e.Scales.Slice(mlx.Slice(), mlx.Slice(start, start+32), mlx.Slice())
			e.GlobalScale = model.PrepareGatherQMMGlobalScale(scale, 3)
			return &e
		}
		unfused.Gate, unfused.Up = split(0, m.GateScale), split(32, m.UpScale)
		for _, length := range []int{1, 64} {
			input := x.Slice(mlx.Slice(), mlx.Slice(0, length), mlx.Slice()).AsType(mlx.DTypeBFloat16)
			fused := m.Forward(input, cfg).AsType(mlx.DTypeFloat32)
			separate := unfused.Forward(input, cfg).AsType(mlx.DTypeFloat32)
			mlx.Eval(fused, separate)
			var diff2, norm2, maxDiff float64
			fusedValues, separateValues := fused.Floats(), separate.Floats()
			for i, v := range fusedValues {
				diff := float64(v - separateValues[i])
				diff2 += diff * diff
				norm2 += float64(v) * float64(v)
				maxDiff = max(maxDiff, math.Abs(diff))
			}
			t.Logf("BF16 fusion length=%d: relative L2=%g max absolute=%g", length, math.Sqrt(diff2/norm2), maxDiff)
			// The separate gather kernel scales before the projection's BF16
			// rounding; fusion scales the rounded projection inside SwiGLU.
			if math.Sqrt(diff2/norm2) > 0.03 {
				t.Fatal("excessive BF16 fusion error")
			}
			assertClose(t, fusedValues, separateValues, 5e-5)
		}
		// The quantized bank must also agree with explicitly dequantized weights.
		for _, e := range []*ExpertLinear{m.GateUp, m.Down} {
			e.Weight = mlx.Dequantize(e.Weight, e.Scales, e.Biases, e.GroupSize, e.Bits, e.Mode, e.GlobalScale)
			e.Scales = nil
		}
		dense := m.Forward(x, cfg)
		mlx.Eval(dense)
		assertClose(t, sorted.Floats(), dense.Floats(), 2e-5)
	})
}
