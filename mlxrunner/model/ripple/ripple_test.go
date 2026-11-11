package ripple

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
	modeltypes "github.com/ollama/ollama/types/model"
)

func TestMain(m *testing.M) {
	if err := os.Setenv("MLX_ENABLE_TF32", "0"); err != nil {
		panic(err)
	}
	os.Exit(m.Run())
}

func fixtureConfig(t *testing.T) map[string]any {
	t.Helper()
	return map[string]any{
		"architectures": []string{"RicursiveForCausalLM"},
		"vocab_size":    19, "hidden_size": 8, "intermediate_size": 12,
		"num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
		"layer_types":    []string{"sliding_attention", "full_attention", "sliding_attention"},
		"sliding_window": 3, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
		"use_valres": true, "use_gated_attention": true, "gate_full_precision": true,
		"ffn_clamp_threshold": 20.0, "embed_scale": 71.55417527999327,
		"max_position_embeddings": 2097152, "tie_word_embeddings": false,
	}
}

func testRoot(t *testing.T, cfg map[string]any) *model.Root {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	root := &model.Root{Manifest: &manifest.Manifest{}}
	for name, data := range map[string]string{
		"config.json":    string(data),
		"tokenizer.json": `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`,
	} {
		layer, err := manifest.NewLayer(strings.NewReader(data), manifest.MediaTypeImageJSON)
		if err != nil {
			t.Fatal(err)
		}
		layer.Name = name
		root.Manifest.Layers = append(root.Manifest.Layers, layer)
	}
	return root
}

type referenceTensor struct {
	Data  []float32 `json:"data"`
	Shape []int     `json:"shape"`
	DType string    `json:"dtype"`
}

func (r referenceTensor) array() *mlx.Array {
	var a *mlx.Array
	if len(r.Shape) == 0 {
		a = mlx.FromValue(r.Data[0])
	} else {
		a = mlx.FromValues(r.Data, r.Shape...)
	}
	if r.DType == "bfloat16" {
		a = a.AsType(mlx.DTypeBFloat16)
	}
	return a
}

type reference struct {
	Weights map[string]map[string]referenceTensor
}

func readReference(t *testing.T) reference {
	t.Helper()
	m := &Model{Config: &Config{
		HiddenSize: 8, IntermediateSize: 12, NumHiddenLayers: 3,
		NumAttentionHeads: 4, NumKeyValueHeads: 2, HeadDim: 4, VocabSize: 19,
	}}
	r := reference{Weights: make(map[string]map[string]referenceTensor)}
	for _, dtype := range []string{"float32", "bfloat16"} {
		weights := make(map[string]referenceTensor)
		for name, shape := range m.weightShapes() {
			size := 1
			for _, dim := range shape {
				size *= dim
			}
			values := make([]float32, size)
			hash := sha256.Sum256([]byte(name))
			phase := int(binary.BigEndian.Uint32(hash[:4]) % 101)
			if strings.HasSuffix(name, "value_res_lambda") {
				layer := strings.Split(name, ".")[2]
				values[0] = map[string]float32{"0": -1.25, "1": 0.75, "2": 1.5}[layer]
			} else {
				for i := range values {
					v := float32((i*37+phase)%101-50) / 50
					switch {
					case strings.Contains(name, "norm.weight"):
						scaled := v * 0.375
						v = 1 + scaled
					case strings.Contains(name, "embed_tokens"):
						v *= 0.045
					case strings.Contains(name, ".mlp.gate_proj") || strings.Contains(name, ".mlp.up_proj"):
						v *= 18
					default:
						v *= 0.3
					}
					values[i] = v
				}
			}
			if dtype == "bfloat16" {
				for i, v := range values {
					bits := math.Float32bits(v)
					values[i] = math.Float32frombits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
				}
			}
			weights[name] = referenceTensor{Data: values, Shape: shape, DType: dtype}
		}
		r.Weights[dtype] = weights
	}
	return r
}

func checkTensor(t *mlxtest.T, label string, got *mlx.Array, want referenceTensor, atol, rtol float64) {
	t.Helper()
	if !slices.Equal(got.Dims(), want.Shape) {
		t.Fatalf("%s shape=%v want %v", label, got.Dims(), want.Shape)
	}
	a := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false)
	mlx.Eval(a)
	values := a.Floats()
	for i, v := range values {
		d := math.Abs(float64(v - want.Data[i]))
		bound := atol + rtol*math.Abs(float64(want.Data[i]))
		if math.IsNaN(float64(v)) || d > bound {
			t.Fatalf("%s[%d]=%.9g want %.9g (abs error %.9g > %.9g)", label, i, v, want.Data[i], d, bound)
		}
	}
}

type activationLinear struct {
	value *mlx.Array
	width int32
}

func (l activationLinear) Forward(x *mlx.Array) *mlx.Array {
	if l.value != nil {
		return l.value
	}
	return x
}

func (l activationLinear) OutputDim() int32 { return l.width }

func clipFFN(gate, up *mlx.Array, threshold *float32) (*mlx.Array, *mlx.Array) {
	if threshold != nil && *threshold > 0 {
		gate = mlx.Clamp(gate, float32(math.Inf(-1)), *threshold).AsType(gate.DType())
		up = mlx.Clamp(up, -*threshold, *threshold).AsType(up.DType())
	}
	return gate, up
}

func eagerFFN(gate, up *mlx.Array, threshold *float32) *mlx.Array {
	gate, up = clipFFN(gate, up, threshold)
	f := gate.AsType(mlx.DTypeFloat32)
	silu := mlx.Mul(f, mlx.Sigmoid(f)).AsType(gate.DType())
	return mlx.Mul(silu, up)
}

func TestMLPActivationFusionMemory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		const width, rows = 1024, 256
		gateValues, upValues := make([]float32, rows*width), make([]float32, rows*width)
		for i := range gateValues {
			gateValues[i] = float32(i%513-256) / 8
			upValues[i] = float32((i*7)%513-256) / 8
		}
		gate := mlx.FromValues(gateValues, 1, rows, width)
		up := mlx.FromValues(upValues, 1, rows, width)
		mlx.Eval(gate, up)
		m := MLP{GateProj: activationLinear{gate, width}, UpProj: activationLinear{up, width}, DownProj: activationLinear{width: width}}
		threshold := float32(20)
		eager := func() *mlx.Array { return eagerFFN(gate, up, &threshold) }
		actual := func() *mlx.Array { return m.Forward(gate, &threshold) }
		for _, fn := range []func() *mlx.Array{eager, actual} {
			mlx.Scoped(func() { mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{fn()} }) })
		}
		measure := func(fn func() *mlx.Array) int {
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			baseline := mlx.ActiveMemory()
			mlx.ResetPeakMemory()
			var peak int
			mlx.Scoped(func() {
				mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{fn()} })
				mlx.DefaultStream().Synchronize()
				peak = mlx.PeakMemory() - baseline
			})
			return peak
		}
		want, got := measure(eager), measure(actual)
		t.Logf("activation temporary peak: eager=%d actual=%d", want, got)
		if got >= want {
			t.Fatalf("MLP activation peak %d did not improve on eager %d", got, want)
		}
	})
}

func TestMLPActivationPrecision(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		t.Run(fmt.Sprint(dtype), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.EnableCompile()
				zero, small, large := float32(0), float32(2), float32(20)
				for _, rows := range []int{1, 8, 3, 1} {
					mlx.Scoped(func() {
						const width = 256
						gateValues, upValues := make([]float32, rows*width), make([]float32, rows*width)
						for i := range gateValues {
							gateValues[i] = float32(i%1025-512) / 16
							upValues[i] = float32((i*29)%701-350) / 8
						}
						copy(gateValues, []float32{-40, -20.125, -20, -19.875, -2.015625, -2, -1.984375, -0.125, 0, 0.125, 1.984375, 2, 2.015625, 19.875, 20, 20.125, 40})
						gate := mlx.FromValues(gateValues, 1, rows, width).AsType(dtype)
						up := mlx.FromValues(upValues, 1, rows, width).AsType(dtype)
						m := MLP{GateProj: activationLinear{gate, width}, UpProj: activationLinear{up, width}, DownProj: activationLinear{width: width}}
						for _, threshold := range []*float32{nil, &zero, &small, &large, &small} {
							want := eagerFFN(gate, up, threshold)
							got := m.Forward(gate, threshold)
							label := fmt.Sprintf("rows=%d unclipped", rows)
							if threshold != nil {
								label = fmt.Sprintf("rows=%d threshold=%g", rows, *threshold)
							}
							checkExactArray(t, label, got, want)
						}
					})
				}
			})
		})
	}
}

func eagerValueBlend(seed, x, lambda *mlx.Array) *mlx.Array {
	a := mlx.Sigmoid(lambda.AsType(mlx.DTypeFloat32)).AsType(lambda.DType()).AsType(x.DType()).AsType(mlx.DTypeFloat32)
	start, end := seed.AsType(mlx.DTypeFloat32), x.AsType(mlx.DTypeFloat32)
	delta := mlx.Sub(end, start)
	low := mlx.Add(start, mlx.Mul(a, delta))
	high := mlx.Sub(end, mlx.Mul(mlx.Sub(mlx.FromValue(float32(1)), a), delta))
	return mlx.Where(a.Less(mlx.FromValue(float32(0.5))), low, high).AsType(x.DType())
}

func TestValueBlendFusionMemory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		const width, rows = 1024, 256
		seedValues, xValues := make([]float32, rows*width), make([]float32, rows*width)
		for i := range seedValues {
			seedValues[i] = float32(i%257-128) / 64
			xValues[i] = float32((i*7)%257-128) / 64
		}
		seed := mlx.FromValues(seedValues, 1, rows, width)
		x := mlx.FromValues(xValues, 1, rows, width)
		lambda := mlx.FromValues([]float32{-0.5}, 1)
		mlx.Eval(seed, x, lambda)
		eager := func() *mlx.Array { return eagerValueBlend(seed, x, lambda) }
		actual := func() *mlx.Array { return blendValues(seed, x, lambda) }
		for _, fn := range []func() *mlx.Array{eager, actual} {
			mlx.Scoped(func() { mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{fn()} }) })
		}
		measure := func(fn func() *mlx.Array) int {
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			baseline := mlx.ActiveMemory()
			mlx.ResetPeakMemory()
			var peak int
			mlx.Scoped(func() {
				mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{fn()} })
				mlx.DefaultStream().Synchronize()
				peak = mlx.PeakMemory() - baseline
			})
			return peak
		}
		want, got := measure(eager), measure(actual)
		t.Logf("value-blend temporary peak: eager=%d actual=%d", want, got)
		if got >= want {
			t.Fatalf("value-blend peak %d did not improve on eager %d", got, want)
		}
	})
}

func TestValueBlendFusionPrecision(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		for _, parameterType := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
			t.Run(fmt.Sprintf("%v/parameter_%v", dtype, parameterType), func(t *testing.T) {
				mlxtest.Run(t, func(t *mlxtest.T) {
					mlx.EnableCompile()
					var maxError float64
					for _, rows := range []int{1, 8, 3, 1} {
						mlx.Scoped(func() {
							const width = 256
							seedValues, xValues := make([]float32, rows*width), make([]float32, rows*width)
							for i := range seedValues {
								seedValues[i] = float32(i%257-128) / 64
								xValues[i] = float32((i*11)%257-128) / 64
							}
							seed := mlx.FromValues(seedValues, 1, rows, width).AsType(dtype)
							x := mlx.FromValues(xValues, 1, rows, width).AsType(dtype)
							for _, value := range []float32{-20, -4, -0.0078125, 0, 0.0078125, 4, 20} {
								lambda := mlx.FromValues([]float32{value}, 1).AsType(parameterType)
								want, got := eagerValueBlend(seed, x, lambda), blendValues(seed, x, lambda)
								if got.DType() != dtype || !slices.Equal(got.Dims(), want.Dims()) {
									t.Fatal("value blend changed dtype or shape")
								}
								wantValues := mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false).Floats()
								gotValues := mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false).Floats()
								atol := 1e-6
								if dtype == mlx.DTypeBFloat16 {
									atol = 0
								}
								for i, v := range gotValues {
									delta := math.Abs(float64(v - wantValues[i]))
									maxError = math.Max(maxError, delta)
									if math.IsNaN(delta) || delta > atol {
										t.Fatalf("rows=%d lambda=%g element=%d: error=%g exceeds %g", rows, value, i, delta, atol)
									}
								}
							}
						})
					}
					t.Logf("maximum absolute difference=%g", maxError)
				})
			})
		}
	}
}

// Pre-fusion attention path for the allocation comparison.
func eagerAttentionForGateTest(a *Attention, x, seed *mlx.Array, b *batch.Batch, c cache.Cache, cos, sin *mlx.Array, local bool, cfg *Config) *mlx.Array {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	q := mlx.Reshape(a.QProj.Forward(x), B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	k := mlx.Reshape(a.KProj.Forward(x), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	v := mlx.Reshape(a.VProj.Forward(blendValues(seed, x, a.ValueResLambda)), B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	q = mlx.Transpose(rmsNorm(q, a.QNorm, cfg.RMSNormEps), 0, 2, 1, 3)
	k = mlx.Transpose(rmsNorm(k, a.KNorm, cfg.RMSNormEps), 0, 2, 1, 3)
	v = mlx.Transpose(v, 0, 2, 1, 3)
	if local {
		q, k = applyRotary(q, cos, sin), applyRotary(k, cos, sin)
	}
	mask := nn.CausalMask()
	var kv nn.SDPAOption
	if c != nil {
		kv = nn.WithKVHistory(c.(cache.Attention).Update(b, k, v))
	} else {
		kv = nn.WithKV(k, v, b.SeqQueryLens)
		if local {
			mask = mask.Intersect(nn.SlidingWindowMask(b, int(L), cfg.SlidingWindow, k.DType()))
		}
	}
	out := nn.ScaledDotProductAttention(b, q, cfg.Scale, kv, nn.WithMask(mask))
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, L, cfg.NumAttentionHeads*cfg.HeadDim)
	return a.OProj.Forward(mlx.Mul(out, attentionGate(a.GProj.Forward(x))))
}

func TestAttentionGateFusionMemory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.ClearCache()
		const heads, kvHeads, headDim = 80, 8, 128
		const width, kvWidth = heads * headDim, kvHeads * headDim
		qValues, gateValues := make([]float32, width), make([]float32, width)
		vValues, normValues := make([]float32, kvWidth), make([]float32, headDim)
		for i := range qValues {
			qValues[i] = float32(i%17-8) / 32
			gateValues[i] = float32(i%65-32) / 4
		}
		for i := range vValues {
			vValues[i] = float32(i%13-6) / 8
		}
		for i := range normValues {
			normValues[i] = 1
		}
		q := mlx.FromValues(qValues, 1, 1, width)
		k := mlx.Zeros(mlx.DTypeFloat32, 1, 1, kvWidth)
		v := mlx.FromValues(vValues, 1, 1, kvWidth)
		gate := mlx.FromValues(gateValues, 1, 1, width)
		norm := mlx.FromValues(normValues, headDim)
		// Keep the unrelated value blend scalar in this allocation probe.
		x := mlx.FromValues([]float32{1}, 1, 1, 1)
		lambda := mlx.FromValues([]float32{0}, 1)
		mlx.Eval(q, k, v, gate, norm, x, lambda)
		a := Attention{
			QProj: activationLinear{q, width}, KProj: activationLinear{k, kvWidth}, VProj: activationLinear{v, kvWidth},
			GProj: activationLinear{gate, width}, OProj: activationLinear{width: width},
			QNorm: norm, KNorm: norm, ValueResLambda: lambda,
		}
		cfg := Config{NumAttentionHeads: heads, NumKeyValueHeads: kvHeads, HeadDim: headDim, RMSNormEps: 1e-6, Scale: 1 / float32(math.Sqrt(headDim))}
		b := inputBatch([]float32{1}, 0)
		eager := func() *mlx.Array { return eagerAttentionForGateTest(&a, x, x, b, nil, nil, nil, false, &cfg) }
		actual := func() *mlx.Array { return a.Forward(x, x, b, nil, nil, nil, false, &cfg) }
		mlx.EnableCompile()
		for _, fn := range []func() *mlx.Array{eager, actual} {
			mlx.Scoped(func() { mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{fn()} }) })
		}
		measure := func(fn func() *mlx.Array) (int, []float32) {
			mlx.DefaultStream().Synchronize()
			mlx.ClearCache()
			baseline := mlx.ActiveMemory()
			mlx.ResetPeakMemory()
			var peak int
			var values []float32
			mlx.Scoped(func() {
				out := mlx.ScopedEval(func() []*mlx.Array { return []*mlx.Array{fn()} })[0]
				mlx.DefaultStream().Synchronize()
				peak = mlx.PeakMemory() - baseline
				values = mlx.Contiguous(out, false).Floats()
			})
			return peak, values
		}
		eagerPeak, want := measure(eager)
		fusedPeak, got := measure(actual)
		if !slices.Equal(got, want) {
			t.Fatal("compilation changed attention output")
		}
		t.Logf("attention temporary peak: eager=%d compiled=%d gate=%d", eagerPeak, fusedPeak, gate.NumBytes())
		if eagerPeak-fusedPeak < gate.NumBytes() {
			t.Fatalf("attention did not eliminate a gate-sized temporary: eager=%d compiled=%d gate=%d", eagerPeak, fusedPeak, gate.NumBytes())
		}
	})
}

func TestAttentionGateFusionPrecision(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		t.Run(fmt.Sprint(dtype), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.EnableCompile()
				for _, rows := range []int{1, 8, 3, 1} {
					mlx.Scoped(func() {
						const width = 256
						outValues, gateValues := make([]float32, rows*width), make([]float32, rows*width)
						for i := range gateValues {
							outValues[i] = float32((i*11)%257-128) / 16
							gateValues[i] = float32(i%1025-512) / 32
						}
						copy(gateValues, []float32{-40, -20, -4, -0.0078125, 0, 0.0078125, 4, 20, 40})
						out := mlx.FromValues(outValues, 1, rows, width).AsType(dtype)
						gate := mlx.FromValues(gateValues, 1, rows, width).AsType(dtype)
						want := mlx.Mul(out, mlx.Sigmoid(gate.AsType(mlx.DTypeFloat32)).AsType(dtype))
						checkExactArray(t, fmt.Sprintf("rows=%d", rows), gatedAttention(out, gate), want)
					})
				}
			})
		})
	}
}

func fixtureWeights(r reference, dtype string) map[string]*mlx.Array {
	out := make(map[string]*mlx.Array)
	for name, v := range r.Weights[dtype] {
		out[name] = v.array()
	}
	return out
}

func TestLoadWeights(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	r := readReference(t)
	root := testRoot(t, fixtureConfig(t))
	mlxtest.Run(t, func(t *mlxtest.T) {
		loaded, err := model.New(root)
		if err != nil {
			t.Fatal(err)
		}
		m := loaded.(*Model)
		weights := fixtureWeights(r, "bfloat16")
		if err := m.LoadWeights(weights); err != nil {
			t.Fatal(err)
		}
		if len(m.Layers) != 3 || m.Tokenizer() == nil || m.MaxContextLength() != 2097152 {
			t.Fatal("incomplete model")
		}
		for name, w := range weights {
			delete(weights, name)
			if err := m.LoadWeights(weights); err == nil || !strings.Contains(err.Error(), name) {
				t.Fatalf("missing %s: %v", name, err)
			}
			weights[name] = mlx.Zeros(mlx.DTypeBFloat16, 1, 1)
			if err := m.LoadWeights(weights); err == nil {
				t.Fatalf("accepted bad shape for %s", name)
			}
			weights[name] = w
		}
		if err := m.LoadWeights(weights); err != nil {
			t.Fatal(err)
		}
		caches := m.NewCaches()
		defer func() {
			for _, c := range caches {
				c.Free()
			}
		}()
		if _, ok := caches[0].(*cache.RotatingKVCache); !ok {
			t.Fatal("local cache type")
		}
		if _, ok := caches[1].(*cache.KVCache); !ok {
			t.Fatal("global cache type")
		}
	})
}

type prequantProjection struct {
	name       string
	rows, cols int
	global     float32
}

var prequantProjections = []prequantProjection{
	{"gate_proj", 64, 32, 0.5},
	{"up_proj", 64, 32, 2},
	{"down_proj", 32, 64, 0.125},
}

func prequantWeights() map[string]*mlx.Array {
	shapes := map[string][]int{
		"model.embed_tokens.weight": {32, 32}, "lm_head.weight": {32, 32}, "model.norm.weight": {32},
		"model.layers.0.input_layernorm.weight": {32}, "model.layers.0.post_attention_layernorm.weight": {32},
		"model.layers.0.pre_feedforward_layernorm.weight": {32}, "model.layers.0.post_feedforward_layernorm.weight": {32},
		"model.layers.0.self_attn.q_proj.weight": {32, 32}, "model.layers.0.self_attn.k_proj.weight": {16, 32},
		"model.layers.0.self_attn.v_proj.weight": {16, 32}, "model.layers.0.self_attn.o_proj.weight": {32, 32},
		"model.layers.0.self_attn.g_proj.weight": {32, 32}, "model.layers.0.self_attn.q_norm.weight": {16},
		"model.layers.0.self_attn.k_norm.weight": {16}, "model.layers.0.self_attn.value_res_lambda": {1},
	}
	weights := make(map[string]*mlx.Array)
	for name, shape := range shapes {
		size := shape[0]
		if len(shape) == 2 {
			size *= shape[1]
		}
		values := make([]float32, size)
		for i := range values {
			if len(shape) == 1 {
				values[i] = 1
			} else if i/shape[1] == i%shape[1] {
				values[i] = 0.5
			}
		}
		weights[name] = mlx.FromValues(values, shape...).AsType(mlx.DTypeBFloat16)
	}
	for _, p := range prequantProjections {
		name := "model.layers.0.mlp." + p.name + ".weight"
		packed := make([]uint32, p.rows*p.cols/8)
		scales := make([]uint8, p.rows*p.cols/16)
		for row := range p.rows {
			for col := range p.cols {
				packed[row*p.cols/8+col/8] |= uint32((row*3+col)%16) << (4 * (col % 8))
			}
			for group := range p.cols / 16 {
				// E4M3 encodings for 0.5, 1, 2, and 4.
				scales[row*p.cols/16+group] = []uint8{0x30, 0x38, 0x40, 0x48}[(row+group)%4]
			}
		}
		weights[name] = mlx.FromValues(packed, p.rows, p.cols/8)
		weights[name+"_scale"] = mlx.FromValues(scales, p.rows, p.cols/16)
		weights[name+".global_scale"] = mlx.FromValue(p.global)
	}
	return weights
}

func prequantRoot(t *testing.T) *model.Root {
	t.Helper()
	root := testRoot(t, map[string]any{
		"architectures": []string{"RicursiveForCausalLM"}, "hidden_size": 32, "intermediate_size": 64,
		"num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
		"vocab_size": 32, "max_position_embeddings": 64, "layer_types": []string{"full_attention"},
		"use_valres": true, "ffn_clamp_threshold": 20,
	})
	path := filepath.Join(t.TempDir(), "weights.safetensors")
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			weights := prequantWeights()
			metadata := make(map[string]string)
			for _, p := range prequantProjections {
				name := "model.layers.0.mlp." + p.name + ".weight"
				weights[name+".scale"] = weights[name+"_scale"]
				delete(weights, name+"_scale")
				metadata[name+".quant_type"] = "nvfp4"
				metadata[name+".group_size"] = "16"
			}
			if err := mlx.SaveSafetensorsWithMetadata(path, weights, metadata); err != nil {
				t.Fatal(err)
			}
		})
	})
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	layer, err := manifest.NewLayer(f, manifest.MediaTypeImageTensor)
	if err != nil {
		t.Fatal(err)
	}
	root.Manifest.Layers = append(root.Manifest.Layers, layer)
	config, err := manifest.NewLayer(strings.NewReader(`{"model_format":"safetensors"}`), manifest.MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}
	const name = "ripple-prequant-test"
	if err := manifest.WriteManifestWithMetadata(modeltypes.ParseName(name), config, root.Manifest.Layers, manifest.RunnerMLX, manifest.FormatSafetensors); err != nil {
		t.Fatal(err)
	}
	root, err = model.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func loadPrequantWeights(t *mlxtest.T, root *model.Root) map[string]*mlx.Array {
	t.Helper()
	weights := make(map[string]*mlx.Array)
	for _, layer := range root.Manifest.TensorLayers() {
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		for name, weight := range mlx.Load(path) {
			if strings.HasSuffix(name, ".scale") {
				name = strings.TrimSuffix(name, ".scale") + "_scale"
			}
			weights[name] = weight
		}
	}
	mlx.Eval(slices.Collect(maps.Values(weights))...)
	return weights
}

func checkExactArray(t *mlxtest.T, name string, got, want *mlx.Array) {
	t.Helper()
	if got == nil || want == nil {
		if got != want {
			t.Fatalf("%s: mismatched nil arrays", name)
		}
		return
	}
	if got.DType() != want.DType() || !slices.Equal(got.Dims(), want.Dims()) {
		t.Fatalf("%s: got %v %v, want %v %v", name, got.DType(), got.Dims(), want.DType(), want.Dims())
	}
	// Compare in the original dtype, including every bit of packed uint32 words.
	for i, equal := range mlx.Contiguous(got.Equal(want).AsType(mlx.DTypeInt32), false).Ints() {
		if equal != 1 {
			t.Fatalf("%s: element %d differs", name, i)
		}
	}
}

func checkPrequantBasis(t *mlxtest.T, m *Model, withGlobal bool) {
	t.Helper()
	layers := []nn.LinearLayer{m.Layers[0].MLP.GateProj, m.Layers[0].MLP.UpProj, m.Layers[0].MLP.DownProj}
	// Literal E2M1 values, independent of MLX's dequantizer.
	fp4 := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	block := []float32{0.5, 1, 2, 4}
	for i, p := range prequantProjections {
		basis := make([]float32, p.cols*p.cols)
		want := make([]float32, p.cols*p.rows)
		global := float32(1)
		if withGlobal {
			global = p.global
		}
		for col := range p.cols {
			basis[col*p.cols+col] = 1
			for row := range p.rows {
				want[col*p.rows+row] = fp4[(row*3+col)%16] * block[(row+col/16)%4] * global
			}
		}
		x := mlx.FromValues(basis, p.cols, p.cols).AsType(mlx.DTypeBFloat16)
		checkTensor(t, p.name, layers[i].Forward(x), referenceTensor{Data: want, Shape: []int{p.cols, p.rows}}, 0, 0)
	}
}

func TestLoadWeightsPrequantized(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	root := prequantRoot(t)
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			loaded, err := model.New(root)
			if err != nil {
				t.Fatal(err)
			}
			m := loaded.(*Model)
			weights := loadPrequantWeights(t, root)
			if err := m.LoadWeights(weights); err != nil {
				t.Fatal(err)
			}
			layers := []nn.LinearLayer{m.Layers[0].MLP.GateProj, m.Layers[0].MLP.UpProj, m.Layers[0].MLP.DownProj}
			for i, p := range prequantProjections {
				name := "model.layers.0.mlp." + p.name + ".weight"
				if q := m.TensorQuant[name]; q == nil || q.QuantType != "NVFP4" || q.GroupSize != 16 {
					t.Fatalf("%s metadata = %+v", name, q)
				}
				q, ok := layers[i].(*nn.QuantizedLinear)
				if !ok {
					t.Fatalf("%s type = %T", name, layers[i])
				}
				if q.Mode != "nvfp4" || q.Bits != 4 || q.GroupSize != 16 || q.QBiases != nil || q.Bias != nil {
					t.Fatalf("%s lost quantization parameters", name)
				}
				if q.Weight != weights[name] || q.Scales != weights[name+"_scale"] {
					t.Fatalf("%s did not preserve packed weights and scales", name)
				}
				if q.GlobalScale == nil || q.GlobalScale.DType() != mlx.DTypeFloat32 {
					t.Fatalf("%s lost float32 global scale", name)
				}
			}
			a := m.Layers[0].Attention
			for name, linear := range map[string]nn.LinearLayer{
				"lm_head.weight":                         m.LMHead,
				"model.layers.0.self_attn.q_proj.weight": a.QProj, "model.layers.0.self_attn.k_proj.weight": a.KProj,
				"model.layers.0.self_attn.v_proj.weight": a.VProj, "model.layers.0.self_attn.o_proj.weight": a.OProj,
				"model.layers.0.self_attn.g_proj.weight": a.GProj,
			} {
				dense, ok := linear.(*nn.Linear)
				if !ok || dense.Weight != weights[name] || dense.Weight.DType() != mlx.DTypeBFloat16 {
					t.Fatalf("%s did not remain dense BF16", name)
				}
			}
			embedding, ok := m.EmbedTokens.(*nn.Embedding)
			if !ok || embedding.Weight != weights["model.embed_tokens.weight"] || embedding.Weight.DType() != mlx.DTypeBFloat16 {
				t.Fatal("embedding did not remain dense BF16")
			}
			checkPrequantBasis(t, m, true)
		})
	})
}

func TestLoadWeightsPrequantizedScaleFormats(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	root := prequantRoot(t)
	for _, suffix := range []string{"_scale_2", ""} {
		name := "legacy_global"
		if suffix == "" {
			name = "block_only"
		}
		t.Run(name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					loaded, err := model.New(root)
					if err != nil {
						t.Fatal(err)
					}
					m := loaded.(*Model)
					weights := loadPrequantWeights(t, root)
					for _, p := range prequantProjections {
						name := "model.layers.0.mlp." + p.name + ".weight"
						if suffix != "" {
							weights[name+suffix] = weights[name+".global_scale"]
						}
						delete(weights, name+".global_scale")
					}
					if err := m.LoadWeights(weights); err != nil {
						t.Fatal(err)
					}
					checkPrequantBasis(t, m, suffix != "")
				})
			})
		})
	}
}

func TestLoadWeightsPrequantizedInvalid(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	root := prequantRoot(t)
	const name = "model.layers.0.mlp.gate_proj.weight"
	for _, tc := range []struct {
		name, want string
		mutate     func(*Model, map[string]*mlx.Array)
	}{
		{"quant_type", "unsupported quantization", func(m *Model, _ map[string]*mlx.Array) {
			m.TensorQuant[name].QuantType = "unknown"
		}},
		{"group_size", "invalid quantization group/bits", func(m *Model, _ map[string]*mlx.Array) {
			m.TensorQuant[name].GroupSize = 32
		}},
		{"missing_scale", "missing scale for quantized weight", func(_ *Model, w map[string]*mlx.Array) {
			delete(w, name+"_scale")
			w[name] = mlx.Zeros(mlx.DTypeBFloat16, 64, 32)
		}},
		{"packed_without_scale", "invalid weight", func(_ *Model, w map[string]*mlx.Array) {
			delete(w, name+"_scale")
		}},
		{"packed_dtype", "invalid quantized shape", func(_ *Model, w map[string]*mlx.Array) {
			w[name] = mlx.Zeros(mlx.DTypeUint8, 64, 4)
		}},
		{"packed_rows", "invalid quantized shape", func(_ *Model, w map[string]*mlx.Array) {
			w[name] = mlx.Zeros(mlx.DTypeUint32, 32, 4)
		}},
		{"packed_columns", "invalid quantized shape", func(_ *Model, w map[string]*mlx.Array) {
			w[name] = mlx.Zeros(mlx.DTypeUint32, 64, 8)
		}},
		{"scale_dtype", "invalid packed scale dtype", func(_ *Model, w map[string]*mlx.Array) {
			w[name+"_scale"] = mlx.Zeros(mlx.DTypeFloat32, 64, 2)
		}},
		{"scale_rows", "invalid quantized shape", func(_ *Model, w map[string]*mlx.Array) {
			w[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 32, 2)
		}},
		{"scale_columns", "invalid quantized shape", func(_ *Model, w map[string]*mlx.Array) {
			w[name+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 64, 1)
		}},
		{"global_dtype", "invalid " + name + ".global_scale", func(_ *Model, w map[string]*mlx.Array) {
			w[name+".global_scale"] = mlx.FromValues([]uint32{1}, 1)
		}},
		{"global_shape", "invalid " + name + ".global_scale", func(_ *Model, w map[string]*mlx.Array) {
			w[name+".global_scale"] = mlx.FromValues([]float32{1, 2}, 2)
		}},
		{"activation_shape", "invalid " + name + ".input_scale", func(_ *Model, w map[string]*mlx.Array) {
			w[name+".input_scale"] = mlx.FromValues([]float32{1, 2}, 2)
		}},
		{"unexpected_bias", "unexpected weight: " + name + "_qbias", func(_ *Model, w map[string]*mlx.Array) {
			w[name+"_qbias"] = mlx.Zeros(mlx.DTypeFloat32, 64, 2)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					loaded, err := model.New(root)
					if err != nil {
						t.Fatal(err)
					}
					m := loaded.(*Model)
					weights := loadPrequantWeights(t, root)
					tc.mutate(m, weights)
					err = m.LoadWeights(weights)
					if err == nil || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), name) {
						t.Fatalf("LoadWeights error = %v, want %q for %s", err, tc.want, name)
					}
				})
			})
		})
	}
}

func TestWeightLifetime(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	root := prequantRoot(t)
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Warm compiled constants before measuring weight ownership.
		mlx.Scoped(func() {
			w := mlx.Zeros(mlx.DTypeUint32, 32, 4)
			s := mlx.Zeros(mlx.DTypeUint8, 32, 2)
			g := mlx.FromValues([]float32{mlx.Nvfp4MaxProduct}, 1)
			for _, shape := range [][]int{{1, 32}, {1, 1, 32}} {
				x := mlx.Zeros(mlx.DTypeBFloat16, shape...)
				mlx.Eval(mlx.QuantizedMatmul(x, w, s, nil, true, 16, 4, "nvfp4", g))
			}
			gate := mlx.Zeros(mlx.DTypeBFloat16, 1, 1, 32)
			up := mlx.AddScalar(gate, float32(1))
			mlx.Eval(clippedFFNActivation(gate, up, mlx.FromValue(float32(20))))
			lambda := mlx.FromValues([]float32{0.25}, 1).AsType(mlx.DTypeBFloat16)
			mlx.Eval(blendValues(gate, up, lambda))
		})
		mlx.DefaultStream().Synchronize()
		baseline := mlx.ActiveMemory()
		for cycle := range 2 {
			held := mlx.NewScope()
			defer held.Close()
			var m *Model
			var loadMemory, temporaryBytes int
			retained := mlx.ScopedArrays(func() []*mlx.Array {
				loaded, err := model.New(root)
				if err != nil {
					t.Fatal(err)
				}
				m = loaded.(*Model)
				weights := loadPrequantWeights(t, root)
				if err := m.LoadWeights(weights); err != nil {
					t.Fatal(err)
				}
				arrays := mlx.Collect(m)
				// Check membership while every source handle is still live.
				for name, w := range weights {
					if strings.HasSuffix(name, ".global_scale") {
						if slices.Contains(arrays, w) {
							t.Fatalf("retained load-only source %s", name)
						}
					} else if !slices.Contains(arrays, w) {
						t.Fatalf("model did not retain %s", name)
					}
				}
				for i, linear := range []nn.LinearLayer{m.Layers[0].MLP.GateProj, m.Layers[0].MLP.UpProj, m.Layers[0].MLP.DownProj} {
					q, ok := linear.(*nn.QuantizedLinear)
					if !ok || q.GlobalScale == nil || !slices.Contains(arrays, q.GlobalScale) {
						t.Fatalf("model did not retain converted global scale %d", i)
					}
				}
				temporary := mlx.Zeros(mlx.DTypeFloat32, 256, 256)
				mlx.Eval(temporary)
				temporaryBytes = temporary.NumBytes()
				mlx.DefaultStream().Synchronize()
				loadMemory = mlx.ActiveMemory()
				return arrays
			})
			mlx.DefaultStream().Synchronize()
			if got := mlx.ActiveMemory(); got > loadMemory-temporaryBytes {
				t.Fatalf("load temporary survived scope: active=%d, load=%d, temporary=%d", got, loadMemory, temporaryBytes)
			}
			mlx.Eval(retained...)
			held.Attach(retained...)
			var previous []float32
			for request := range 2 {
				mlx.Scoped(func() {
					checkPrequantBasis(t, m, true)
					hidden, _ := m.Forward(inputBatch([]float32{1, 4, 7}, 0), nil)
					logits := mlx.Contiguous(m.Unembed(hidden).AsType(mlx.DTypeFloat32), false)
					mlx.Eval(logits)
					if !slices.Equal(logits.Dims(), []int{1, 3, 32}) {
						t.Fatalf("logits shape = %v", logits.Dims())
					}
					values := logits.Floats()
					nonzero := false
					for _, v := range values {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Fatal("non-finite logits after load scope ended")
						}
						nonzero = nonzero || v != 0
					}
					if !nonzero {
						t.Fatal("zero logits after load scope ended")
					}
					if request > 0 && !slices.Equal(values, previous) {
						t.Fatal("weights changed between request scopes")
					}
					previous = slices.Clone(values)
				})
			}
			held.Close()
			mlx.DefaultStream().Synchronize()
			if got := mlx.ActiveMemory(); got != baseline {
				t.Fatalf("cycle %d leaked weight memory: active=%d, baseline=%d", cycle, got, baseline)
			}
		}
	})
}

func TestDraftWeightNamespace(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	r := readReference(t)
	for _, tc := range []struct {
		name, prefix, extra, missing, wantError string
		declared                                bool
	}{
		{name: "default", declared: true, extra: "draft.fc.weight"},
		{name: "explicit", declared: true, prefix: "draft.", extra: "draft.fc.weight"},
		{name: "custom", declared: true, prefix: "assistant.", extra: "assistant.fc.weight"},
		{name: "undeclared", extra: "draft.fc.weight", wantError: "unexpected weight: draft.fc.weight"},
		{name: "outside prefix", declared: true, prefix: "assistant.", extra: "draft.fc.weight", wantError: "unexpected weight: draft.fc.weight"},
		{name: "unknown target", declared: true, extra: "model.unknown.weight", wantError: "unexpected weight: model.unknown.weight"},
		{name: "missing target", declared: true, extra: "draft.fc.weight", missing: "lm_head.weight", wantError: "missing weight: lm_head.weight"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := testRoot(t, fixtureConfig(t))
			if tc.declared {
				root.Draft = &modeltypes.Draft{TensorPrefix: tc.prefix}
			}
			m, err := model.New(root)
			if err != nil {
				t.Fatal(err)
			}
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					tensors := fixtureWeights(r, "bfloat16")
					tensors[tc.extra] = mlx.FromValues([]float32{2}, 1)
					delete(tensors, tc.missing)
					before := maps.Clone(tensors)
					err := m.LoadWeights(tensors)
					if tc.wantError != "" {
						if err == nil || !strings.Contains(err.Error(), tc.wantError) {
							t.Fatalf("LoadWeights error=%v, want %q", err, tc.wantError)
						}
					} else if err != nil {
						t.Fatalf("declared draft namespace rejected: %v", err)
					}
					if !maps.Equal(before, tensors) {
						t.Fatal("LoadWeights modified the caller's tensor map")
					}
				})
			})
		})
	}
}

func TestDFlashTargetCombinedCaches(t *testing.T) {
	m, r := dflashTargetFixture(t)
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			if err := m.LoadWeights(fixtureWeights(r, "bfloat16")); err != nil {
				t.Fatal(err)
			}
			plain, combined := m.NewCaches(), m.NewCaches()
			draft := []cache.Cache{cache.NewKVCache(), cache.NewRotatingKVCache(3)}
			combined = append(combined, draft...)
			defer func() {
				for _, c := range append(plain, combined...) {
					c.Free()
				}
			}()
			b := inputBatch([]float32{1, 4, 7}, 0)
			want, _ := m.Forward(b, cache.Layers(plain))
			mlx.Eval(want)
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("target rejected trailing draft caches: %v", p)
					}
				}()
				got, _ := m.Forward(b, cache.Layers(combined))
				checkExactArray(t, "combined-cache target hidden", got, want)
			}()
			for _, c := range combined[:len(m.Layers)] {
				if c.Offset() != 3 {
					t.Fatalf("target cache offset=%d, want 3", c.Offset())
				}
			}
			for _, c := range draft {
				if c.Offset() != 0 {
					t.Fatalf("target forward modified draft cache offset: %d", c.Offset())
				}
			}
			func() {
				defer func() {
					if p := recover(); p == nil || !strings.Contains(fmt.Sprint(p), "cache count") {
						t.Fatalf("short cache slice panic=%v, want cache count rejection", p)
					}
				}()
				m.Forward(b, cache.Layers(plain[:len(plain)-1]))
			}()
		})
	})
}

func dflashTargetFixture(t *testing.T) (*Model, reference) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	loaded, err := model.New(testRoot(t, fixtureConfig(t)))
	if err != nil {
		t.Fatal(err)
	}
	return loaded.(*Model), readReference(t)
}

func TestDFlashTargetNumLayers(t *testing.T) {
	m, _ := dflashTargetFixture(t)
	target, ok := any(m).(interface{ NumLayers() int })
	if !ok {
		t.Fatal("Ripple target is missing NumLayers")
	}
	if got := target.NumLayers(); got != 3 {
		t.Fatalf("NumLayers() = %d, want 3 fixture blocks", got)
	}
}

func TestDFlashTargetDraftWindow(t *testing.T) {
	m, _ := dflashTargetFixture(t)
	target, ok := any(m).(interface{ DraftWindow() int })
	if !ok {
		t.Fatal("Ripple target is missing DraftWindow")
	}
	if got := target.DraftWindow(); got != 16384 {
		t.Fatalf("DraftWindow() = %d, want 16384", got)
	}
}

func TestDFlashTargetTokenEmbeddings(t *testing.T) {
	m, r := dflashTargetFixture(t)
	target, ok := any(m).(interface {
		TokenEmbeddings(*mlx.Array) *mlx.Array
	})
	if !ok {
		t.Fatal("Ripple target is missing TokenEmbeddings")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			if err := m.LoadWeights(fixtureWeights(r, "bfloat16")); err != nil {
				t.Fatal(err)
			}
			if m.EmbedScale == nil || *m.EmbedScale == 1 {
				t.Fatal("fixture must distinguish raw and scaled embeddings")
			}
			ids := []float32{1, 4, 7}
			weight := r.Weights["bfloat16"]["model.embed_tokens.weight"]
			width := weight.Shape[1]
			var want []float32
			for _, id := range ids {
				start := int(id) * width
				want = append(want, weight.Data[start:start+width]...)
			}
			got := target.TokenEmbeddings(inputBatch(ids, 0).InputIDs)
			if got.DType() != mlx.DTypeBFloat16 {
				t.Fatalf("embedding dtype = %v, want bfloat16", got.DType())
			}
			checkTensor(t, "raw table rows", got, referenceTensor{Data: want, Shape: []int{1, len(ids), width}}, 0, 0)
		})
	})
}

func TestDFlashTargetRawLogits(t *testing.T) {
	m, r := dflashTargetFixture(t)
	target, ok := any(m).(interface {
		RawLogits(*mlx.Array) *mlx.Array
	})
	if !ok {
		t.Fatal("Ripple target is missing RawLogits")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			if err := m.LoadWeights(fixtureWeights(r, "bfloat16")); err != nil {
				t.Fatal(err)
			}
			weight := r.Weights["bfloat16"]["lm_head.weight"]
			vocab, width := weight.Shape[0], weight.Shape[1]
			basis := make([]float32, width*width)
			want := make([]float32, width*vocab)
			for col := range width {
				basis[col*width+col] = 2
				for row := range vocab {
					want[col*vocab+row] = 2 * weight.Data[row*width+col]
				}
			}
			x := mlx.FromValues(basis, 1, width, width).AsType(mlx.DTypeBFloat16)
			got := target.RawLogits(x)
			if got.DType() != mlx.DTypeBFloat16 {
				t.Fatalf("logit dtype = %v, want bfloat16", got.DType())
			}
			checkTensor(t, "unnormalized head columns", got, referenceTensor{Data: want, Shape: []int{1, width, vocab}}, 0, 0)
		})
	})
}

// Run the existing blocks directly, independently of Forward's tap selection.
func dflashLayerOutputs(m *Model, b *batch.Batch) []*mlx.Array {
	h := scaleEmbedding(m.EmbedTokens.Forward(b.InputIDs), m.EmbedScale)
	seed := h
	positions := mlx.Add(mlx.Reshape(mlx.FromValues(b.SeqOffsets, len(b.SeqOffsets)), int32(len(b.SeqOffsets)), 1), mlx.Arange(0, float64(b.InputIDs.Dim(1)), 1, mlx.DTypeInt32))
	cos, sin := rotaryTables(positions, m.HeadDim, m.RopeTheta, h.DType())
	states := make([]*mlx.Array, len(m.Layers))
	for i, layer := range m.Layers {
		h = layer.Forward(h, seed, b, nil, cos, sin, m.LayerTypes[i] == "sliding_attention", m.Config)
		states[i] = h
	}
	return states
}

func TestDFlashTargetAuxHiddenLayers(t *testing.T) {
	m, r := dflashTargetFixture(t)
	target, ok := any(m).(interface{ SetAuxHiddenLayers([]int) })
	if !ok {
		t.Fatal("Ripple target is missing SetAuxHiddenLayers")
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			if err := m.LoadWeights(fixtureWeights(r, "bfloat16")); err != nil {
				t.Fatal(err)
			}
			b := inputBatch([]float32{1, 4, 7}, 0)
			states := dflashLayerOutputs(m, b)
			wantHidden := rmsNorm(states[len(states)-1], m.Norm, m.RMSNormEps)
			var observed [][]float32
			for i, state := range append(slices.Clone(states), wantHidden) {
				values := mlx.Contiguous(state.AsType(mlx.DTypeFloat32), false).Floats()
				for j, previous := range observed {
					if slices.Equal(values, previous) {
						t.Fatalf("fixture states %d and %d are indistinguishable", i, j)
					}
				}
				observed = append(observed, values)
			}

			baseline, aux := m.Forward(b, nil)
			if baseline != aux {
				t.Fatal("without taps, Forward must return h, h")
			}
			checkExactArray(t, "no-taps hidden", baseline, wantHidden)
			baselineLogits := m.Unembed(baseline)
			mlx.Eval(baselineLogits)

			for _, tc := range []struct {
				name string
				ids  []int
			}{
				{"first_last", []int{0, 2}},
				{"middle", []int{1}},
				{"disabled", nil},
			} {
				target.SetAuxHiddenLayers(tc.ids)
				hidden, aux := m.Forward(b, nil)
				checkExactArray(t, tc.name+" hidden", hidden, wantHidden)
				checkExactArray(t, tc.name+" logits", m.Unembed(hidden), baselineLogits)
				if len(tc.ids) == 0 {
					if hidden != aux {
						t.Fatal("clearing taps must restore h, h")
					}
					continue
				}
				if aux == nil || !slices.Equal(aux.Dims(), []int{1, 3, len(tc.ids) * int(m.HiddenSize)}) {
					t.Fatalf("%s: incorrect auxiliary feature shape", tc.name)
				}
				for slot, id := range tc.ids {
					start := slot * int(m.HiddenSize)
					got := aux.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, start+int(m.HiddenSize)))
					checkExactArray(t, fmt.Sprintf("%s tap %d", tc.name, id), got, states[id])
				}
			}
		})
	})
}

func inputBatch(ids []float32, offset int32) *batch.Batch {
	tokens := make([]int32, len(ids))
	for i, v := range ids {
		tokens[i] = int32(v)
	}
	return &batch.Batch{InputIDs: mlx.FromValues(tokens, 1, len(tokens)), SeqOffsets: []int32{offset}, SeqQueryLens: []int32{int32(len(tokens))}}
}

func TestAttentionWindow(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		cfg := &Config{HiddenSize: 4, HeadDim: 4, NumAttentionHeads: 1, NumKeyValueHeads: 1, RMSNormEps: 1e-6, Scale: 0.5, SlidingWindow: 1024}
		identity := mlx.FromValues([]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 4, 4)
		zero := nn.NewLinear(mlx.Zeros(mlx.DTypeFloat32, 4, 4), nil)
		a := &Attention{QProj: zero, KProj: zero, GProj: zero, VProj: nn.NewLinear(identity, nil), OProj: nn.NewLinear(identity, nil), QNorm: mlx.FromValues([]float32{1, 1, 1, 1}, 4), KNorm: mlx.FromValues([]float32{1, 1, 1, 1}, 4), ValueResLambda: mlx.FromValues([]float32{100}, 1)}
		for _, local := range []bool{false, true} {
			for _, length := range []int{1023, 1024, 1025} {
				vals := make([]float32, length*4)
				for i := range length {
					for j := range 4 {
						vals[i*4+j] = float32(i + 1)
					}
				}
				x := mlx.FromValues(vals, 1, length, 4)
				b := inputBatch(make([]float32, length), 0)
				positions := mlx.Reshape(mlx.Arange(0, float64(length), 1, mlx.DTypeFloat32), 1, int32(length))
				cos, sin := rotaryTables(positions, 4, 10000, mlx.DTypeFloat32)
				expected := make([]float32, length*4)
				for i := range length {
					lo := 0
					if local {
						lo = max(0, i-1023)
					}
					for j := range 4 {
						expected[i*4+j] = float32(lo+i+2) / 4
					}
				}
				want := referenceTensor{Data: expected, Shape: []int{1, length, 4}}
				for _, cached := range []bool{false, true} {
					var c cache.Cache
					if cached {
						if local {
							c = cache.NewRotatingKVCache(1024)
						} else {
							c = cache.NewKVCache()
						}
					}
					got := a.Forward(x, x, b, c, cos, sin, local, cfg)
					checkTensor(t, fmt.Sprintf("local=%v cached=%v length=%d", local, cached, length), got, want, 2e-3, 1e-5)
					if c != nil {
						c.Free()
					}
				}
			}
		}
	})
}

func TestCacheContinuation(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	r := readReference(t)
	cfg := fixtureConfig(t)
	cfg["sliding_window"] = 1024
	root := testRoot(t, cfg)
	mlxtest.Run(t, func(t *mlxtest.T) {
		loaded, err := model.New(root)
		if err != nil {
			t.Fatal(err)
		}
		m := loaded.(*Model)
		if err := m.LoadWeights(fixtureWeights(r, "float32")); err != nil {
			t.Fatal(err)
		}
		tokens := make([]float32, 1030)
		for i := range tokens {
			tokens[i] = float32((i*7 + 2) % 19)
		}
		all, _ := m.Forward(inputBatch(tokens, 0), nil)
		all = mlx.Contiguous(all, false)
		mlx.Eval(all)
		want := all.Floats()
		caches := m.NewCaches()
		defer func() {
			for _, c := range caches {
				c.Free()
			}
		}()
		run := func(start, end int) {
			h, _ := m.Forward(inputBatch(tokens[start:end], int32(start)), cache.Layers(caches))
			checkTensor(t, fmt.Sprintf("continuation %d:%d", start, end), h, referenceTensor{Data: want[start*8 : end*8], Shape: []int{1, end - start, 8}}, 3e-5, 3e-5)
		}
		run(0, 1023)
		run(1023, 1024)
		snaps := make([]cache.Snapshot, len(caches))
		for i, c := range caches {
			snaps[i] = c.Snapshot(0)
			defer snaps[i].Close()
		}
		run(1024, 1025)
		run(1025, 1030)
		for i, c := range caches {
			if !c.Restore(snaps[i], 1024) {
				t.Fatalf("restore layer %d", i)
			}
		}
		run(1024, 1027)
		run(1027, 1030)
		// A fresh request must not inherit the prior embedding seed.
		fresh, _ := m.Forward(inputBatch(tokens[:6], 0), nil)
		checkTensor(t, "fresh request", fresh, referenceTensor{Data: want[:48], Shape: []int{1, 6, 8}}, 3e-5, 3e-5)
	})
}

func TestConfig(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	cfg := fixtureConfig(t)
	cfg["is_hybrid_swa"] = false
	cfg["hybrid_layer_pattern"] = []int{0, 0, 0}
	got, err := loadConfig(testRoot(t, cfg))
	if err != nil {
		t.Fatal(err)
	}
	if got.HeadDim != 4 || got.HiddenSize != 8 || got.LayerTypes[1] != "full_attention" || got.SlidingWindow != 3 {
		t.Fatalf("unexpected config: %+v", got)
	}
	for _, tc := range []struct {
		field string
		value any
	}{
		{"hidden_size", 0},
		{"num_hidden_layers", 0},
		{"intermediate_size", -1},
		{"num_attention_heads", 3},
		{"num_key_value_heads", 0},
		{"head_dim", nil},
		{"head_dim", 3},
		{"vocab_size", 0},
		{"max_position_embeddings", 0},
		{"rms_norm_eps", -1},
		{"rope_theta", 0},
		{"sliding_window", 0},
		{"hidden_act", "gelu"},
		{"layer_types", []string{"full_attention"}},
		{"layer_types", []string{"sliding_attention", "bad", "full_attention"}},
		{"use_valres", false},
		{"use_gated_attention", false},
		{"gate_full_precision", false},
		{"tie_word_embeddings", true},
		{"attention_bias", true},
		{"mlp_bias", true},
		{"rope_scaling", map[string]any{"type": "linear", "factor": 2}},
		{"ffn_clamp_threshold", -1},
	} {
		t.Run(tc.field, func(t *testing.T) {
			bad := fixtureConfig(t)
			bad[tc.field] = tc.value
			_, err := loadConfig(testRoot(t, bad))
			if err == nil || !strings.Contains(err.Error(), tc.field) {
				t.Fatalf("want %s rejection, got %v", tc.field, err)
			}
		})
	}
}
