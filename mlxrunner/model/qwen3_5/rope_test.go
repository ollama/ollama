package qwen3_5

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestRopeReference(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	data, err := os.ReadFile("../../testdata/qwen35_yarn.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		SampleTokens   []int   `json:"sample_tokens"`
		SampleDims     []int   `json:"sample_dims"`
		MropePositions []int32 `json:"mrope_positions"`
		Cases          []struct {
			Name           string          `json:"name"`
			Dtype          string          `json:"dtype"`
			Modes          []string        `json:"modes"`
			Config         json.RawMessage `json:"config"`
			InvFreq        []float32       `json:"inv_freq"`
			TextOutput     []float32       `json:"text_output"`
			LongTextOutput []float32       `json:"long_text_output"`
			TextKeys       []float32       `json:"text_keys"`
			MropeOutput    []float32       `json:"mrope_output"`
			MropeKeys      []float32       `json:"mrope_keys"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	for _, schema := range []struct {
		name, field string
		nested      bool
		legacy      bool
	}{
		{name: "rope_parameters", field: "rope_parameters", nested: true},
		{name: "rope_scaling", field: "rope_scaling"},
		{name: "legacy_type", field: "rope_scaling", nested: true, legacy: true},
	} {
		for _, tc := range reference.Cases {
			modes := tc.Modes
			if len(modes) == 0 {
				modes = []string{"text", "mrope"}
			}
			for _, mode := range modes {
				t.Run(schema.name+"/"+tc.Name+"/"+mode, func(t *testing.T) {
					var config map[string]any
					if err := json.Unmarshal(tc.Config, &config); err != nil {
						t.Fatal(err)
					}
					params := config["rope_parameters"]
					delete(config, "rope_parameters")
					if schema.legacy && params != nil {
						rp := params.(map[string]any)
						rp["type"] = rp["rope_type"]
						delete(rp, "rope_type")
					}
					config[schema.field] = params
					limit := int(config["max_position_embeddings"].(float64))
					if tc.Name == "unextended_limit" || tc.Name == "implicit_original" {
						limit = 1048576
					}
					var envelope any = config
					if schema.nested {
						envelope = map[string]any{"text_config": config}
					}
					configData, err := json.Marshal(envelope)
					if err != nil {
						t.Fatal(err)
					}
					root := ropeTestRoot(t, configData)
					mlxtest.Run(t, func(t *mlxtest.T) {
						loaded, err := NewModel(root)
						if err != nil {
							t.Fatal(err)
						}
						m := loaded.(*Model)
						if m.MaxContextLength() != limit || int(m.RopeDim) != 2*len(tc.InvFreq) {
							t.Fatalf("context/rotary dimensions = %d/%d, want %d/%d", m.MaxContextLength(), m.RopeDim, limit, 2*len(tc.InvFreq))
						}
						dtype := mlx.DTypeFloat32
						outputTolerance, keyTolerance := 2e-3, 1e-5
						if tc.Dtype == "bfloat16" {
							dtype = mlx.DTypeBFloat16
							outputTolerance, keyTolerance = 2e-2, 3e-2
						} else if mode == "mrope" {
							keyTolerance = 1e-3
						}
						attention, x := ropeTestAttention(dtype)
						c := cache.NewKVCache()
						var layout []any
						if mode == "mrope" {
							positions := make([]int32, 0, 45)
							for channel := range 3 {
								positions = append(positions, reference.MropePositions[channel*16:channel*16+15]...)
							}
							layout = []any{&visionLayout{positions: positions, promptLen: 15, delta: reference.MropePositions[15] - 15}}
						}
						forward := func(c cache.Cache, offset, start, stop int32) *mlx.Array {
							length := stop - start
							b := &batch.Batch{
								InputIDs:   mlx.FromValues(make([]int32, length), 1, int(length)),
								SeqOffsets: []int32{offset + start}, SeqQueryLens: []int32{length}, Layout: layout,
							}
							var cos, sin *mlx.Array
							if positions := ropePositions(b, length); positions != nil {
								cos, sin = mropeCosSin(m.Config, positions, length)
							}
							chunk := mlx.SliceStartStop(x, []int32{0, start, 0}, []int32{1, stop, 256})
							return attention.Forward(chunk, b, c, mlx.FromValues([]int32{offset + start}, 1), 1, length, m.Config, cos, sin)
						}
						prefill, decode := forward(c, 0, 0, 15), forward(c, 0, 15, 16)
						output := mlx.Concatenate([]*mlx.Array{prefill, decode}, 1).AsType(mlx.DTypeFloat32)
						keys := c.State()[0].AsType(mlx.DTypeFloat32)
						rawKeys := mlx.Transpose(attention.KNorm.Forward(mlx.Reshape(attention.KProj.Forward(x), 1, 16, 1, 256), m.RMSNormEps), 0, 2, 1, 3).AsType(mlx.DTypeFloat32)
						mlx.Eval(output, keys, rawKeys)
						wantOutput, wantKeys := tc.TextOutput, tc.TextKeys
						if mode == "mrope" {
							wantOutput, wantKeys = tc.MropeOutput, tc.MropeKeys
						}
						check := func(label string, values, want []float32, tolerance float64) {
							if len(want) != len(reference.SampleTokens)*len(reference.SampleDims) {
								t.Fatalf("%s reference has %d samples", label, len(want))
							}
							index := 0
							for _, token := range reference.SampleTokens {
								for _, dim := range reference.SampleDims {
									got := values[token*256+dim]
									if diff := math.Abs(float64(got - want[index])); math.IsNaN(diff) || diff > tolerance {
										t.Fatalf("%s[%d,%d] = %g, want %g (tolerance %g)", label, token, dim, got, want[index], tolerance)
									}
									index++
								}
							}
						}
						check("cached key", keys.Floats(), wantKeys, keyTolerance)
						check("output", output.Floats(), wantOutput, outputTolerance)
						if mode == "text" {
							longCache := cache.NewKVCache()
							prefill := forward(longCache, 32768, 0, 15)
							decode := forward(longCache, 32768, 15, 16)
							longOutput := mlx.Concatenate([]*mlx.Array{prefill, decode}, 1).AsType(mlx.DTypeFloat32)
							mlx.Eval(longOutput)
							check("long-position output", longOutput.Floats(), tc.LongTextOutput, outputTolerance)
						}
						gotKeys, unrotated := keys.Floats(), rawKeys.Floats()
						for token := range 16 {
							for dim := int(m.RopeDim); dim < 256; dim++ {
								index := token*256 + dim
								if gotKeys[index] != unrotated[index] {
									t.Fatalf("passthrough key[%d,%d] changed: %g, want %g", token, dim, gotKeys[index], unrotated[index])
								}
							}
						}
					})
				})
			}
		}
	}
}

func ropeTestRoot(t *testing.T, config []byte) *model.Root {
	t.Helper()
	root := &model.Root{Manifest: &manifest.Manifest{}}
	for _, file := range []struct{ name, data string }{
		{"config.json", string(config)},
		{"tokenizer.json", `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`},
	} {
		layer, err := manifest.NewLayer(strings.NewReader(file.data), manifest.MediaTypeImageJSON)
		if err != nil {
			t.Fatal(err)
		}
		layer.Name = file.name
		root.Manifest.Layers = append(root.Manifest.Layers, layer)
	}
	return root
}

func ropeTestAttention(dtype mlx.DType) (*FullAttention, *mlx.Array) {
	weights, gated, normWeights := make([]float32, 256*256), make([]float32, 512*256), make([]float32, 256)
	for i := range 256 {
		weights[i*256+i], gated[i*256+i], normWeights[i] = 1, 1, 0.5
	}
	linear := nn.NewLinear(mlx.FromValues(weights, 256, 256).AsType(dtype), nil)
	norm := nn.NewRMSNorm(mlx.FromValues(normWeights, 256).AsType(dtype), 1e-6)
	attention := &FullAttention{
		QProj: nn.NewLinear(mlx.FromValues(gated, 512, 256).AsType(dtype), nil),
		KProj: linear, VProj: linear, OProj: linear, QNorm: norm, KNorm: norm,
	}
	data := make([]float32, 16*256)
	for i := range data {
		data[i] = float32(i%37-18) / 16
	}
	return attention, mlx.FromValues(data, 1, 16, 256).AsType(dtype)
}
