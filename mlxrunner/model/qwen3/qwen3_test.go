package qwen3

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

func TestAttentionRopeReference(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	data, err := os.ReadFile("../../testdata/yarn.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		SampleTokens []int `json:"sample_tokens"`
		SampleDims   []int `json:"sample_dims"`
		Cases        []struct {
			Name                  string          `json:"name"`
			MaxPositionEmbeddings int             `json:"max_position_embeddings"`
			RopeScaling           json.RawMessage `json:"rope_scaling"`
			AttentionFactor       float32         `json:"attention_factor"`
			OutputSamples         []float32       `json:"output_samples"`
			KeySamples            []float32       `json:"key_samples"`
			LongKeySamples        []float32       `json:"long_key_samples"`
			LongOutputSamples     []float32       `json:"long_output_samples"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	for _, schema := range []struct {
		field  string
		legacy bool
	}{
		{field: "rope_scaling"},
		{field: "rope_parameters"},
		{field: "rope_scaling", legacy: true},
	} {
		name := schema.field
		if schema.legacy {
			name += "_legacy_type"
		}
		for _, tc := range reference.Cases {
			t.Run(name+"/"+tc.Name, func(t *testing.T) {
				config := map[string]any{
					"hidden_size": 128, "num_hidden_layers": 1,
					"num_attention_heads": 1, "num_key_value_heads": 1,
					"head_dim": 128, "rms_norm_eps": 1e-6,
					"rope_theta": 500000, "max_position_embeddings": tc.MaxPositionEmbeddings,
				}
				var params map[string]any
				if err := json.Unmarshal(tc.RopeScaling, &params); err != nil {
					t.Fatal(err)
				}
				if schema.field == "rope_parameters" {
					if params == nil {
						params = map[string]any{"rope_type": "default"}
					}
					params["rope_theta"] = 500000
					delete(config, "rope_theta")
				}
				if schema.legacy && params != nil {
					params["type"] = params["rope_type"]
					delete(params, "rope_type")
				}
				config[schema.field] = params
				configData, err := json.Marshal(config)
				if err != nil {
					t.Fatal(err)
				}
				root := testModelRoot(t, string(configData))
				mlxtest.Run(t, func(t *mlxtest.T) {
					loaded, err := newModel(root)
					if err != nil {
						t.Fatal(err)
					}
					m := loaded.(*Model)
					limit := tc.MaxPositionEmbeddings
					if tc.Name == "unextended_limit" || tc.Name == "implicit_original" {
						limit = 131072
					}
					if got := m.MaxContextLength(); got != limit {
						t.Fatalf("context limit = %d, want %d", got, limit)
					}
					attention, x := testAttention()
					c := cache.NewKVCache()
					forward := func(c cache.Cache, offset, start, stop int32) *mlx.Array {
						length := stop - start
						b := &batch.Batch{
							InputIDs:   mlx.FromValues(make([]int32, length), 1, int(length)),
							SeqOffsets: []int32{offset + start}, SeqQueryLens: []int32{length},
						}
						chunk := mlx.SliceStartStop(x, []int32{0, start, 0}, []int32{1, stop, 128})
						return attention.Forward(chunk, b, c, mlx.FromValues([]int32{offset + start}, 1), 1, length, m.Config)
					}
					prefill := forward(c, 0, 0, 15)
					decode := forward(c, 0, 15, 16)
					out := mlx.Concatenate([]*mlx.Array{prefill, decode}, 1)
					// Restore the reference amplitude for cached-key comparisons.
					keys := mlx.MulScalar(c.State()[0], tc.AttentionFactor)
					mlx.Eval(out, keys)
					check := func(label string, values, want []float32, tolerance float64) {
						if len(want) != len(reference.SampleTokens)*len(reference.SampleDims) {
							t.Fatalf("%s reference has %d samples", label, len(want))
						}
						index := 0
						for _, token := range reference.SampleTokens {
							for _, dim := range reference.SampleDims {
								got := values[token*128+dim]
								if math.IsNaN(float64(got)) || math.Abs(float64(got-want[index])) > tolerance {
									t.Fatalf("%s[%d,%d] = %g, want %g (tolerance %g)", label, token, dim, got, want[index], tolerance)
								}
								index++
							}
						}
					}
					check("output", out.Floats(), tc.OutputSamples, 2e-3)
					check("cached key", keys.Floats(), tc.KeySamples, 1e-5)

					longCache := cache.NewKVCache()
					longPrefill := forward(longCache, 32768, 0, 15)
					longDecode := forward(longCache, 32768, 15, 16)
					longOut := mlx.Concatenate([]*mlx.Array{longPrefill, longDecode}, 1)
					longKeys := mlx.MulScalar(longCache.State()[0], tc.AttentionFactor)
					mlx.Eval(longOut, longKeys)
					check("long-position output", longOut.Floats(), tc.LongOutputSamples, 2e-3)
					check("long-position key", longKeys.Floats(), tc.LongKeySamples, 2e-3)
				})
			})
		}
	}
}

func testModelRoot(t *testing.T, config string) *model.Root {
	t.Helper()
	root := &model.Root{Manifest: &manifest.Manifest{}}
	for _, file := range []struct{ name, data string }{
		{"config.json", config},
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

func testAttention() (*Attention, *mlx.Array) {
	weights, normWeights := make([]float32, 128*128), make([]float32, 128)
	for i := range 128 {
		weights[i*128+i] = 1
		normWeights[i] = 0.5
	}
	linear := nn.NewLinear(mlx.FromValues(weights, 128, 128), nil)
	norm := nn.NewRMSNorm(mlx.FromValues(normWeights, 128), 1e-6)
	attention := &Attention{QProj: linear, KProj: linear, VProj: linear, OProj: linear, QNorm: norm, KNorm: norm}
	data := make([]float32, 16*128)
	for i := range data {
		data[i] = float32(i%37-18) / 16
	}
	return attention, mlx.FromValues(data, 1, 16, 128)
}
