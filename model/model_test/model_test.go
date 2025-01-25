package modeltest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	_ "github.com/ollama/ollama/model/qwen2"
)

func TestForward(t *testing.T) {
	cases := []string{
		"qwen2",
		// Add more model architectures here...
	}

	for _, tt := range cases {
		t.Run(tt, func(t *testing.T) {
			t.Parallel()

			p := filepath.Join("testdata", tt)
			if testing.Short() {
				t.Skip("skipping in short mode")
			} else if _, err := os.Stat(p); err != nil {
				t.Skipf("%s not found", p)
			}

			f, err := os.CreateTemp(t.TempDir(), "f16")
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				f.Close()
				os.Remove(f.Name())
			}()

			if err := convert.ConvertModel(os.DirFS(p), f); err != nil {
				t.Fatal(err)
			}

			m, err := model.New(f.Name())
			if err != nil {
				t.Fatal(err)
			}
			b := m.Backend()
			ctx := b.NewContext()
			ctx.SetDebug(true)

			// Run forward pass
			_, err = model.Forward(ctx, m, model.WithCache(cache.NewCausalCache(m.Backend(), 2048, ml.DTypeF32)))
			if err != nil {
				t.Fatal(err)
			}

			// Validate the graph layers
			data, err := os.ReadFile(filepath.Join("testdata", tt+".json"))
			if err != nil {
				t.Fatal(err)
			}
			var expected ml.Graph
			if err := json.Unmarshal(data, &expected); err != nil {
				t.Fatal(err)
			}

			result := ctx.GetTrace()

			if len(result.Graph) != len(expected.Graph) {
				t.Errorf("expected %d layers, got %d", len(expected.Graph), len(result.Graph))
			}

			for i, layer := range expected.Graph {
				if i >= len(result.Graph) {
					break
				}
				actual := result.Graph[i]
				if layer.Name != actual.Name {
					t.Errorf("layer %d: expected name %s, got %s", i, layer.Name, actual.Name)
				}
				if !reflect.DeepEqual(layer.Shape, actual.Shape) {
					t.Errorf("layer %d: expected shape %v, got %v", i, layer.Shape, actual.Shape)
				}
			}
		})
	}
}
