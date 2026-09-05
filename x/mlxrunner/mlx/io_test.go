package mlx

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthreadtest"
)

func TestSaveSafetensorsWithMetadataDeterministic(t *testing.T) {
	root := t.TempDir()
	withMLXThread(t, func(t *mlxthreadtest.T) {
		values := map[string]*Array{
			"model.layers.0.a.weight": FromValues([]float32{1, 2, 3, 4}, 2, 2),
			"model.layers.0.m.weight": FromValues([]float32{5, 6}, 1, 2),
			"model.layers.0.z.weight": FromValues([]float32{7, 8}, 2, 1),
		}
		Eval(values["model.layers.0.a.weight"], values["model.layers.0.m.weight"], values["model.layers.0.z.weight"])

		arrayNames := []string{"model.layers.0.a.weight", "model.layers.0.m.weight", "model.layers.0.z.weight"}
		metadataValues := map[string]string{"alpha": "first", "middle": "second", "zeta": "last"}
		metadataKeys := []string{"alpha", "middle", "zeta"}
		var want [sha256.Size]byte
		for attempt := range 6 {
			arrays := make(map[string]*Array, len(values))
			metadata := make(map[string]string, len(metadataValues))
			for i := range arrayNames {
				name := arrayNames[(i+attempt)%len(arrayNames)]
				arrays[name] = values[name]
				key := metadataKeys[(i+attempt)%len(metadataKeys)]
				metadata[key] = metadataValues[key]
			}

			path := filepath.Join(root, fmt.Sprintf("model-%d.safetensors", attempt))
			if err := SaveSafetensorsWithMetadata(path, arrays, metadata); err != nil {
				t.Fatalf("SaveSafetensorsWithMetadata() attempt %d error = %v", attempt, err)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read attempt %d: %v", attempt, err)
			}
			got := sha256.Sum256(data)
			if attempt == 0 {
				want = got
			} else if got != want {
				t.Fatalf("attempt %d digest = %x, want %x", attempt, got, want)
			}
		}
	})
}
