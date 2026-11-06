package nn

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestBuildYarnRopeFreqsReference(t *testing.T) {
	data, err := os.ReadFile("../testdata/yarn.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Cases []struct {
			Name                  string          `json:"name"`
			MaxPositionEmbeddings int32           `json:"max_position_embeddings"`
			RopeScaling           json.RawMessage `json:"rope_scaling"`
			InvFreq               []float32       `json:"inv_freq"`
			AttentionFactor       float32         `json:"attention_factor"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	for _, tc := range reference.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				var params *RopeParameters
				if err := json.Unmarshal(tc.RopeScaling, &params); err != nil {
					t.Fatal(err)
				}
				if params != nil && params.OriginalMaxPositionEmbeddings == 0 {
					params.OriginalMaxPositionEmbeddings = tc.MaxPositionEmbeddings
				}
				freqs, scale := BuildYarnRopeFreqs(128, 500000, params)
				if diff := math.Abs(float64(scale - tc.AttentionFactor)); math.IsNaN(diff) || diff > 1e-6 {
					t.Errorf("amplitude = %g, want %g", scale, tc.AttentionFactor)
				}
				if freqs == nil {
					if params != nil && params.Factor > 1 {
						t.Fatal("scaled frequencies are missing")
					}
					return
				}
				mlx.Eval(freqs)
				got := freqs.Floats()
				if len(got) != len(tc.InvFreq) {
					t.Fatalf("frequency count = %d, want %d", len(got), len(tc.InvFreq))
				}
				for i, want := range tc.InvFreq {
					inv := 1 / got[i]
					if rel := math.Abs(float64(inv-want)) / float64(want); math.IsNaN(rel) || rel > 3e-7 {
						t.Errorf("inverse frequency[%d] = %g, want %g", i, inv, want)
					}
				}
			})
		})
	}
}
