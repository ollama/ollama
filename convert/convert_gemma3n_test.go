package convert

import (
	"encoding/json"
	"math"
	"testing"

	"gonum.org/v1/gonum/stat/distuv"
)

func TestGemma3nIntermediateSize(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    gemma3nIntermediateSize
		wantErr bool
	}{
		{
			name: "scalar",
			json: `8192`,
			want: 8192,
		},
		{
			name: "uniform array",
			json: `[8192,8192,8192]`,
			want: 8192,
		},
		{
			name:    "mixed array",
			json:    `[8192,4096]`,
			wantErr: true,
		},
		{
			name:    "empty array",
			json:    `[]`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got gemma3nIntermediateSize
			err := json.Unmarshal([]byte(tt.json), &got)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestGemma3nActivationSparsityScale(t *testing.T) {
	tok := &Tokenizer{Vocabulary: &Vocabulary{}}

	kvFor := func(pattern []float32) []float32 {
		var m gemma3nModel
		m.TextModel.ActivationSparsityPattern = pattern
		scale, ok := m.KV(tok)["gemma3n.activation_sparsity_scale"].([]float32)
		if !ok {
			t.Fatalf("gemma3n.activation_sparsity_scale is not a []float32")
		}
		return scale
	}

	t.Run("in range", func(t *testing.T) {
		// 0.0 and 1.0 map to the -Inf/+Inf boundaries and 0.95 to the standard
		// normal quantile; guard the values that a valid config actually uses.
		got := kvFor([]float32{0.0, 0.95, 1.0})
		if len(got) != 3 {
			t.Fatalf("got %d scales, want 3", len(got))
		}
		if !math.IsInf(float64(got[0]), -1) {
			t.Errorf("scale for 0.0 = %v, want -Inf", got[0])
		}
		if want := float32(distuv.Normal{Mu: 0, Sigma: 1}.Quantile(float64(float32(0.95)))); got[1] != want {
			t.Errorf("scale for 0.95 = %v, want %v", got[1], want)
		}
		if !math.IsInf(float64(got[2]), 1) {
			t.Errorf("scale for 1.0 = %v, want +Inf", got[2])
		}
	})

	t.Run("out of range does not panic", func(t *testing.T) {
		// A value outside [0, 1] previously panicked distuv.Normal.Quantile and
		// crashed conversion; it must now yield NaN instead.
		got := kvFor([]float32{0.0, 0.95, 2.0})
		if len(got) != 3 {
			t.Fatalf("got %d scales, want 3", len(got))
		}
		if !math.IsNaN(float64(got[2])) {
			t.Errorf("scale for 2.0 = %v, want NaN", got[2])
		}
	})
}
