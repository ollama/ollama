package mlxrunner

import (
	"reflect"
	"sort"
	"testing"
)

// TestNormaliseAuxNames_PluralAndSingular verifies that normaliseAuxNames
// targets the canonical "<base>.weight_scale" / "<base>.weight_qbias" form for
// both the Ollama-native dot-child singular aux naming ("<weight>.scale" /
// "<weight>.bias") and the mlx-lm sibling-plural naming ("<module>.scales" /
// "<module>.biases").
//
// The helper is generic over the value type so this test can run as pure Go
// without touching the MLX runtime; each tensor is represented by a unique
// integer sentinel to confirm the VALUE is preserved through the remap.
func TestNormaliseAuxNames_PluralAndSingular(t *testing.T) {
	cases := []struct {
		name string
		raw  map[string]int
		want map[string]int
	}{
		{
			name: "singular dot-child",
			raw: map[string]int{
				"layer.weight":       1,
				"layer.weight.scale": 2,
				"layer.weight.bias":  3,
			},
			want: map[string]int{
				"layer.weight":       1,
				"layer.weight_scale": 2,
				"layer.weight_qbias": 3,
			},
		},
		{
			name: "mlx-lm sibling plural",
			raw: map[string]int{
				"layer.weight": 1,
				"layer.scales": 2,
				"layer.biases": 3,
			},
			want: map[string]int{
				"layer.weight":       1,
				"layer.weight_scale": 2,
				"layer.weight_qbias": 3,
			},
		},
		{
			name: "mixed conventions in one blob",
			raw: map[string]int{
				"a.weight":       1,
				"a.weight.scale": 2,
				"b.weight":       3,
				"b.scales":       4,
			},
			want: map[string]int{
				"a.weight":       1,
				"a.weight_scale": 2,
				"b.weight":       3,
				"b.weight_scale": 4,
			},
		},
		{
			name: "dense bias with no matching scale passes through unchanged",
			raw: map[string]int{
				"dense.weight": 1,
				"dense.bias":   2,
			},
			want: map[string]int{
				"dense.weight": 1,
				"dense.bias":   2,
			},
		},
		{
			name: "plural bias with no matching scale passes through unchanged",
			raw: map[string]int{
				"stats.biases": 5,
			},
			want: map[string]int{
				"stats.biases": 5,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := normaliseAuxNames(tc.raw)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf(
					"normaliseAuxNames() mismatch\n got keys: %v\nwant keys: %v",
					sortedKeys(got), sortedKeys(tc.want),
				)
				for k, v := range got {
					if tc.want[k] != v {
						t.Errorf("  %q: got %d, want %d", k, v, tc.want[k])
					}
				}
			}
		})
	}
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
