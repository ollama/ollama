package mlx

import (
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestMetalTensorOpsFallback(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if hasMetalTensorOps() {
			t.Skip("requires a backend without Metal tensor ops")
		}
		// These shapes qualify on a Metal 4 driver. An older driver must
		// return to the native model path before constructing the shader.
		q := Zeros(DTypeFloat32, 1, 1, 16, 128)
		k := Zeros(DTypeBFloat16, 1, 1, 16, 128)
		table := Zeros(DTypeBFloat16, 1, 1, 16, 16)
		sink := Zeros(DTypeBFloat16, 1, 1, 16, 1)
		if RelativePositionAttentionBF16Probabilities(q, k, k, table, sink, 0, 16, 0.125) != nil {
			t.Fatal("selected MPP attention")
		}
		x := Zeros(DTypeBFloat16, 16, 1, 1, 32)
		w := Zeros(DTypeUint32, 1, 64, 4)
		s := Zeros(DTypeUint8, 1, 64, 2)
		ids := Zeros(DTypeInt32, 16, 1)
		if PrefersCompactNVFP4QMM() || GatherNVFP4QMM(x, w, s, ids) != nil {
			t.Fatal("selected MPP grouped matmul")
		}
	})
}

func TestM5MatrixDevice(t *testing.T) {
	for _, tc := range []struct {
		name, architecture string
		want               bool
	}{
		{"Apple M5 Max", "applegpu_g17s", true},
		{"Apple M5 Ultra", "applegpu_g17d", true},
		{"Apple M5 Max", "applegpu_g17d", false},
		{"Apple M5 Pro", "applegpu_g17s", true},
		{"Apple M3 Ultra", "applegpu_g15d", false},
	} {
		if got := isM5MatrixDevice(tc.name, tc.architecture); got != tc.want {
			t.Errorf("device %s/%s returns %t, want %t", tc.name, tc.architecture, got, tc.want)
		}
	}
}
