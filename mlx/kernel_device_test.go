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
	})
}

func TestAppleGPU(t *testing.T) {
	type device struct {
		generation int
		size       byte
	}
	for architecture, want := range map[string]device{
		"applegpu_g17s": {17, 's'},
		"applegpu_g15d": {15, 'd'},
		"applegpu_g14d": {14, 'd'},
		"applegpu_g13g": {13, 'g'},
		"applegpu_g":    {},
		"applegpu_gxs":  {},
		"sm_90":         {},
		"":              {},
	} {
		generation, size := appleGPU(architecture)
		if got := (device{generation, size}); got != want {
			t.Errorf("appleGPU(%q) = %v, want %v", architecture, got, want)
		}
	}
}

// The device ladders keep the stock path where M2 can reject a pipeline when
// it is evaluated, keep the separate products M4 Pro measured faster, and
// otherwise follow the measured tunings.
func TestKernelDeviceTuning(t *testing.T) {
	t.Run("block_residual_mix", func(t *testing.T) {
		// Decode and 16 tokens, each plain then normalized.
		for arch, want := range map[string][4]bool{
			"applegpu_g14d": {false, false, true, false},
			"applegpu_g14s": {true, false, true, false},
			"applegpu_g15d": {true, true, true, true},
			"applegpu_g16s": {true, true, true, true},
			"applegpu_g17s": {true, true, true, true},
			"applegpu_g17d": {true, true, true, true},
		} {
			got := [4]bool{
				blockResidualMixRuns(1, false, arch), blockResidualMixRuns(1, true, arch),
				blockResidualMixRuns(16, false, arch), blockResidualMixRuns(16, true, arch),
			}
			if got != want {
				t.Errorf("%s: %v, want %v", arch, got, want)
			}
		}
	})
	t.Run("loop_controller", func(t *testing.T) {
		type controller struct {
			down    int
			grouped [2]bool // 8 and 16 rows
		}
		for arch, want := range map[string]controller{
			"applegpu_g14d": {0, [2]bool{true, true}},
			"applegpu_g14s": {0, [2]bool{true, true}},
			"applegpu_g15d": {4, [2]bool{true, true}},
			"applegpu_g16s": {0, [2]bool{true, false}},
			"applegpu_g17s": {4, [2]bool{true, true}},
			"applegpu_g17d": {4, [2]bool{true, true}},
		} {
			got := controller{loopControllerDownConfig(arch), [2]bool{groupedShortMatmulRuns(8, arch), groupedShortMatmulRuns(16, arch)}}
			if got != want {
				t.Errorf("%s: %+v, want %+v", arch, got, want)
			}
		}
	})
	t.Run("nvfp4_quantize", func(t *testing.T) {
		// Values per thread for 8, 64 and 129 rows of 2304, then 8 rows of 1152.
		for arch, want := range map[string][4]int{
			"applegpu_g14d": {16, 16, 16, 16},
			"applegpu_g14s": {16, 16, 16, 16},
			"applegpu_g15d": {4, 4, 16, 2},
			"applegpu_g16s": {4, 16, 16, 2},
			"applegpu_g17s": {4, 4, 16, 2},
			"applegpu_g17d": {4, 4, 16, 2},
		} {
			got := [4]int{
				tuneNVFP4Quantize(8, 2304, arch), tuneNVFP4Quantize(64, 2304, arch),
				tuneNVFP4Quantize(129, 2304, arch), tuneNVFP4Quantize(8, 1152, arch),
			}
			if got != want {
				t.Errorf("%s: %v, want %v", arch, got, want)
			}
		}
	})
	t.Run("short_gather", func(t *testing.T) {
		for _, tc := range []struct {
			arch     string
			mode     string
			rows     int
			perRoute bool
			want     shortGatherTuning
			ok       bool
		}{
			{"applegpu_g14d", "nvfp4", 1, false, shortGatherTuning{}, false},
			{"applegpu_g14s", "mxfp8", 1, true, shortGatherTuning{}, false},
			{"applegpu_g14d", "nvfp4", 2, false, shortGatherTuning{2, 2, 2, false}, true},
			{"applegpu_g15d", "nvfp4", 1, false, shortGatherTuning{2, 2, 1, false}, true},
			{"applegpu_g16s", "nvfp4", 8, false, shortGatherTuning{2, 2, 4, false}, true},
			{"applegpu_g16s", "nvfp4", 8, true, shortGatherTuning{4, 4, 4, false}, true},
			{"applegpu_g16s", "mxfp8", 16, true, shortGatherTuning{4, 2, 8, false}, true},
			{"applegpu_g17s", "nvfp4", 1, false, shortGatherTuning{2, 2, 1, true}, true},
			{"applegpu_g17s", "nvfp4", 8, false, shortGatherTuning{4, 4, 4, true}, true},
			{"applegpu_g17s", "nvfp4", 2, true, shortGatherTuning{2, 4, 2, true}, true},
			{"applegpu_g17d", "nvfp4", 8, true, shortGatherTuning{4, 4, 4, true}, true},
			{"applegpu_g17d", "mxfp8", 3, false, shortGatherTuning{4, 2, 3, false}, true},
		} {
			got, ok := tuneShortGather(tc.mode, tc.rows, tc.perRoute, tc.arch)
			if got != tc.want || ok != tc.ok {
				t.Errorf("%s %s rows %d per route %t: %+v %t, want %+v %t", tc.arch, tc.mode, tc.rows, tc.perRoute, got, ok, tc.want, tc.ok)
			}
		}
	})
}
