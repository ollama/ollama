package discover

import (
	"log/slog"
	"os"
	"testing"
)

func init() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
}

func TestFilterOverlapByLibrary(t *testing.T) {
	type testcase struct {
		name string
		inp  map[string]map[string]map[string]int
		exp  []bool
	}
	for _, tc := range []testcase{
		{
			name: "empty",
			inp:  map[string]map[string]map[string]int{},
			exp:  []bool{}, // needs deletion
		},
		{
			name: "single no overlap",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
					},
				},
			},
			exp: []bool{false},
		},
		{
			name: "100% overlap pick 2nd",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 1,
					},
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 2,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 3,
					},
				},
			},
			exp: []bool{true, true, false, false},
		},
		{
			name: "100% overlap pick 1st",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 1,
					},
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 2,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 3,
					},
				},
			},
			exp: []bool{false, false, true, true},
		},
		{
			name: "partial overlap pick older",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
					},
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 1,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 2,
					},
				},
			},
			exp: []bool{true, false, false},
		},
		{
			name: "no overlap",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
					},
					"cuda_v12": {
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 1,
					},
				},
			},
			exp: []bool{false, false},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			needsDelete := make([]bool, len(tc.exp))
			filterOverlapByLibrary(tc.inp, needsDelete)
			for i, exp := range tc.exp {
				if needsDelete[i] != exp {
					t.Fatalf("expected: %v\ngot: %v", tc.exp, needsDelete)
				}
			}
		})
	}
}
