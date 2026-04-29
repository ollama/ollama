package discover

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

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

func TestRecordPersistentRunnerEnv(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "Metal", ID: "0"}},
		{DeviceID: ml.DeviceID{Library: "CUDA", ID: "1"}},
	}

	recordPersistentRunnerEnv(devices, map[string]string{
		"GGML_METAL_TENSOR_DISABLE": "1",
		"CUDA_VISIBLE_DEVICES":      "1",
	})

	if got := devices[0].RunnerEnvOverrides["GGML_METAL_TENSOR_DISABLE"]; got != "1" {
		t.Fatalf("Metal RunnerEnvOverrides = %q, want %q", got, "1")
	}

	if _, ok := devices[0].RunnerEnvOverrides["CUDA_VISIBLE_DEVICES"]; ok {
		t.Fatal("unexpected CUDA_VISIBLE_DEVICES in Metal RunnerEnvOverrides")
	}

	if devices[1].RunnerEnvOverrides != nil {
		t.Fatalf("unexpected RunnerEnvOverrides recorded for non-Metal device: %#v", devices[1].RunnerEnvOverrides)
	}
}

func TestBootstrapDevicesWithStatusWatchdogReturnsResult(t *testing.T) {
	want := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"}}}
	devices, _, err := runBootstrapDevicesWithStatusWatchdog(
		t.Context(),
		[]string{"/lib/ollama", "/lib/ollama/cuda_v12"},
		nil,
		func(context.Context, []string, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
			return want, nil, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(devices) != 1 || devices[0].DeviceID != want[0].DeviceID {
		t.Fatalf("devices = %#v, want %#v", devices, want)
	}
}

func TestBootstrapDevicesWithStatusWatchdogReturnsOnDeadline(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
	defer cancel()

	started := make(chan struct{})
	release := make(chan struct{})
	finished := make(chan struct{})

	_, _, err := runBootstrapDevicesWithStatusWatchdog(
		ctx,
		[]string{"/lib/ollama", "/lib/ollama/rocm"},
		nil,
		func(context.Context, []string, map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
			close(started)
			defer close(finished)
			<-release
			return nil, nil, nil
		},
	)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context deadline exceeded", err)
	}
	close(release)

	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("discovery function was not called")
	}
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("discovery function did not finish after release")
	}
}
