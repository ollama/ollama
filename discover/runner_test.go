package discover

import (
	"context"
	"errors"
	"runtime"
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

func TestFilterIntegratedGPUs(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"}, Description: "NVIDIA integrated", Integrated: true},
		{DeviceID: ml.DeviceID{Library: "Metal", ID: "0"}, Description: "Apple GPU", Integrated: true},
		{DeviceID: ml.DeviceID{Library: "Vulkan", ID: "0"}, Description: "AMD Radeon(TM) Graphics", Integrated: true},
		{DeviceID: ml.DeviceID{Library: "ROCm", ID: "0"}, Description: "AMD Radeon 780M", Integrated: true, GFXTarget: "gfx1103"},
		{DeviceID: ml.DeviceID{Library: "ROCm", ID: "1"}, Description: "AMD Radeon 8060S Graphics", Integrated: true, GFXTarget: "gfx1151"},
		{DeviceID: ml.DeviceID{Library: "Vulkan", ID: "1"}, Description: "AMD Radeon RX 6800"},
	}

	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		t.Setenv("OLLAMA_IGPU_ENABLE", "false")
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{
			{Library: "CUDA", ID: "0"},
			{Library: "Metal", ID: "0"},
			{Library: "Vulkan", ID: "0"},
			{Library: "ROCm", ID: "0"},
			{Library: "ROCm", ID: "1"},
			{Library: "Vulkan", ID: "1"},
		}
		assertDeviceIDs(t, got, want)
		return
	}

	t.Run("auto admits only allowlisted integrated GPUs", func(t *testing.T) {
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{
			{Library: "CUDA", ID: "0"},
			{Library: "ROCm", ID: "1"},
			{Library: "Vulkan", ID: "1"},
		}
		assertDeviceIDs(t, got, want)
	})

	t.Run("explicit true admits all integrated GPUs", func(t *testing.T) {
		t.Setenv("OLLAMA_IGPU_ENABLE", "true")
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{
			{Library: "CUDA", ID: "0"},
			{Library: "Metal", ID: "0"},
			{Library: "Vulkan", ID: "0"},
			{Library: "ROCm", ID: "0"},
			{Library: "ROCm", ID: "1"},
			{Library: "Vulkan", ID: "1"},
		}
		assertDeviceIDs(t, got, want)
	})

	t.Run("explicit false drops integrated GPUs", func(t *testing.T) {
		t.Setenv("OLLAMA_IGPU_ENABLE", "false")
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{{Library: "Vulkan", ID: "1"}}
		assertDeviceIDs(t, got, want)
	})
}

func assertDeviceIDs(t *testing.T, got []ml.DeviceInfo, want []ml.DeviceID) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d devices, want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].DeviceID != want[i] {
			t.Fatalf("device %d = %#v, want %#v", i, got[i].DeviceID, want[i])
		}
	}
}

func TestRemapFilterIDForUserVisibleDevices(t *testing.T) {
	tests := []struct {
		name       string
		env        map[string]string
		device     ml.DeviceInfo
		wantID     string
		wantFilter string
	}{
		{
			name: "cuda numeric parent filter",
			env:  map[string]string{"CUDA_VISIBLE_DEVICES": "1"},
			device: ml.DeviceInfo{
				DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"},
				FilterID: "0",
			},
			wantID:     "0",
			wantFilter: "1",
		},
		{
			name: "cuda uuid parent filter",
			env:  map[string]string{"CUDA_VISIBLE_DEVICES": "GPU-f3a94ab8-b31d-61ff-9fbb-ce91ac1cdd95"},
			device: ml.DeviceInfo{
				DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"},
				FilterID: "0",
			},
			wantID:     "0",
			wantFilter: "GPU-f3a94ab8-b31d-61ff-9fbb-ce91ac1cdd95",
		},
		{
			name: "rocm hip parent filter",
			env:  map[string]string{"HIP_VISIBLE_DEVICES": "2,0"},
			device: ml.DeviceInfo{
				DeviceID: ml.DeviceID{Library: "ROCm", ID: "1"},
				FilterID: "1",
			},
			wantID:     "1",
			wantFilter: "0",
		},
		{
			name: "vulkan parent filter",
			env:  map[string]string{"GGML_VK_VISIBLE_DEVICES": "1"},
			device: ml.DeviceInfo{
				DeviceID: ml.DeviceID{Library: "Vulkan", ID: "0"},
				FilterID: "0",
			},
			wantID:     "0",
			wantFilter: "1",
		},
		{
			name: "no parent filter keeps internal filter id",
			device: ml.DeviceInfo{
				DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"},
				FilterID: "3",
			},
			wantID:     "0",
			wantFilter: "3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for key, value := range tt.env {
				t.Setenv(key, value)
			}

			remapFilterIDForUserVisibleDevices(&tt.device)

			if tt.device.ID != tt.wantID {
				t.Fatalf("ID = %q, want %q", tt.device.ID, tt.wantID)
			}
			if tt.device.FilterID != tt.wantFilter {
				t.Fatalf("FilterID = %q, want %q", tt.device.FilterID, tt.wantFilter)
			}
		})
	}
}

func TestNormalizeROCmDiscoveryEnv(t *testing.T) {
	tests := []struct {
		name        string
		env         map[string]string
		extra       map[string]string
		wantROCR    string
		wantSource  string
		wantOrdinal string
		wantSame    bool
	}{
		{
			name:        "hip becomes rocr",
			env:         map[string]string{"HIP_VISIBLE_DEVICES": "2"},
			wantROCR:    "2",
			wantSource:  "HIP_VISIBLE_DEVICES",
			wantOrdinal: "0",
		},
		{
			name:        "gpu ordinal becomes rocr",
			env:         map[string]string{"GPU_DEVICE_ORDINAL": "3"},
			wantROCR:    "3",
			wantSource:  "GPU_DEVICE_ORDINAL",
			wantOrdinal: "0",
		},
		{
			name:        "cuda numeric becomes rocr",
			env:         map[string]string{"CUDA_VISIBLE_DEVICES": "2,0"},
			wantROCR:    "2,0",
			wantSource:  "CUDA_VISIBLE_DEVICES",
			wantOrdinal: "0,1",
		},
		{
			name:     "rocr wins",
			env:      map[string]string{"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "2"},
			wantSame: true,
		},
		{
			name:     "cuda uuid does not become rocr",
			env:      map[string]string{"CUDA_VISIBLE_DEVICES": "GPU-f3a94ab8-b31d-61ff-9fbb-ce91ac1cdd95"},
			wantSame: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for key, value := range tt.env {
				t.Setenv(key, value)
			}

			got := normalizeDiscoveryEnvForGOOS("linux", []string{"/lib/ollama", "/lib/ollama/rocm_v7_2"}, tt.extra)

			if tt.wantSame {
				if got != nil && got["ROCR_VISIBLE_DEVICES"] != "" {
					t.Fatalf("ROCR_VISIBLE_DEVICES = %q, want unset", got["ROCR_VISIBLE_DEVICES"])
				}
				return
			}
			if got["ROCR_VISIBLE_DEVICES"] != tt.wantROCR {
				t.Fatalf("ROCR_VISIBLE_DEVICES = %q, want %q", got["ROCR_VISIBLE_DEVICES"], tt.wantROCR)
			}
			if got[tt.wantSource] != tt.wantOrdinal {
				t.Fatalf("%s = %q, want %q", tt.wantSource, got[tt.wantSource], tt.wantOrdinal)
			}
		})
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
		[]string{"/lib/ollama", "/lib/ollama/rocm_v7_2"},
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
