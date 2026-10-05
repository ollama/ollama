package discover

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestFilterOldCUDADriver(t *testing.T) {
	tests := []struct {
		name       string
		runtimeLib string
		devices    []ml.DeviceInfo
		wantIDs    []string
	}{
		{
			name:       "driver before CUDA compression support filters all CUDA devices",
			runtimeLib: "libcudart.so.12.8.90",
			devices: []ml.DeviceInfo{
				{
					DeviceID:          ml.DeviceID{ID: "GPU-0", Library: "CUDA"},
					Description:       "NVIDIA V100",
					ComputeMajor:      7,
					ComputeMinor:      0,
					NVIDIADriverMajor: 535,
				},
				{
					DeviceID:          ml.DeviceID{ID: "GPU-1", Library: "CUDA"},
					Description:       "NVIDIA L4",
					ComputeMajor:      8,
					ComputeMinor:      9,
					NVIDIADriverMajor: 535,
				},
				{
					DeviceID: ml.DeviceID{ID: "GPU-2", Library: "Vulkan"},
				},
			},
			wantIDs: []string{"GPU-2"},
		},
		{
			name:       "driver before legacy compute support filters only older CUDA devices",
			runtimeLib: "libcudart.so.12.8.90",
			devices: []ml.DeviceInfo{
				{
					DeviceID:          ml.DeviceID{ID: "GPU-0", Library: "CUDA"},
					Description:       "NVIDIA GTX 1080 Ti",
					ComputeMajor:      6,
					ComputeMinor:      1,
					NVIDIADriverMajor: 565,
				},
				{
					DeviceID:          ml.DeviceID{ID: "GPU-1", Library: "CUDA"},
					Description:       "NVIDIA RTX 6000",
					ComputeMajor:      7,
					ComputeMinor:      5,
					NVIDIADriverMajor: 565,
				},
			},
			wantIDs: []string{"GPU-1"},
		},
		{
			name:       "driver with full support keeps all CUDA devices",
			runtimeLib: "libcudart.so.12.8.90",
			devices: []ml.DeviceInfo{
				{
					DeviceID:          ml.DeviceID{ID: "GPU-0", Library: "CUDA"},
					Description:       "NVIDIA GTX 1080 Ti",
					ComputeMajor:      6,
					ComputeMinor:      1,
					NVIDIADriverMajor: 570,
				},
				{
					DeviceID:          ml.DeviceID{ID: "GPU-1", Library: "CUDA"},
					Description:       "NVIDIA RTX 6000",
					ComputeMajor:      7,
					ComputeMinor:      5,
					NVIDIADriverMajor: 570,
				},
			},
			wantIDs: []string{"GPU-0", "GPU-1"},
		},
		{
			name:       "source build with older CUDA runtime keeps devices",
			runtimeLib: "libcudart.so.12.2.140",
			devices: []ml.DeviceInfo{
				{
					DeviceID:          ml.DeviceID{ID: "GPU-0", Library: "CUDA"},
					Description:       "NVIDIA V100",
					ComputeMajor:      7,
					ComputeMinor:      0,
					NVIDIADriverMajor: 535,
				},
				{
					DeviceID:          ml.DeviceID{ID: "GPU-1", Library: "CUDA"},
					Description:       "NVIDIA GTX 1080 Ti",
					ComputeMajor:      6,
					ComputeMinor:      1,
					NVIDIADriverMajor: 535,
				},
			},
			wantIDs: []string{"GPU-0", "GPU-1"},
		},
		{
			name: "unknown driver keeps devices",
			devices: []ml.DeviceInfo{
				{
					DeviceID:     ml.DeviceID{ID: "GPU-0", Library: "CUDA"},
					Description:  "NVIDIA V100",
					ComputeMajor: 7,
					ComputeMinor: 0,
				},
			},
			wantIDs: []string{"GPU-0"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.runtimeLib != "" {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, tt.runtimeLib), nil, 0o644); err != nil {
					t.Fatal(err)
				}
				for i := range tt.devices {
					if tt.devices[i].Library == "CUDA" {
						tt.devices[i].LibraryPath = []string{dir}
					}
				}
			}

			got := filterOldCUDADriver(context.Background(), tt.devices)
			if len(got) != len(tt.wantIDs) {
				t.Fatalf("got %d devices, want %d: %#v", len(got), len(tt.wantIDs), got)
			}
			for i, wantID := range tt.wantIDs {
				if got[i].ID != wantID {
					t.Fatalf("device %d ID = %q, want %q", i, got[i].ID, wantID)
				}
			}
		})
	}
}
