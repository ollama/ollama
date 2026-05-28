package ml

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

func TestMergeEnvWithRunnerEnvOverrides(t *testing.T) {
	devices := []DeviceInfo{
		{
			DeviceID:           DeviceID{Library: "Metal", ID: "0"},
			RunnerEnvOverrides: map[string]string{"GGML_METAL_TENSOR_DISABLE": "1"},
		},
	}

	env := GetDevicesEnv(devices)

	if got, want := env["GGML_METAL_TENSOR_DISABLE"], "1"; got != want {
		t.Fatalf("GGML_METAL_TENSOR_DISABLE = %q, want %q", got, want)
	}
}

func TestGetDevicesEnvWarnsOnConflictingOverrides(t *testing.T) {
	var logs bytes.Buffer
	oldLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() {
		slog.SetDefault(oldLogger)
	})

	devices := []DeviceInfo{
		{
			DeviceID:           DeviceID{Library: "Metal", ID: "0"},
			RunnerEnvOverrides: map[string]string{"TEST_OVERRIDE": "one"},
		},
		{
			DeviceID:           DeviceID{Library: "Metal", ID: "1"},
			RunnerEnvOverrides: map[string]string{"TEST_OVERRIDE": "two"},
		},
	}

	env := GetDevicesEnv(devices)

	if got, want := env["TEST_OVERRIDE"], "two"; got != want {
		t.Fatalf("TEST_OVERRIDE = %q, want %q", got, want)
	}

	if !strings.Contains(logs.String(), "conflicting device environment override") {
		t.Fatalf("expected warning log, got %q", logs.String())
	}
}

func TestGetDevicesEnvFiltersVisibleDevices(t *testing.T) {
	tests := []struct {
		name string
		gpus []DeviceInfo
		key  string
		want string
	}{
		{
			name: "single CUDA",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA", ID: "3"}}},
			key:  "CUDA_VISIBLE_DEVICES",
			want: "3",
		},
		{
			name: "multiple CUDA",
			gpus: []DeviceInfo{
				{DeviceID: DeviceID{Library: "CUDA", ID: "3"}},
				{DeviceID: DeviceID{Library: "CUDA", ID: "4"}},
			},
			key:  "CUDA_VISIBLE_DEVICES",
			want: "3,4",
		},
		{
			name: "multiple remapped CUDA",
			gpus: []DeviceInfo{
				{DeviceID: DeviceID{Library: "CUDA", ID: "0"}, FilterID: "0"},
				{DeviceID: DeviceID{Library: "CUDA", ID: "1"}, FilterID: "2"},
			},
			key:  "CUDA_VISIBLE_DEVICES",
			want: "0,2",
		},
		{
			name: "single Vulkan",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "Vulkan", ID: "0"}, FilterID: "1"}},
			key:  "GGML_VK_VISIBLE_DEVICES",
			want: "1",
		},
		{
			name: "multiple Vulkan",
			gpus: []DeviceInfo{
				{DeviceID: DeviceID{Library: "Vulkan", ID: "0"}, FilterID: "1"},
				{DeviceID: DeviceID{Library: "Vulkan", ID: "1"}, FilterID: "0"},
			},
			key: "GGML_VK_VISIBLE_DEVICES",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := GetDevicesEnv(tt.gpus)
			if got := env[tt.key]; got != tt.want {
				t.Fatalf("%s = %q, want %q", tt.key, got, tt.want)
			}
		})
	}
}

func TestDeviceCompareVulkanDuplicates(t *testing.T) {
	tests := []struct {
		name string
		a    DeviceInfo
		b    DeviceInfo
		want DeviceComparison
	}{
		{
			name: "vulkan duplicate with missing pci",
			a: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA", ID: "0"},
				Description: "NVIDIA GeForce RTX 4060 Ti",
				PCIID:       "0000:01:00.0",
				TotalMemory: 16 * 1024 * 1024 * 1024,
			},
			b: DeviceInfo{
				DeviceID:    DeviceID{Library: "Vulkan", ID: "2"},
				Description: "NVIDIA GeForce RTX 4060 Ti",
				TotalMemory: 16107 * 1024 * 1024,
			},
			want: DuplicateDevice,
		},
		{
			name: "vulkan radv suffix duplicate",
			a: DeviceInfo{
				DeviceID:    DeviceID{Library: "ROCm", ID: "0"},
				Description: "AMD Radeon RX 6400",
				PCIID:       "0000:03:00.0",
				TotalMemory: 4 * 1024 * 1024 * 1024,
			},
			b: DeviceInfo{
				DeviceID:    DeviceID{Library: "Vulkan", ID: "0"},
				Description: "AMD Radeon RX 6400 (RADV NAVI24)",
				TotalMemory: 4096 * 1024 * 1024,
			},
			want: DuplicateDevice,
		},
		{
			name: "different vulkan igpu",
			a: DeviceInfo{
				DeviceID:    DeviceID{Library: "CUDA", ID: "0"},
				Description: "NVIDIA GeForce RTX 4060 Ti",
				PCIID:       "0000:01:00.0",
				TotalMemory: 16 * 1024 * 1024 * 1024,
			},
			b: DeviceInfo{
				DeviceID:    DeviceID{Library: "Vulkan", ID: "0"},
				Description: "Intel(R) UHD Graphics 770",
				TotalMemory: 32 * 1024 * 1024 * 1024,
			},
			want: UniqueDevice,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.a.Compare(tt.b); got != tt.want {
				t.Fatalf("Compare = %v, want %v", got, tt.want)
			}
			if got := tt.b.Compare(tt.a); got != tt.want {
				t.Fatalf("reverse Compare = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFlashAttentionSupported(t *testing.T) {
	tests := []struct {
		name string
		gpus []DeviceInfo
		want bool
	}{
		{
			name: "cuda compute 5.0 unsupported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 5, ComputeMinor: 0}},
		},
		{
			name: "cuda compute 6.2 unsupported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 6, ComputeMinor: 2}},
		},
		{
			name: "cuda compute 7.2 unsupported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 7, ComputeMinor: 2}},
		},
		{
			name: "cuda compute 7.0 supported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 7, ComputeMinor: 0}},
			want: true,
		},
		{
			name: "cuda compute 7.5 supported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 7, ComputeMinor: 5}},
			want: true,
		},
		{
			name: "cuda unknown driver supported by compute",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, ComputeMajor: 8, ComputeMinor: 9}},
			want: true,
		},
		{
			name: "cuda unknown compute unsupported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: -1, ComputeMinor: -1}},
		},
		{
			name: "mixed cuda unsupported",
			gpus: []DeviceInfo{
				{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 8, ComputeMinor: 9},
				{DeviceID: DeviceID{Library: "CUDA"}, DriverMajor: 12, ComputeMajor: 6, ComputeMinor: 2},
			},
		},
		{
			name: "metal supported",
			gpus: []DeviceInfo{{DeviceID: DeviceID{Library: "Metal"}}},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FlashAttentionSupported(tt.gpus); got != tt.want {
				t.Fatalf("FlashAttentionSupported = %t, want %t", got, tt.want)
			}
		})
	}
}

func TestFlashAttentionSupportedWarnsWhenCUDADriverUnknown(t *testing.T) {
	var logs bytes.Buffer
	oldLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() {
		slog.SetDefault(oldLogger)
	})

	gpus := []DeviceInfo{{
		DeviceID:     DeviceID{Library: "CUDA"},
		Description:  "NVIDIA RTX",
		ComputeMajor: 8,
		ComputeMinor: 9,
	}}

	if !FlashAttentionSupported(gpus) {
		t.Fatal("expected unknown-driver modern CUDA GPU to allow flash attention")
	}
	if !strings.Contains(logs.String(), "CUDA driver version unavailable") {
		t.Fatalf("expected unknown driver warning, got %q", logs.String())
	}
}
