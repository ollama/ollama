package discover

import (
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/logutil"
)

func TestLlamaServerDiscoveryOutputOnlyTrace(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() {
		slog.SetDefault(original)
	})

	slog.SetDefault(logutil.NewLogger(io.Discard, slog.LevelDebug))
	if got := llamaServerDiscoveryOutput(t.Context()); got != io.Discard {
		t.Fatal("debug logging should discard raw llama-server discovery output")
	}

	slog.SetDefault(logutil.NewLogger(io.Discard, logutil.LevelTrace))
	if got := llamaServerDiscoveryOutput(t.Context()); got == io.Discard {
		t.Fatal("trace logging should emit raw llama-server discovery output")
	}
}

func TestParseLlamaServerDevices(t *testing.T) {
	tests := []struct {
		name     string
		output   string
		libDirs  []string
		wantLen  int
		wantName string
		wantLib  string
		wantMiB  uint64
	}{
		{
			name: "NVIDIA CUDA",
			output: `load_backend: loaded CUDA backend from /lib/ollama/cuda_v12/libggml-cuda.so
Available devices:
  NVIDIA GeForce RTX 4090: NVIDIA CUDA (24564 MiB, 23592 MiB free)
`,
			libDirs:  []string{"/lib/ollama", "/lib/ollama/cuda_v12"},
			wantLen:  1,
			wantName: "NVIDIA GeForce RTX 4090",
			wantLib:  "CUDA",
			wantMiB:  24564,
		},
		{
			name: "Metal",
			output: `Available devices:
  Metal: Apple M3 Max (98304 MiB, 98303 MiB free)
`,
			libDirs:  []string{"/lib/ollama"},
			wantLen:  1,
			wantName: "Metal",
			wantLib:  "Metal",
			wantMiB:  98304,
		},
		{
			name: "ROCm with gfx target",
			output: `  Device 0: AMD Radeon RX 6700 XT, gfx1031 (0x1031), VMM: no, Wave Size: 32, VRAM: 12272 MiB
Available devices:
  ROCm0: AMD Radeon RX 6700 XT (12272 MiB, 12248 MiB free)
`,
			libDirs:  []string{"/lib/ollama", "/lib/ollama/rocm"},
			wantLen:  1,
			wantName: "ROCm0",
			wantLib:  "ROCm",
			wantMiB:  12272,
		},
		{
			name: "multi GPU",
			output: `Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 23592 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (12288 MiB, 11500 MiB free)
`,
			libDirs: []string{"/lib/ollama", "/lib/ollama/cuda_v12"},
			wantLen: 2,
		},
		{
			name:    "no devices",
			output:  "Available devices:\n",
			libDirs: []string{"/lib/ollama"},
			wantLen: 0,
		},
		{
			name:    "empty output",
			output:  "",
			libDirs: []string{"/lib/ollama"},
			wantLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			devices := parseLlamaServerDevices(tt.output, tt.libDirs)
			if len(devices) != tt.wantLen {
				t.Fatalf("got %d devices, want %d", len(devices), tt.wantLen)
			}
			if tt.wantLen > 0 {
				if tt.wantName != "" && devices[0].Name != tt.wantName {
					t.Errorf("name = %q, want %q", devices[0].Name, tt.wantName)
				}
				if tt.wantLib != "" && devices[0].Library != tt.wantLib {
					t.Errorf("library = %q, want %q", devices[0].Library, tt.wantLib)
				}
				if tt.wantMiB > 0 {
					expectedBytes := tt.wantMiB * 1024 * 1024
					if devices[0].TotalMemory != expectedBytes {
						t.Errorf("total memory = %d, want %d", devices[0].TotalMemory, expectedBytes)
					}
				}
			}
		})
	}
}

func TestParseLlamaServerDevicesMarksVulkanUMAGPUsIntegrated(t *testing.T) {
	output := `ggml_vulkan: 0 = Intel(R) Graphics (Intel open-source Mesa driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
Available devices:
  Vulkan0: Intel(R) Graphics (16384 MiB, 12288 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama", "/lib/ollama/vulkan"})
	if len(devices) != 1 {
		t.Fatalf("expected 1 device, got %d", len(devices))
	}
	if !devices[0].Integrated {
		t.Fatal("expected Vulkan UMA device to be marked integrated")
	}
}

func TestCUDADeviceFilteredByArchs(t *testing.T) {
	// GTX 1060 (CC 6.1 = 610) with v13 ARCHS that don't include 610
	output := `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 6063 MiB):
  Device 0: NVIDIA GeForce GTX 1060 6GB, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
load_backend: loaded CUDA backend from /lib/ollama/cuda_v13/libggml-cuda.so
system_info: n_threads = 4 | CUDA : ARCHS = 750,800,860,890,900,1000,1030,1100,1200,1210 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama", "/lib/ollama/cuda_v13"})
	if len(devices) != 0 {
		t.Fatalf("expected 0 devices (GTX 1060 CC 610 not in ARCHS), got %d", len(devices))
	}
}

func TestCUDADeviceKeptByArchs(t *testing.T) {
	// RTX 4060 Ti (CC 8.9 = 890) with v13 ARCHS that include 890
	output := `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16379 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB
system_info: n_threads = 16 | CUDA : ARCHS = 750,800,860,890,900,1000,1030,1100,1200,1210 |
Available devices:
  CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama"})
	if len(devices) != 1 {
		t.Fatalf("expected 1 device (CC 890 in ARCHS), got %d", len(devices))
	}
	if devices[0].ComputeMajor != 8 || devices[0].ComputeMinor != 9 {
		t.Fatalf("expected compute 8.9, got %s", devices[0].Compute())
	}
}

func TestCUDANoArchsFailOpen(t *testing.T) {
	// No system_info line — should keep all devices (fail open)
	output := `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 6063 MiB):
  Device 0: NVIDIA GeForce GTX 1060 6GB, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama"})
	if len(devices) != 1 {
		t.Fatalf("expected 1 device (no ARCHS = fail open), got %d", len(devices))
	}
	if devices[0].ComputeMajor != 6 || devices[0].ComputeMinor != 1 {
		t.Fatalf("expected compute 6.1, got %s", devices[0].Compute())
	}
}

func TestCUDANoCCFailOpen(t *testing.T) {
	// Device line without compute capability — should keep (fail open)
	output := `system_info: n_threads = 4 | CUDA : ARCHS = 750,800 |
Available devices:
  CUDA0: Some Future GPU (8192 MiB, 8000 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama"})
	if len(devices) != 1 {
		t.Fatalf("expected 1 device (no CC = fail open), got %d", len(devices))
	}
}

func TestCUDAMultiDeviceMixedFilter(t *testing.T) {
	// Two devices: one supported (CC 890), one not (CC 610)
	output := `ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce GTX 1060, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
  Device 1: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB
system_info: n_threads = 8 | CUDA : ARCHS = 750,800,860,890 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 (6063 MiB, 5900 MiB free)
  CUDA1: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama"})
	if len(devices) != 1 {
		t.Fatalf("expected 1 device (only RTX 4060 Ti), got %d", len(devices))
	}
	if devices[0].Name != "CUDA1" {
		t.Errorf("expected CUDA1, got %s", devices[0].Name)
	}
}

func TestROCmDeviceGFXTarget(t *testing.T) {
	output := `ggml_cuda_init: found 1 ROCm devices (Total VRAM: 12272 MiB):
  Device 0: AMD Radeon RX 6700 XT, gfx1031 (0x1031), VMM: no, Wave Size: 32, VRAM: 12272 MiB
Available devices:
  ROCm0: AMD Radeon RX 6700 XT (12272 MiB, 12248 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama"})
	if len(devices) != 1 {
		t.Fatalf("expected 1 device, got %d", len(devices))
	}
	if devices[0].GFXTarget != "gfx1031" {
		t.Errorf("expected gfx1031, got %s", devices[0].GFXTarget)
	}
	if devices[0].Compute() != "gfx1031" {
		t.Errorf("expected compute gfx1031, got %s", devices[0].Compute())
	}
}

func TestROCmDeviceGFXTargetWithXnack(t *testing.T) {
	// gfx906 with :sramecc+:xnack- suffix (e.g., Radeon Pro VII)
	output := `ggml_cuda_init: found 2 ROCm devices (Total VRAM: 32736 MiB):
  Device 0: AMD Radeon RX 6800, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 16368 MiB
  Device 1: AMD Radeon Pro VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 16368 MiB
Available devices:
  ROCm0: AMD Radeon RX 6800 (16368 MiB, 16342 MiB free)
  ROCm1: AMD Radeon Pro VII (16368 MiB, 16348 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama"})
	if len(devices) != 2 {
		t.Fatalf("expected 2 devices, got %d", len(devices))
	}
	if devices[0].GFXTarget != "gfx1030" {
		t.Errorf("device 0: expected gfx1030, got %s", devices[0].GFXTarget)
	}
	if devices[1].GFXTarget != "gfx906" {
		t.Errorf("device 1: expected gfx906, got %s", devices[1].GFXTarget)
	}
	if devices[0].Compute() != "gfx1030" {
		t.Errorf("device 0: expected compute gfx1030, got %s", devices[0].Compute())
	}
	if devices[1].Compute() != "gfx906" {
		t.Errorf("device 1: expected compute gfx906, got %s", devices[1].Compute())
	}
}

func TestInferLibrary(t *testing.T) {
	tests := []struct {
		name string
		desc string
		want string
	}{
		{"NVIDIA CUDA", "NVIDIA GeForce RTX 4090", "CUDA"},
		{"CUDA0", "NVIDIA GeForce RTX 4090", "CUDA"},
		{"AMD ROCm", "AMD Radeon RX 6700 XT", "ROCm"},
		{"ROCm0", "AMD Radeon RX 6700 XT", "ROCm"},
		{"Metal", "Apple M3 Max", "Metal"},
		{"Vulkan0", "NVIDIA GeForce RTX 4090 (Vulkan)", "Vulkan"},
		{"Unknown", "Unknown Backend", "Unknown Backend"},
	}
	for _, tt := range tests {
		got := inferLibrary(tt.name, tt.desc)
		if got != tt.want {
			t.Errorf("inferLibrary(%q, %q) = %q, want %q", tt.name, tt.desc, got, tt.want)
		}
	}
}

func TestCudaCCRegex(t *testing.T) {
	tests := []struct {
		line    string
		wantIdx int
		wantCC  string
	}{
		{"  Device 0: NVIDIA GeForce GTX 1060 6GB, compute capability 6.1, VMM: yes, VRAM: 6063 MiB", 0, "610"},
		{"  Device 1: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB", 1, "890"},
		{"  Device 0: NVIDIA RTX PRO 6000, compute capability 12.0, VMM: yes, VRAM: 97250 MiB", 0, "1200"},
		{"  Device 0: Tesla V100-PCIE-16GB, compute capability 7.0, VMM: yes, VRAM: 16160 MiB", 0, "700"},
	}
	for _, tt := range tests {
		matches := cudaCCRegex.FindStringSubmatch(tt.line)
		if matches == nil {
			t.Errorf("expected match for %q", tt.line)
			continue
		}
		idx, _ := strconv.Atoi(matches[1])
		major, _ := strconv.Atoi(matches[2])
		minor, _ := strconv.Atoi(matches[3])
		cc := fmt.Sprintf("%d%d0", major, minor)
		if idx != tt.wantIdx {
			t.Errorf("for %q: got idx %d, want %d", tt.line, idx, tt.wantIdx)
		}
		if cc != tt.wantCC {
			t.Errorf("for %q: got CC %s, want %s", tt.line, cc, tt.wantCC)
		}
	}
}

func TestCudaArchsRegex(t *testing.T) {
	tests := []struct {
		line string
		want []string
	}{
		{
			"system_info: n_threads = 16 | CUDA : ARCHS = 750,800,860,890 | USE_GRAPHS = 1 |",
			[]string{"750", "800", "860", "890"},
		},
		{
			"system_info: | CUDA : ARCHS = 500,520,600,610,700,750,800,860,890,900,1200 |",
			[]string{"500", "520", "600", "610", "700", "750", "800", "860", "890", "900", "1200"},
		},
		{
			"no archs here",
			nil,
		},
	}
	for _, tt := range tests {
		matches := cudaArchsRegex.FindStringSubmatch(tt.line)
		if tt.want == nil {
			if matches != nil {
				t.Errorf("expected no match for %q, got %v", tt.line, matches)
			}
			continue
		}
		if matches == nil {
			t.Errorf("expected match for %q, got nil", tt.line)
			continue
		}
		got := strings.Split(matches[1], ",")
		if len(got) != len(tt.want) {
			t.Errorf("for %q: got %v, want %v", tt.line, got, tt.want)
		}
	}
}
