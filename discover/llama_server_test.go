package discover

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

func TestLlamaServerDiscovery(t *testing.T) {
	originalProbe := probeLlamaServerVulkanDevices
	probeLlamaServerVulkanDevices = func(_ []string) ([]vulkanPhysicalDevice, error) {
		return nil, errWindowsVulkanProbeUnsupported
	}
	t.Cleanup(func() {
		probeLlamaServerVulkanDevices = originalProbe
	})

	t.Run("output only trace", func(t *testing.T) {
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
	})

	t.Run("parse devices", func(t *testing.T) {
		type wantDevice struct {
			name            string
			library         string
			totalMiB        uint64
			compute         string
			driver          string
			gfxTarget       string
			checkIntegrated bool
			integrated      bool
		}

		tests := []struct {
			name    string
			output  string
			libDirs []string
			want    []wantDevice
		}{
			{
				name: "NVIDIA CUDA",
				output: `load_backend: loaded CUDA backend from /lib/ollama/cuda_v12/libggml-cuda.so
Available devices:
  NVIDIA GeForce RTX 4090: NVIDIA CUDA (24564 MiB, 23592 MiB free)
`,
				libDirs: []string{"/lib/ollama", "/lib/ollama/cuda_v12"},
				want: []wantDevice{{
					name:     "NVIDIA GeForce RTX 4090",
					library:  "CUDA",
					totalMiB: 24564,
					driver:   "12.0",
				}},
			},
			{
				name: "Metal",
				output: `Available devices:
  Metal: Apple M3 Max (98304 MiB, 98303 MiB free)
`,
				want: []wantDevice{{
					name:     "Metal",
					library:  "Metal",
					totalMiB: 98304,
				}},
			},
			{
				name: "ROCm with gfx target",
				output: `  Device 0: AMD Radeon RX 6700 XT, gfx1031 (0x1031), VMM: no, Wave Size: 32, VRAM: 12272 MiB
Available devices:
  ROCm0: AMD Radeon RX 6700 XT (12272 MiB, 12248 MiB free)
`,
				libDirs: []string{"/lib/ollama", "/lib/ollama/rocm_v7_2"},
				want: []wantDevice{{
					name:      "ROCm0",
					library:   "ROCm",
					totalMiB:  12272,
					compute:   "gfx1031",
					gfxTarget: "gfx1031",
				}},
			},
			{
				name: "multi GPU",
				output: `Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 23592 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (12288 MiB, 11500 MiB free)
`,
				libDirs: []string{"/lib/ollama", "/lib/ollama/cuda_v12"},
				want: []wantDevice{
					{name: "CUDA0", library: "CUDA", totalMiB: 24564},
					{name: "CUDA1", library: "CUDA", totalMiB: 12288},
				},
			},
			{
				name: "Vulkan UMA",
				output: `ggml_vulkan: 0 = Intel(R) Graphics (Intel open-source Mesa driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
Available devices:
  Vulkan0: Intel(R) Graphics (16384 MiB, 12288 MiB free)
`,
				libDirs: []string{"/lib/ollama", "/lib/ollama/vulkan"},
				want: []wantDevice{{
					name:            "Vulkan0",
					library:         "Vulkan",
					totalMiB:        16384,
					checkIntegrated: true,
					integrated:      true,
				}},
			},
			{
				name: "Vulkan without UMA metadata",
				output: `Available devices:
  Vulkan0: AMD Radeon(TM) Graphics (32768 MiB, 31000 MiB free)
`,
				libDirs: []string{"/lib/ollama", "/lib/ollama/vulkan"},
				want: []wantDevice{{
					name:            "Vulkan0",
					library:         "Vulkan",
					totalMiB:        32768,
					checkIntegrated: true,
				}},
			},
			{
				name: "CUDA device filtered by compiled archs",
				output: `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 6063 MiB):
  Device 0: NVIDIA GeForce GTX 1060 6GB, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
load_backend: loaded CUDA backend from /lib/ollama/cuda_v13/libggml-cuda.so
system_info: n_threads = 4 | CUDA : ARCHS = 750,800,860,890,900,1000,1030,1100,1200,1210 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`,
				libDirs: []string{"/lib/ollama", "/lib/ollama/cuda_v13"},
			},
			{
				name: "CUDA device kept by compiled archs",
				output: `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16379 MiB):
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB
system_info: n_threads = 16 | CUDA : ARCHS = 750,800,860,890,900,1000,1030,1100,1200,1210 |
Available devices:
  CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
`,
				want: []wantDevice{{
					name:     "CUDA0",
					library:  "CUDA",
					totalMiB: 16379,
					compute:  "8.9",
				}},
			},
			{
				name: "CUDA without compiled archs fails open",
				output: `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 6063 MiB):
  Device 0: NVIDIA GeForce GTX 1060 6GB, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`,
				want: []wantDevice{{
					name:     "CUDA0",
					library:  "CUDA",
					totalMiB: 6063,
					compute:  "6.1",
				}},
			},
			{
				name: "CUDA without compute capability fails open",
				output: `system_info: n_threads = 4 | CUDA : ARCHS = 750,800 |
Available devices:
  CUDA0: Some Future GPU (8192 MiB, 8000 MiB free)
`,
				want: []wantDevice{{
					name:     "CUDA0",
					library:  "CUDA",
					totalMiB: 8192,
				}},
			},
			{
				name: "CUDA mixed arch support",
				output: `ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce GTX 1060, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
  Device 1: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB
system_info: n_threads = 8 | CUDA : ARCHS = 750,800,860,890 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 (6063 MiB, 5900 MiB free)
  CUDA1: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
`,
				want: []wantDevice{{
					name:     "CUDA1",
					library:  "CUDA",
					totalMiB: 16379,
					compute:  "8.9",
				}},
			},
			{
				name: "ROCm gfx target with xnack suffix",
				output: `ggml_cuda_init: found 2 ROCm devices (Total VRAM: 32736 MiB):
  Device 0: AMD Radeon RX 6800, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 16368 MiB
  Device 1: AMD Radeon Pro VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 16368 MiB
Available devices:
  ROCm0: AMD Radeon RX 6800 (16368 MiB, 16342 MiB free)
  ROCm1: AMD Radeon Pro VII (16368 MiB, 16348 MiB free)
`,
				want: []wantDevice{
					{name: "ROCm0", library: "ROCm", totalMiB: 16368, compute: "gfx1030", gfxTarget: "gfx1030"},
					{name: "ROCm1", library: "ROCm", totalMiB: 16368, compute: "gfx906", gfxTarget: "gfx906"},
				},
			},
			{
				name: "unknown library",
				output: `Available devices:
  Future0: Mystery Accelerator (8192 MiB, 8000 MiB free)
`,
				want: []wantDevice{{
					name:     "Future0",
					library:  "Mystery Accelerator",
					totalMiB: 8192,
				}},
			},
			{
				name:   "no devices",
				output: "Available devices:\n",
			},
			{
				name: "empty output",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if tt.libDirs == nil {
					tt.libDirs = []string{"/lib/ollama"}
				}
				devices := parseLlamaServerDevices(tt.output, tt.libDirs)
				if len(devices) != len(tt.want) {
					t.Fatalf("got %d devices, want %d", len(devices), len(tt.want))
				}
				for i, want := range tt.want {
					got := devices[i]
					if want.name != "" && got.Name != want.name {
						t.Errorf("device %d name = %q, want %q", i, got.Name, want.name)
					}
					if want.library != "" && got.Library != want.library {
						t.Errorf("device %d library = %q, want %q", i, got.Library, want.library)
					}
					if want.totalMiB > 0 && got.TotalMemory != want.totalMiB*1024*1024 {
						t.Errorf("device %d total memory = %d, want %d MiB", i, got.TotalMemory, want.totalMiB)
					}
					if want.compute != "" && got.Compute() != want.compute {
						t.Errorf("device %d compute = %q, want %q", i, got.Compute(), want.compute)
					}
					if want.driver != "" && got.Driver() != want.driver {
						t.Errorf("device %d driver = %q, want %q", i, got.Driver(), want.driver)
					}
					if want.gfxTarget != "" && got.GFXTarget != want.gfxTarget {
						t.Errorf("device %d gfx target = %q, want %q", i, got.GFXTarget, want.gfxTarget)
					}
					if want.checkIntegrated && got.Integrated != want.integrated {
						t.Errorf("device %d integrated = %v, want %v", i, got.Integrated, want.integrated)
					}
				}
			})
		}
	})

	t.Run("parse fixtures", func(t *testing.T) {
		type wantDevice struct {
			name       string
			library    string
			totalMiB   uint64
			compute    string
			gfxTarget  string
			integrated bool
		}

		tests := []struct {
			name     string
			output   string
			libDirs  []string
			want     []wantDevice
			wantSkip string
		}{
			{
				name: "cuda mixed archs filters unsupported device",
				output: `ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce GTX 1060, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
  Device 1: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB
system_info: n_threads = 8 | CUDA : ARCHS = 750,800,860,890 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 (6063 MiB, 5900 MiB free)
  CUDA1: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
`,
				want: []wantDevice{{
					name:     "CUDA1",
					library:  "CUDA",
					totalMiB: 16379,
					compute:  "8.9",
				}},
			},
			{
				name: "rocm gfx targets preserve suffix-free compute",
				output: `ggml_cuda_init: found 2 ROCm devices (Total VRAM: 32736 MiB):
  Device 0: AMD Radeon RX 6800, gfx1030 (0x1030), VMM: no, Wave Size: 32, VRAM: 16368 MiB
  Device 1: AMD Radeon Pro VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 16368 MiB
Available devices:
  ROCm0: AMD Radeon RX 6800 (16368 MiB, 16342 MiB free)
  ROCm1: AMD Radeon Pro VII (16368 MiB, 16348 MiB free)
`,
				want: []wantDevice{
					{name: "ROCm0", library: "ROCm", totalMiB: 16368, compute: "gfx1030", gfxTarget: "gfx1030"},
					{name: "ROCm1", library: "ROCm", totalMiB: 16368, compute: "gfx906", gfxTarget: "gfx906"},
				},
			},
			{
				name: "vulkan uma marks integrated",
				output: `ggml_vulkan: 0 = Intel(R) Graphics (Intel open-source Mesa driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 65536 | int dot: 1 | matrix cores: none
Available devices:
  Vulkan0: Intel(R) Graphics (16384 MiB, 12288 MiB free)
`,
				want: []wantDevice{{
					name:       "Vulkan0",
					library:    "Vulkan",
					totalMiB:   16384,
					integrated: true,
				}},
			},
			{
				name: "windows vulkan without uma stays unclassified",
				output: `load_backend: loaded Vulkan backend from C:\ollama\lib\ollama\vulkan\ggml-vulkan.dll
Available devices:
  Vulkan0: AMD Radeon(TM) Graphics (32768 MiB, 31000 MiB free)
  Vulkan1: AMD Radeon RX 7900 XTX (24564 MiB, 23000 MiB free)
`,
				want: []wantDevice{
					{name: "Vulkan0", library: "Vulkan", totalMiB: 32768},
					{name: "Vulkan1", library: "Vulkan", totalMiB: 24564},
				},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				libDirs := tt.libDirs
				if libDirs == nil {
					libDirs = []string{"/lib/ollama"}
				}

				got := parseLlamaServerDevices(tt.output, libDirs)
				if len(got) != len(tt.want) {
					t.Fatalf("got %d devices, want %d", len(got), len(tt.want))
				}
				for i, want := range tt.want {
					if got[i].Name != want.name {
						t.Fatalf("device %d name = %q, want %q", i, got[i].Name, want.name)
					}
					if got[i].Library != want.library {
						t.Fatalf("device %d library = %q, want %q", i, got[i].Library, want.library)
					}
					if got[i].TotalMemory != want.totalMiB*1024*1024 {
						t.Fatalf("device %d total memory = %d, want %d MiB", i, got[i].TotalMemory, want.totalMiB)
					}
					if want.compute != "" && got[i].Compute() != want.compute {
						t.Fatalf("device %d compute = %q, want %q", i, got[i].Compute(), want.compute)
					}
					if want.gfxTarget != "" && got[i].GFXTarget != want.gfxTarget {
						t.Fatalf("device %d gfx target = %q, want %q", i, got[i].GFXTarget, want.gfxTarget)
					}
					if got[i].Integrated != want.integrated {
						t.Fatalf("device %d integrated = %v, want %v", i, got[i].Integrated, want.integrated)
					}
				}
			})
		}
	})

	t.Run("skips mismatched Vulkan native metadata", func(t *testing.T) {
		output := `Available devices:
  Vulkan0: Intel(R) UHD Graphics 770 (32768 MiB, 31000 MiB free)
`
		nativeDevices := []nativeProbeDevice{{
			Library:             "Vulkan",
			Index:               0,
			IndexMatchesBackend: true,
			Description:         "NVIDIA GeForce RTX 4060 Ti",
			DeviceID:            "0000:05:00.0",
			IntegratedKnown:     true,
			TotalMemory:         16107 * 1024 * 1024,
		}}

		devices := parseLlamaServerDevicesWithNative(output, []string{"/lib/ollama", "/lib/ollama/vulkan"}, nativeDevices)
		if len(devices) != 1 {
			t.Fatalf("got %d devices, want 1", len(devices))
		}
		if devices[0].PCIID != "" {
			t.Fatalf("PCIID = %q, want empty", devices[0].PCIID)
		}
		if devices[0].Integrated {
			t.Fatal("Integrated = true, want false")
		}
	})

	t.Run("cuda runtime version", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "libcudart.so.12.8.90"), nil, 0o644); err != nil {
			t.Fatal(err)
		}

		major, minor, ok := cudaRuntimeVersion([]string{dir})
		if !ok || major != 12 || minor != 8 {
			t.Fatalf("cudaRuntimeVersion = %d.%d, %v, want 12.8, true", major, minor, ok)
		}

		major, minor, ok = cudaRuntimeVersion([]string{filepath.Join(t.TempDir(), "cuda_v13")})
		if !ok || major != 13 || minor != 0 {
			t.Fatalf("cudaRuntimeVersion fallback = %d.%d, %v, want 13.0, true", major, minor, ok)
		}
	})

	t.Run("refine windows vulkan devices", func(t *testing.T) {
		makeDevices := func() []ml.DeviceInfo {
			return []ml.DeviceInfo{
				{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}, Description: "AMD Radeon(TM) Graphics"},
				{DeviceID: ml.DeviceID{ID: "1", Library: "Vulkan"}, Description: "AMD Radeon RX 7900 XTX"},
				{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Description: "NVIDIA GeForce RTX 4090"},
			}
		}

		tests := []struct {
			name    string
			devices []ml.DeviceInfo
			probed  []vulkanPhysicalDevice
			want    []bool
			applied bool
		}{
			{
				name: "fills missing integrated bit",
				probed: []vulkanPhysicalDevice{
					{Name: "AMD Radeon(TM) Graphics", Integrated: true},
					{Name: "AMD Radeon RX 7900 XTX", Integrated: false},
				},
				want:    []bool{true, false, false},
				applied: true,
			},
			{
				name: "matches names when order differs",
				probed: []vulkanPhysicalDevice{
					{Name: "AMD Radeon RX 7900 XTX", Integrated: false},
					{Name: "AMD Radeon(TM) Graphics", Integrated: true},
				},
				want:    []bool{true, false, false},
				applied: true,
			},
			{
				name: "skips when names do not line up",
				probed: []vulkanPhysicalDevice{
					{Name: "Wrong GPU", Integrated: true},
					{Name: "AMD Radeon RX 7900 XTX", Integrated: false},
				},
				want: []bool{false, false, false},
			},
			{
				name:   "skips when counts do not line up",
				probed: []vulkanPhysicalDevice{{Name: "AMD Radeon(TM) Graphics", Integrated: true}},
				want:   []bool{false, false, false},
			},
			{
				name: "overwrites stale classification",
				devices: []ml.DeviceInfo{
					{DeviceID: ml.DeviceID{ID: "0", Library: "Vulkan"}, Description: "AMD Radeon(TM) Graphics", Integrated: true},
					{DeviceID: ml.DeviceID{ID: "1", Library: "Vulkan"}, Description: "AMD Radeon RX 7900 XTX"},
				},
				probed: []vulkanPhysicalDevice{
					{Name: "AMD Radeon(TM) Graphics", Integrated: false},
					{Name: "AMD Radeon RX 7900 XTX", Integrated: false},
				},
				want:    []bool{false, false},
				applied: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				devices := tt.devices
				if devices == nil {
					devices = makeDevices()
				}
				applied := applyWindowsVulkanRefinement(devices, tt.probed)
				if applied != tt.applied {
					t.Fatalf("applied = %v, want %v", applied, tt.applied)
				}
				got := devices
				if len(got) != len(tt.want) {
					t.Fatalf("got %d devices, want %d", len(got), len(tt.want))
				}
				for i, want := range tt.want {
					if got[i].Integrated != want {
						t.Fatalf("device %d integrated = %v, want %v", i, got[i].Integrated, want)
					}
				}
			})
		}
	})
}
