package discover

import (
	"testing"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

func TestGGMLBackendDevPropsLayout(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("GGML probe layout assertions are for 64-bit builds")
	}

	var props ggmlBackendDevProps
	if got, want := unsafe.Sizeof(props), uintptr(56); got != want {
		t.Fatalf("ggmlBackendDevProps size = %d, want %d", got, want)
	}
	checks := []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"Name", unsafe.Offsetof(props.Name), 0},
		{"Description", unsafe.Offsetof(props.Description), 8},
		{"MemoryFree", unsafe.Offsetof(props.MemoryFree), 16},
		{"MemoryTotal", unsafe.Offsetof(props.MemoryTotal), 24},
		{"Type", unsafe.Offsetof(props.Type), 32},
		{"DeviceID", unsafe.Offsetof(props.DeviceID), 40},
		{"Caps", unsafe.Offsetof(props.Caps), 48},
	}
	for _, tt := range checks {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Fatalf("offset = %d, want %d", tt.got, tt.want)
			}
		})
	}
	if got, want := unsafe.Sizeof(ggmlBackendDevCaps{}), uintptr(4); got != want {
		t.Fatalf("ggmlBackendDevCaps size = %d, want %d", got, want)
	}
}

func TestParseLlamaServerDevicesUsesNativeCUDAComputeCapability(t *testing.T) {
	output := `system_info: n_threads = 4 | CUDA : ARCHS = 750,800 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`
	devices := parseLlamaServerDevicesWithNative(output, []string{"/lib/ollama", "/lib/ollama/cuda_v13"}, []nativeProbeDevice{{
		Library:             "CUDA",
		Index:               0,
		IndexMatchesBackend: true,
		DeviceID:            "0000:01:00.0",
		ComputeMajor:        6,
		ComputeMinor:        1,
		CUDADriverMajor:     13,
		NVIDIADriverMajor:   570,
	}})
	if len(devices) != 0 {
		t.Fatalf("got %d devices, want unsupported CUDA device filtered", len(devices))
	}

	output = `system_info: n_threads = 4 | CUDA : ARCHS = 610,750,800 |
Available devices:
  CUDA0: NVIDIA GeForce GTX 1060 6GB (6063 MiB, 5900 MiB free)
`
	devices = parseLlamaServerDevicesWithNative(output, []string{"/lib/ollama", "/lib/ollama/cuda_v12"}, []nativeProbeDevice{{
		Library:             "CUDA",
		Index:               0,
		IndexMatchesBackend: true,
		DeviceID:            "0000:01:00.0",
		ComputeMajor:        6,
		ComputeMinor:        1,
		CUDADriverMajor:     12,
		NVIDIADriverMajor:   570,
	}})
	if len(devices) != 1 {
		t.Fatalf("got %d devices, want 1", len(devices))
	}
	got := devices[0]
	if got.Compute() != "6.1" {
		t.Fatalf("compute = %q, want 6.1", got.Compute())
	}
	if got.PCIID != "0000:01:00.0" {
		t.Fatalf("PCIID = %q, want 0000:01:00.0", got.PCIID)
	}
	if got.Driver() != "12.0" {
		t.Fatalf("driver = %q, want 12.0", got.Driver())
	}
	if got.NVIDIADriverMajor != 570 {
		t.Fatalf("NVIDIADriverMajor = %d, want 570", got.NVIDIADriverMajor)
	}
}

func TestParseLlamaServerDevicesUsesNativeROCmMetadata(t *testing.T) {
	output := `ggml_vulkan: 0 = AMD Radeon RX 7600 | uma: 1 | fp16: 1 |
Available devices:
  ROCm0: AMD Radeon RX 7600 (8176 MiB, 7900 MiB free)
`
	devices := parseLlamaServerDevicesWithNative(output, []string{"/lib/ollama", "/lib/ollama/rocm_v7_2"}, []nativeProbeDevice{{
		Library:             "ROCm",
		Index:               0,
		IndexMatchesBackend: true,
		DeviceID:            "0000:03:00.0",
		GFXTarget:           "gfx1102",
		Integrated:          false,
		IntegratedKnown:     true,
	}})
	if len(devices) != 1 {
		t.Fatalf("got %d devices, want 1", len(devices))
	}
	got := devices[0]
	if got.PCIID != "0000:03:00.0" {
		t.Fatalf("PCIID = %q, want 0000:03:00.0", got.PCIID)
	}
	if got.GFXTarget != "gfx1102" {
		t.Fatalf("GFXTarget = %q, want gfx1102", got.GFXTarget)
	}
	if got.Compute() != "gfx1102" {
		t.Fatalf("compute = %q, want gfx1102", got.Compute())
	}
	if got.Integrated {
		t.Fatalf("Integrated = true, want false")
	}
}

func TestNVIDIADriverMajorFromDevices(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "CUDA"}, NVIDIADriverMajor: 565},
	}
	if got := nvidiaDriverMajorFromDevices(devices); got != 565 {
		t.Fatalf("driver = %d, want 565", got)
	}
}

func TestMergeNativeProbeDevicesAvoidsUnreliableIndexMatch(t *testing.T) {
	tests := []struct {
		name       string
		base       []nativeProbeDevice
		supplement []nativeProbeDevice
		wantLen    int
		wantPCI    string
		wantKnown  bool
		wantIGPU   bool
	}{
		{
			name: "filtered sysfs cannot overwrite a different backend device by index",
			base: []nativeProbeDevice{{
				Library:             "ROCm",
				Index:               0,
				IndexMatchesBackend: true,
				DeviceID:            "0000:03:00.0",
			}},
			supplement: []nativeProbeDevice{{
				Library:         "ROCm",
				Index:           0,
				DeviceID:        "0000:04:00.0",
				Integrated:      true,
				IntegratedKnown: true,
			}},
			wantLen: 1,
			wantPCI: "0000:03:00.0",
		},
		{
			name: "filtered sysfs can still merge by PCI ID",
			base: []nativeProbeDevice{{
				Library:             "ROCm",
				Index:               0,
				IndexMatchesBackend: true,
				DeviceID:            "0000:04:00.0",
			}},
			supplement: []nativeProbeDevice{{
				Library:         "ROCm",
				Index:           1,
				DeviceID:        "0000:04:00.0",
				Integrated:      true,
				IntegratedKnown: true,
			}},
			wantLen:   1,
			wantPCI:   "0000:04:00.0",
			wantKnown: true,
			wantIGPU:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeNativeProbeDevices(tt.base, tt.supplement)
			if len(got) != tt.wantLen {
				t.Fatalf("got %d devices, want %d: %#v", len(got), tt.wantLen, got)
			}
			if got[0].DeviceID != tt.wantPCI {
				t.Fatalf("DeviceID = %q, want %q", got[0].DeviceID, tt.wantPCI)
			}
			if got[0].IntegratedKnown != tt.wantKnown {
				t.Fatalf("IntegratedKnown = %v, want %v", got[0].IntegratedKnown, tt.wantKnown)
			}
			if got[0].Integrated != tt.wantIGPU {
				t.Fatalf("Integrated = %v, want %v", got[0].Integrated, tt.wantIGPU)
			}
		})
	}
}
