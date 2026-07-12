package discover

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// Native GPU discovery runs in a short-lived Ollama subprocess so loading GGML
// and driver libraries cannot crash the main server process. The subprocess
// keeps stdout reserved for JSON and lets GGML's default logger write to
// stderr; the parent captures that stderr for trace/debug diagnostics.
const nativeProbeTimeout = 15 * time.Second

type nativeProbeDevice struct {
	Library string `json:"library"`
	Index   int    `json:"index"`
	// IndexMatchesBackend means Index is in the same visible-device order that
	// llama-server reports, so it is safe to correlate when PCI ID is missing.
	IndexMatchesBackend bool   `json:"index_matches_backend,omitempty"`
	Name                string `json:"name,omitempty"`
	Description         string `json:"description,omitempty"`
	DeviceID            string `json:"device_id,omitempty"`
	Integrated          bool   `json:"integrated,omitempty"`
	IntegratedKnown     bool   `json:"integrated_known"`
	TotalMemory         uint64 `json:"total_memory,omitempty"`
	FreeMemory          uint64 `json:"free_memory,omitempty"`
	ComputeMajor        int    `json:"compute_major,omitempty"`
	ComputeMinor        int    `json:"compute_minor,omitempty"`
	CUDADriverMajor     int    `json:"cuda_driver_major,omitempty"`
	CUDADriverMinor     int    `json:"cuda_driver_minor,omitempty"`
	NVIDIADriverMajor   int    `json:"nvidia_driver_major,omitempty"`
	GFXTarget           string `json:"gfx_target,omitempty"`
}

type nativeProbeResult struct {
	Devices []nativeProbeDevice `json:"devices"`
}

type ggmlBackendDevCaps struct {
	Async             uint8
	HostBuffer        uint8
	BufferFromHostPtr uint8
	Events            uint8
}

type ggmlBackendDevProps struct {
	Name        uintptr
	Description uintptr
	MemoryFree  uintptr
	MemoryTotal uintptr
	Type        int32
	_           [4]byte
	DeviceID    uintptr
	Caps        ggmlBackendDevCaps
	_           [4]byte
}

func discoverNativeDevices(ctx context.Context, llamaServer string, libDirs []string, extraEnvs map[string]string) ([]nativeProbeDevice, string, error) {
	if runtime.GOOS != "linux" && runtime.GOOS != "windows" {
		return nil, "", nil
	}

	exe, err := os.Executable()
	if err != nil {
		return nil, "", err
	}

	ctx, cancel := context.WithTimeout(ctx, nativeProbeTimeout)
	defer cancel()

	args := []string{"gpu-discover"}
	for _, dir := range libDirs {
		args = append(args, "--lib-dir", dir)
	}
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.WaitDelay = llamaServerDiscoveryWaitDelay
	llm.SetupLlamaServerCommandEnv(cmd, llamaServer, libDirs, extraEnvs)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return nil, stderr.String(), ctx.Err()
		}
		return nil, stderr.String(), err
	}

	var result nativeProbeResult
	if err := json.Unmarshal(stdout, &result); err != nil {
		return nil, stderr.String(), err
	}

	return result.Devices, stderr.String(), nil
}

func RunNativeProbeCommand(ctx context.Context, libDirs []string, out io.Writer) error {
	if len(libDirs) == 0 {
		libDirs = []string{ml.LibOllamaPath}
	}

	devices, err := runNativeProbe(ctx, libDirs)
	if err != nil {
		return err
	}

	return json.NewEncoder(out).Encode(nativeProbeResult{Devices: devices})
}

func runNativeProbe(ctx context.Context, libDirs []string) ([]nativeProbeDevice, error) {
	return runPlatformNativeProbe(ctx, libDirs)
}

func mergeNativeProbeDevices(base, supplement []nativeProbeDevice) []nativeProbeDevice {
	if len(base) == 0 {
		var out []nativeProbeDevice
		for _, extra := range supplement {
			if extra.IndexMatchesBackend {
				out = append(out, extra)
			}
		}
		return out
	}

	out := append([]nativeProbeDevice(nil), base...)
	for _, extra := range supplement {
		idx := -1
		for i := range out {
			if sameNativeProbeDevice(out[i], extra) {
				idx = i
				break
			}
		}
		if idx < 0 {
			if !extra.IndexMatchesBackend || nativeProbeLibraryIndexExists(out, extra) {
				continue
			}
			out = append(out, extra)
			continue
		}
		mergeNativeProbeDevice(&out[idx], extra)
	}
	return out
}

func sameNativeProbeDevice(a, b nativeProbeDevice) bool {
	if !strings.EqualFold(a.Library, b.Library) {
		return false
	}
	if a.DeviceID != "" && b.DeviceID != "" {
		return strings.EqualFold(a.DeviceID, b.DeviceID)
	}
	if !a.IndexMatchesBackend || !b.IndexMatchesBackend {
		return false
	}
	return a.Index == b.Index
}

func mergeNativeProbeDevice(dst *nativeProbeDevice, src nativeProbeDevice) {
	dst.IndexMatchesBackend = dst.IndexMatchesBackend || src.IndexMatchesBackend
	if dst.Name == "" {
		dst.Name = src.Name
	}
	if dst.Description == "" {
		dst.Description = src.Description
	}
	if dst.DeviceID == "" {
		dst.DeviceID = src.DeviceID
	}
	if src.IntegratedKnown {
		dst.Integrated = src.Integrated
		dst.IntegratedKnown = true
	} else if !dst.IntegratedKnown && src.Integrated {
		dst.Integrated = true
	}
	if dst.TotalMemory == 0 {
		dst.TotalMemory = src.TotalMemory
	}
	if dst.FreeMemory == 0 {
		dst.FreeMemory = src.FreeMemory
	}
	if dst.ComputeMajor == 0 && src.ComputeMajor != 0 {
		dst.ComputeMajor = src.ComputeMajor
		dst.ComputeMinor = src.ComputeMinor
	}
	if dst.CUDADriverMajor == 0 && src.CUDADriverMajor != 0 {
		dst.CUDADriverMajor = src.CUDADriverMajor
		dst.CUDADriverMinor = src.CUDADriverMinor
	}
	if dst.NVIDIADriverMajor == 0 && src.NVIDIADriverMajor != 0 {
		dst.NVIDIADriverMajor = src.NVIDIADriverMajor
	}
	if dst.GFXTarget == "" {
		dst.GFXTarget = src.GFXTarget
	}
}

func nativeProbeLibraryIndexExists(devices []nativeProbeDevice, target nativeProbeDevice) bool {
	if !target.IndexMatchesBackend {
		return false
	}
	for _, dev := range devices {
		if strings.EqualFold(dev.Library, target.Library) && dev.Index == target.Index {
			return true
		}
	}
	return false
}

func nativeProbeByLibraryIndex(devices []nativeProbeDevice) map[string]map[int]nativeProbeDevice {
	out := map[string]map[int]nativeProbeDevice{}
	for _, dev := range devices {
		if !dev.IndexMatchesBackend {
			continue
		}
		lib := normalizeNativeProbeLibrary(dev.Library)
		if lib == "" {
			continue
		}
		if _, ok := out[lib]; !ok {
			out[lib] = map[int]nativeProbeDevice{}
		}
		out[lib][dev.Index] = dev
	}
	return out
}

func normalizeNativeProbeLibrary(library string) string {
	switch strings.ToLower(library) {
	case "cuda":
		return "CUDA"
	case "hip", "rocm":
		return "ROCm"
	case "vulkan":
		return "Vulkan"
	case "metal":
		return "Metal"
	default:
		return library
	}
}

func logNativeProbeFailure(err error, stderr string, libDirs []string) {
	if err == nil {
		return
	}
	if stderr != "" {
		slog.Debug("native GPU discovery failed", "error", err, "stderr", stderr, "libDirs", libDirs)
		return
	}
	slog.Debug("native GPU discovery failed", "error", err, "libDirs", libDirs)
}
