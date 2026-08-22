package ml

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

// Minimal unique device identification
type DeviceID struct {
	// ID is an identifier for the device for matching with system
	// management libraries.  The ID is only unique for other devices
	// using the same Library.
	// This ID represents a "post filtered" view of the enumerated devices
	// if the ID is numeric
	ID string `json:"id"`

	// Library identifies which library is used for the device (e.g. CUDA, ROCm, etc.)
	Library string `json:"backend,omitempty"`
}

type DeviceInfo struct {
	DeviceID

	// Name is the name of the device as labeled by the backend. It
	// may not be persistent across instances of the runner.
	Name string `json:"name"`

	// Description is the longer user-friendly identification of the device
	Description string `json:"description"`

	// FilterID is populated with the unfiltered device ID if a numeric ID is used
	// so the device can be included.
	FilterID string `json:"filter_id,omitempty"`

	// Integrated is set true for integrated GPUs, false for Discrete GPUs
	Integrated bool `json:"integration,omitempty"`

	// PCIID is the bus, device and domain ID of the device for deduplication
	// when discovered by multiple backends
	PCIID string `json:"pci_id,omitempty"`

	// TotalMemory is the total amount of memory the device can use for loading models
	TotalMemory uint64 `json:"total_memory"`

	// FreeMemory is the amount of memory currently available on the device for loading models
	FreeMemory uint64 `json:"free_memory,omitempty"`

	// ComputeMajor is the major version of capabilities of the device
	// if unsupported by the backend, -1 will be returned
	ComputeMajor int

	// ComputeMinor is the minor version of capabilities of the device
	// if unsupported by the backend, -1 will be returned
	ComputeMinor int

	// Driver Information
	DriverMajor int `json:"driver_major,omitempty"`
	DriverMinor int `json:"driver_minor,omitempty"`

	// NVIDIADriverMajor is the NVIDIA kernel driver branch. CUDA driver APIs
	// expose a separate CUDA compatibility version, so keep this distinct.
	NVIDIADriverMajor int `json:"-"`

	// GFXTarget is the AMD GPU gfx target string (e.g. "gfx1100") for ROCm
	// device validation. Empty on non-AMD devices.
	GFXTarget string `json:"gfx_target,omitempty"`

	// Where backends were loaded from
	LibraryPath []string

	// RunnerEnvOverrides stores exceptional per-device runner environment
	// overrides discovered during bootstrap. This is internal server state and
	// is not serialized.
	RunnerEnvOverrides map[string]string `json:"-"`
}

type SystemInfo struct {
	// TotalMemory is the total amount of system memory
	TotalMemory uint64 `json:"total_memory,omitempty"`

	// FreeMemory is the amount of memory currently available on the system for loading models
	FreeMemory uint64 `json:"free_memory,omitempty"`

	// FreeSwap is the amount of system swap space reported as available
	FreeSwap uint64 `json:"free_swap,omitempty"`
}

func (d DeviceInfo) Compute() string {
	// AMD gfx is encoded into the major minor in hex form
	if strings.EqualFold(d.Library, "ROCm") {
		return fmt.Sprintf("gfx%x%02x", d.ComputeMajor, d.ComputeMinor)
	}
	return strconv.Itoa(d.ComputeMajor) + "." + strconv.Itoa(d.ComputeMinor)
}

func (d DeviceInfo) Driver() string {
	return strconv.Itoa(d.DriverMajor) + "." + strconv.Itoa(d.DriverMinor)
}

// MinimumMemory reports the amount of memory that should be set aside
// on the device for overhead (e.g. VRAM consumed by context structures independent
// of model allocations)
func (d DeviceInfo) MinimumMemory() uint64 {
	if d.Library == "Metal" {
		return 512 * format.MebiByte
	}
	return 457 * format.MebiByte
}

// Sort by Free Space.
// iGPUs are reported first, thus Reverse() yields the largest discrete GPU first
type ByFreeMemory []DeviceInfo

func (a ByFreeMemory) Len() int      { return len(a) }
func (a ByFreeMemory) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByFreeMemory) Less(i, j int) bool {
	if a[i].Integrated && !a[j].Integrated {
		return true
	} else if !a[i].Integrated && a[j].Integrated {
		return false
	}
	return a[i].FreeMemory < a[j].FreeMemory
}

func ByLibrary(l []DeviceInfo) [][]DeviceInfo {
	resp := [][]DeviceInfo{}
	libs := []string{}
	for _, info := range l {
		found := false
		requested := info.Library
		for i, lib := range libs {
			if lib == requested {
				resp[i] = append(resp[i], info)
				found = true
				break
			}
		}
		if !found {
			libs = append(libs, requested)
			resp = append(resp, []DeviceInfo{info})
		}
	}
	return resp
}

func LibraryPaths(l []DeviceInfo) []string {
	gpuLibs := []string{LibOllamaPath}
	for _, gpu := range l {
		for _, dir := range gpu.LibraryPath {
			needed := true
			for _, existing := range gpuLibs {
				if dir == existing {
					needed = false
					break
				}
			}
			if needed {
				gpuLibs = append(gpuLibs, dir)
			}
		}
	}
	return gpuLibs
}

type DeviceComparison int

const (
	UniqueDevice      DeviceComparison = iota
	SameBackendDevice                  // The device is the same, and the library/backend is the same
	DuplicateDevice                    // The same physical device but different library/backend (overlapping device)
)

func (a DeviceInfo) Compare(b DeviceInfo) DeviceComparison {
	if a.PCIID != "" && b.PCIID != "" {
		if !strings.EqualFold(a.PCIID, b.PCIID) {
			return UniqueDevice
		}
		if a.Library == b.Library {
			return SameBackendDevice
		}
		return DuplicateDevice
	}
	if likelyVulkanDuplicate(a, b) {
		return DuplicateDevice
	}
	if a.PCIID != b.PCIID {
		return UniqueDevice
	}
	// If PCIID is empty, we have to use ID + library for uniqueness
	if a.PCIID == "" && a.DeviceID != b.DeviceID {
		return UniqueDevice
	}
	if a.Library == b.Library {
		return SameBackendDevice
	}
	return DuplicateDevice
}

func likelyVulkanDuplicate(a, b DeviceInfo) bool {
	if a.Library == b.Library {
		return false
	}
	vulkan, other := a, b
	if b.Library == "Vulkan" {
		vulkan, other = b, a
	}
	if vulkan.Library != "Vulkan" {
		return false
	}
	if other.Library != "CUDA" && other.Library != "ROCm" {
		return false
	}
	if normalizeDeviceDescription(vulkan.Description) == "" {
		return false
	}
	if !SimilarDeviceDescription(vulkan.Description, other.Description) {
		return false
	}
	return SimilarDeviceMemory(vulkan.TotalMemory, other.TotalMemory)
}

// SimilarDeviceDescription reports whether two backend device descriptions are
// close enough to identify the same physical GPU across different libraries.
func SimilarDeviceDescription(a, b string) bool {
	normalizedA := normalizeDeviceDescription(a)
	return normalizedA != "" && normalizedA == normalizeDeviceDescription(b)
}

func normalizeDeviceDescription(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	depth := 0
	for _, r := range s {
		switch {
		case r == '(':
			depth++
			continue
		case r == ')':
			if depth > 0 {
				depth--
				continue
			}
		case depth > 0:
			continue
		case r >= 'a' && r <= 'z' || r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

func SimilarDeviceMemory(a, b uint64) bool {
	if a == 0 || b == 0 {
		return false
	}
	maxMemory := max(a, b)
	tolerance := maxMemory / 20
	if tolerance < 512*1024*1024 {
		tolerance = 512 * 1024 * 1024
	}
	return maxMemory-min(a, b) <= tolerance
}

// For a SameBackendDevice, return true if b is better than a
// e.g. newer GPU library version
func (a DeviceInfo) IsBetter(b DeviceInfo) bool {
	aLib := a.LibraryPath[len(a.LibraryPath)-1]
	bLib := b.LibraryPath[len(b.LibraryPath)-1]
	if aLib == bLib {
		return false
	}
	aLibSplit := strings.SplitN(aLib, "_", 2)
	bLibSplit := strings.SplitN(bLib, "_", 2)
	if len(aLibSplit) < 2 || len(bLibSplit) < 2 {
		return false
	}
	if aLibSplit[0] != bLibSplit[0] {
		slog.Debug("unexpected libraries", "a", aLib, "b", bLib)
		return false
	}
	if aLibSplit[1] == bLibSplit[1] {
		return false
	}
	cmp := []string{aLibSplit[1], bLibSplit[1]}
	sort.Sort(sort.Reverse(sort.StringSlice(cmp)))
	return cmp[0] == bLibSplit[1]
}

// FlashAttentionSupported reports whether flash attention can be used across
// all selected devices.
func FlashAttentionSupported(l []DeviceInfo) bool {
	for _, gpu := range l {
		supportsFA := gpu.Library == "cpu" ||
			gpu.Name == "Metal" || gpu.Library == "Metal" ||
			cudaFlashAttentionSupported(gpu) ||
			gpu.Library == "ROCm" ||
			gpu.Library == "Vulkan"

		if !supportsFA {
			return false
		}
	}
	return true
}

func cudaFlashAttentionSupported(gpu DeviceInfo) bool {
	if gpu.Library != "CUDA" ||
		gpu.ComputeMajor < 6 ||
		(gpu.ComputeMajor == 7 && gpu.ComputeMinor == 2) {
		return false
	}

	if gpu.DriverMajor == 0 {
		slog.Warn("CUDA driver version unavailable; allowing flash attention based on compute capability",
			"device", gpu.Description, "compute", gpu.Compute())
		return true
	}

	return gpu.DriverMajor >= 7
}

type FlashAttentionType int32

const (
	// Aligned with llama_flash_attn_type
	FlashAttentionAuto     FlashAttentionType = -1
	FlashAttentionDisabled FlashAttentionType = 0
	FlashAttentionEnabled  FlashAttentionType = 1
)

func (f FlashAttentionType) LogValue() slog.Value {
	return slog.AnyValue(f.String())
}

func (f FlashAttentionType) String() string {
	switch f {
	case FlashAttentionAuto:
		return "Auto"
	case FlashAttentionDisabled:
		return "Disabled"
	case FlashAttentionEnabled:
		return "Enabled"
	default:
		return "unknown"
	}
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the device environment variables and any recorded
// per-device runner environment overrides.
func GetDevicesEnv(l []DeviceInfo) map[string]string {
	if len(l) == 0 {
		return nil
	}
	// CUDA-only groups need filtering so devices removed during discovery do
	// not reappear in the child process.
	mustFilter := len(l) == 1 || allDevicesUseLibrary(l, "CUDA")
	env := map[string]string{}
	for _, d := range l {
		d.updateVisibleDevicesEnv(env, mustFilter)
		for k, v := range d.RunnerEnvOverrides {
			if existing, ok := env[k]; ok && existing != v {
				slog.Warn("conflicting device environment override", "key", k, "existing", existing, "new", v, "library", d.Library, "id", d.ID)
			}
			env[k] = v
		}
	}

	return env
}

func allDevicesUseLibrary(l []DeviceInfo, library string) bool {
	for _, d := range l {
		if d.Library != library {
			return false
		}
	}
	return true
}

// NeedsInitValidation returns true if the device in question has the potential
// to crash at inference time and requires deeper validation before we include
// it in the supported devices list.
func (d DeviceInfo) NeedsInitValidation() bool {
	// ROCm: rocblas will crash on unsupported devices.
	// CUDA: verify CC is supported by the version of the library
	return d.Library == "ROCm" || d.Library == "CUDA"
}

// Set the init validation environment variable
func (d DeviceInfo) AddInitValidation(env map[string]string) {
	env["GGML_CUDA_INIT"] = "1" // force deep initialization to trigger crash on unsupported GPUs
}

// PreferredLibrary returns true if this library is preferred over the other input
// library
// Used to filter out Vulkan in favor of CUDA or ROCm
func (d DeviceInfo) PreferredLibrary(other DeviceInfo) bool {
	// TODO in the future if we find Vulkan is better than ROCm on some devices
	// that implementation can live here.

	if d.Library == "CUDA" || d.Library == "ROCm" {
		return true
	}
	return false
}

func (d DeviceInfo) updateVisibleDevicesEnv(env map[string]string, mustFilter bool) {
	var envVar string
	var rocmOrdinalEnv string
	switch d.Library {
	case "ROCm":
		// ROCm must be filtered as it can crash the runner on unsupported devices
		envVar = "ROCR_VISIBLE_DEVICES"
		if runtime.GOOS != "linux" {
			envVar = rocmNonLinuxVisibleDevicesEnv()
		} else {
			rocmOrdinalEnv = rocmLinuxOrdinalVisibleDevicesEnv()
		}
	case "CUDA":
		if !mustFilter {
			// By default we try to avoid filtering CUDA devices because ROCm also
			// looks at the CUDA env var, and gets confused in mixed-vendor environments.
			return
		}
		envVar = "CUDA_VISIBLE_DEVICES"
	case "Vulkan":
		if !mustFilter {
			return
		}
		envVar = "GGML_VK_VISIBLE_DEVICES"
	default:
		return
	}
	v, existing := env[envVar]
	childOrdinal := visibleDeviceCount(v)
	if existing {
		v = v + ","
	}
	if d.FilterID != "" {
		v = v + d.FilterID
	} else {
		v = v + d.ID
	}
	env[envVar] = v

	if rocmOrdinalEnv != "" {
		v, existing = env[rocmOrdinalEnv]
		if existing {
			v = v + ","
		}
		v = v + strconv.Itoa(childOrdinal)
		env[rocmOrdinalEnv] = v
	}
}

func visibleDeviceCount(value string) int {
	count := 0
	for _, field := range strings.Split(value, ",") {
		if strings.TrimSpace(field) != "" {
			count++
		}
	}
	return count
}

func rocmLinuxOrdinalVisibleDevicesEnv() string {
	if runtime.GOOS != "linux" || os.Getenv("ROCR_VISIBLE_DEVICES") != "" {
		return ""
	}
	for _, name := range []string{"HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES"} {
		if numericVisibleDeviceList(os.Getenv(name)) {
			return name
		}
	}
	return ""
}

func rocmNonLinuxVisibleDevicesEnv() string {
	for _, name := range []string{"HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES"} {
		if numericVisibleDeviceList(os.Getenv(name)) {
			return name
		}
	}
	return "HIP_VISIBLE_DEVICES"
}

func numericVisibleDeviceList(value string) bool {
	fields := strings.Split(value, ",")
	found := false
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		index, err := strconv.Atoi(field)
		if err != nil || index < 0 {
			return false
		}
		found = true
	}
	return found
}

type BaseRunner interface {
	// GetPort returns the localhost port number the runner is running on
	GetPort() int

	// HasExited indicates if the runner is no longer running.  This can be used during
	// bootstrap to detect if a given filtered device is incompatible and triggered an assert
	HasExited() bool
}

type RunnerDiscovery interface {
	BaseRunner

	// GetDeviceInfos will perform a query of the underlying device libraries
	// for device identification and free VRAM information
	// During bootstrap scenarios, this routine may take seconds to complete
	GetDeviceInfos(ctx context.Context) []DeviceInfo
}

type FilteredRunnerDiscovery interface {
	RunnerDiscovery

	// GetActiveDeviceIDs returns the filtered set of devices actively in
	// use by this runner for running models.  If the runner is a bootstrap runner, no devices
	// will be active yet so no device IDs are returned.
	// This routine will not query the underlying device and will return immediately
	GetActiveDeviceIDs() []DeviceID
}

