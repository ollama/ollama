package ml

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/maphash"
	"io"
	"log/slog"
	"math"
	"net/http"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/logutil"
)

// GPULayers is a set of layers to be allocated on a single GPU
type GPULayers struct {
	DeviceID

	// Layers is a set of layer indicies to load
	Layers []int
}

// FirstLayer returns the smallest layer index scheduled on this GPU, or MaxInt when empty.
func (g GPULayers) FirstLayer() int {
	if len(g.Layers) == 0 {
		return math.MaxInt
	}

	first := g.Layers[0]
	for i := 1; i < len(g.Layers); i++ {
		if g.Layers[i] < first {
			first = g.Layers[i]
		}
	}

	return first
}

func (g GPULayers) String() string {
	if len(g.Layers) == 0 {
		return ""
	}

	slices.Sort(g.Layers)

	contiguous := true
	base := g.Layers[0]
	for i := range g.Layers {
		if g.Layers[i] != base+i {
			contiguous = false
			break
		}
	}

	if contiguous {
		return fmt.Sprintf("ID:%v Layers:%v(%v..%v)", g.ID, len(g.Layers), g.Layers[0], g.Layers[len(g.Layers)-1])
	} else {
		return fmt.Sprintf("ID:%v Layers:%v%v", g.ID, len(g.Layers), g.Layers)
	}
}

// GPULayersList is a set of layer allocations across multiple GPUs
type GPULayersList []GPULayers

func (l GPULayersList) Len() int      { return len(l) }
func (l GPULayersList) Swap(i, j int) { l[i], l[j] = l[j], l[i] }

// Sort by the ordering of the layers offloaded
func (l GPULayersList) Less(i, j int) bool {
	li := l[i].FirstLayer()
	lj := l[j].FirstLayer()

	return li < lj
}

func (l GPULayersList) String() string {
	if l.Sum() > 0 {
		return fmt.Sprintf("%v%v", l.Sum(), []GPULayers(l))
	} else {
		return fmt.Sprintf("%v", []GPULayers(l))
	}
}

// Sum is the total number of layers assigned across all GPUs
func (l GPULayersList) Sum() int {
	var sum int

	for _, g := range l {
		sum += len(g.Layers)
	}

	return sum
}

var h maphash.Hash

// Hash is an identifier of this layer assignment
func (l GPULayersList) Hash() uint64 {
	h.Reset()
	for _, g := range l {
		if len(g.Layers) > 0 {
			h.WriteString(g.ID + g.Library)
			for _, l := range g.Layers {
				binary.Write(&h, binary.NativeEndian, int64(l))
			}
		}
	}

	return h.Sum64()
}

// ErrNoMem is returned when panicing due to insufficient memory. It includes
// the attempted memory allocation.
type ErrNoMem struct {
	BackendMemory
}

func (e ErrNoMem) Error() string {
	return fmt.Sprintf("insufficient memory - required allocations: %+v", e.BackendMemory)
}

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

// DeviceMemory provides a breakdown of the memory needed
// per device, such as a CPU or GPU.
type DeviceMemory struct {
	DeviceID

	// Name is the name of the device as labeled by the backend. It
	// may not be persistent across instances of the runner.
	Name string

	// Weights is the per-layer memory needed for the model weights.
	Weights []uint64

	// Cache is the per-layer memory needed for the KV cache.
	Cache []uint64

	// Graph is the size of the compute graph. It is not per-layer.
	Graph uint64
}

func sumMemory(mem []uint64) uint64 {
	var sum uint64

	for _, m := range mem {
		sum += m
	}

	return sum
}

// Size returns the total size of the memory required by this device
func (m DeviceMemory) Size() uint64 {
	return sumMemory(m.Weights) + sumMemory(m.Cache) + m.Graph
}

func memoryPresent(mem []uint64) bool {
	return slices.ContainsFunc(mem, func(m uint64) bool { return m != 0 })
}

func (m DeviceMemory) LogValue() slog.Value {
	var attrs []slog.Attr
	if memoryPresent(m.Weights) {
		attrs = append(attrs, slog.Any("Weights", m.Weights))
	}

	if memoryPresent(m.Cache) {
		attrs = append(attrs, slog.Any("Cache", m.Cache))
	}

	if m.Graph != 0 {
		attrs = append(attrs, slog.Any("Graph", m.Graph))
	}

	if len(attrs) > 0 && m.ID != "" {
		attrs = append([]slog.Attr{slog.String("ID", m.ID)}, attrs...)
	}

	return slog.GroupValue(attrs...)
}

// BackendMemory provides the amount of memory required to load the model
// per device based on the BackendParams. In some cases, not all required
// allocations will be known at this point. However, the size of the most recent
// allocation is guaranteed to be provided so that if it failed, the caller can
// accommodate that to make forward progress.
type BackendMemory struct {
	// InputWeights are always located on the CPU and cannot be moved
	InputWeights uint64

	// CPU model components are located in system memory. This does not
	// include unified memory allocated through the GPU.
	CPU DeviceMemory

	// GPU model components are located on one or more GPUs.
	GPUs []DeviceMemory
}

func (m BackendMemory) LogValue() slog.Value {
	var attrs []slog.Attr
	if m.InputWeights != 0 {
		attrs = append(attrs, slog.Any("InputWeights", m.InputWeights))
	}

	attrs = append(attrs, slog.Any(m.CPU.Name, m.CPU))
	for _, g := range m.GPUs {
		attrs = append(attrs, slog.Any(g.Name, g))
	}

	return slog.GroupValue(attrs...)
}

// Log prints a high level summary of the memory
func (m BackendMemory) Log(level slog.Level) {
	var total uint64

	for _, gpu := range m.GPUs {
		if sum := sumMemory(gpu.Weights); sum > 0 {
			slog.Log(context.TODO(), level, "model weights", "device", gpu.Name, "size", format.HumanBytes2(sum))
			total += sum
		}
	}
	if sum := m.InputWeights + sumMemory(m.CPU.Weights); sum > 0 {
		slog.Log(context.TODO(), level, "model weights", "device", m.CPU.Name, "size", format.HumanBytes2(sum))
		total += sum
	}

	for _, gpu := range m.GPUs {
		if sum := sumMemory(gpu.Cache); sum > 0 {
			slog.Log(context.TODO(), level, "kv cache", "device", gpu.Name, "size", format.HumanBytes2(sum))
			total += sum
		}
	}
	if sum := sumMemory(m.CPU.Cache); sum > 0 {
		slog.Log(context.TODO(), level, "kv cache", "device", m.CPU.Name, "size", format.HumanBytes2(sum))
		total += sum
	}

	for _, gpu := range m.GPUs {
		if sum := gpu.Graph; sum > 0 {
			slog.Log(context.TODO(), level, "compute graph", "device", gpu.Name, "size", format.HumanBytes2(sum))
			total += sum
		}
	}
	if sum := m.CPU.Graph; sum > 0 {
		slog.Log(context.TODO(), level, "compute graph", "device", m.CPU.Name, "size", format.HumanBytes2(sum))
		total += sum
	}

	if total > 0 {
		slog.Log(context.TODO(), level, "total memory", "size", format.HumanBytes2(total))
	}
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

	// Where backends were loaded from
	LibraryPath []string
}

type SystemInfo struct {
	// ThreadCount is the optimal number of threads to use for inference
	ThreadCount int `json:"threads,omitempty"`

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

// ByPerformance groups devices by similar speed
func ByPerformance(l []DeviceInfo) [][]DeviceInfo {
	resp := [][]DeviceInfo{}
	scores := []bool{}
	for _, info := range l {
		found := false
		requested := info.Integrated
		for i, score := range scores {
			if score == requested {
				resp[i] = append(resp[i], info)
				found = true
				break
			}
		}
		if !found {
			scores = append(scores, requested)
			resp = append(resp, []DeviceInfo{info})
		}
	}
	return resp
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

// For each GPU, check if it does NOT support flash attention
func FlashAttentionSupported(l []DeviceInfo) bool {
	for _, gpu := range l {
		supportsFA := gpu.Library == "cpu" ||
			gpu.Name == "Metal" || gpu.Library == "Metal" ||
			(gpu.Library == "CUDA" && gpu.DriverMajor >= 7 && !(gpu.ComputeMajor == 7 && gpu.ComputeMinor == 2)) ||
			gpu.Library == "ROCm" ||
			gpu.Library == "Vulkan"

		if !supportsFA {
			return false
		}
	}
	return true
}

// Given the list of GPUs this instantiation is targeted for,
// figure out the visible devices environment variables
// Set mustFilter true to enable filtering of CUDA devices
func GetVisibleDevicesEnv(l []DeviceInfo, mustFilter bool) map[string]string {
	if len(l) == 0 {
		return nil
	}
	env := map[string]string{}
	for _, d := range l {
		d.updateVisibleDevicesEnv(env, mustFilter)
	}
	return env
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
	switch d.Library {
	case "ROCm":
		// ROCm must be filtered as it can crash the runner on unsupported devices
		envVar = "ROCR_VISIBLE_DEVICES"
		if runtime.GOOS != "linux" {
			envVar = "HIP_VISIBLE_DEVICES"
		}
	case "CUDA":
		if !mustFilter {
			// By default we try to avoid filtering CUDA devices because ROCm also
			// looks at the CUDA env var, and gets confused in mixed vendor environments.
			return
		}
		envVar = "CUDA_VISIBLE_DEVICES"
	default:
		// Vulkan is not filtered via env var, but via scheduling decisions
		return
	}
	v, existing := env[envVar]
	if existing {
		v = v + ","
	}
	if d.FilterID != "" {
		v = v + d.FilterID
	} else {
		v = v + d.ID
	}
	env[envVar] = v
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

func GetDevicesFromRunner(ctx context.Context, runner BaseRunner) ([]DeviceInfo, error) {
	var moreDevices []DeviceInfo
	port := runner.GetPort()
	tick := time.Tick(10 * time.Millisecond)
	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("failed to finish discovery before timeout")
		case <-tick:
			r, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/info", port), nil)
			if err != nil {
				return nil, fmt.Errorf("failed to create request: %w", err)
			}
			r.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(r)
			if err != nil {
				// slog.Warn("failed to send request", "error", err)
				if runner.HasExited() {
					return nil, fmt.Errorf("runner crashed")
				}
				continue
			}
			defer resp.Body.Close()

			if resp.StatusCode == http.StatusNotFound {
				// old runner, fall back to bootstrapping model
				return nil, fmt.Errorf("llamarunner free vram reporting not supported")
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				slog.Warn("failed to read response", "error", err)
				continue
			}
			if resp.StatusCode != 200 {
				logutil.Trace("runner failed to discover free VRAM", "status", resp.StatusCode, "response", body)
				return nil, fmt.Errorf("runner error: %s", string(body))
			}

			if err := json.Unmarshal(body, &moreDevices); err != nil {
				slog.Warn("unmarshal encode response", "error", err)
				continue
			}
			return moreDevices, nil
		}
	}
}
