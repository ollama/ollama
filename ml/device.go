package ml

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"log/slog"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

// GPULayers is a set of layers to be allocated on a single GPU
type GPULayers struct {
	DeviceID

	// Layers is a set of layer indicies to load
	Layers []int
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
	FilteredID string `json:"filtered_id,omitempty"`

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
