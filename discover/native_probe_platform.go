//go:build (linux && cgo) || windows

package discover

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

const (
	ggmlBackendDeviceTypeGPU  = 1
	ggmlBackendDeviceTypeIGPU = 2
)

// Vulkan values are defined locally so VK_INTEL_device_info (headers 1.4.363+)
// can be queried regardless of the Vulkan SDK used at build time.
const (
	intelVendorID = 0x8086

	vkIntelDeviceInfoExtensionName = "VK_INTEL_device_info"
	vkEXTPCIBusInfoExtensionName   = "VK_EXT_pci_bus_info"

	vkSuccess                                        = 0
	vkStructureTypeApplicationInfo                   = 0
	vkStructureTypeInstanceCreateInfo                = 1
	vkStructureTypePhysicalDeviceProperties2         = 1000059001
	vkStructureTypePhysicalDevicePCIBusInfoEXT       = 1000212000
	vkStructureTypePhysicalDeviceInfoPropertiesINTEL = 1000708000
	vkPhysicalDeviceTypeIntegratedGPU                = 1

	vkAPIVersion11 = 1<<22 | 1<<12
)

func ggmlDeviceTypeIntegrated(deviceType int32) bool {
	return deviceType == ggmlBackendDeviceTypeIGPU
}

func ggmlProbeLibraryName(name string) string {
	switch strings.ToLower(name) {
	case "cuda":
		return "CUDA"
	case "hip", "rocm":
		return "ROCm"
	case "vulkan":
		return "Vulkan"
	case "metal":
		return "Metal"
	default:
		return name
	}
}

func ggmlLibraryFile(dir, name string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(dir, name+".dll")
	}

	exact := filepath.Join(dir, "lib"+name+".so")
	if _, err := os.Stat(exact); err == nil {
		return exact
	}
	matches, _ := filepath.Glob(exact + ".*")
	if len(matches) > 0 {
		sort.Strings(matches)
		return matches[len(matches)-1]
	}
	return exact
}

func nativeProbeBackendFiles(libDirs []string) []string {
	var files []string
	seen := map[string]bool{}
	for _, dir := range libDirs {
		for _, pattern := range nativeProbeBackendPatterns(dir) {
			matches, _ := filepath.Glob(pattern)
			for _, match := range matches {
				if seen[match] {
					continue
				}
				seen[match] = true
				files = append(files, match)
			}
		}
	}
	return files
}

func nativeProbeBackendPatterns(dir string) []string {
	if runtime.GOOS == "windows" {
		return []string{
			filepath.Join(dir, "ggml-cuda.dll"),
			filepath.Join(dir, "ggml-hip.dll"),
			filepath.Join(dir, "ggml-vulkan.dll"),
		}
	}

	return []string{
		filepath.Join(dir, "libggml-cuda.so"),
		filepath.Join(dir, "libggml-hip.so"),
		filepath.Join(dir, "libggml-vulkan.so"),
	}
}

func nativeProbeHasCUDA(libDirs []string) bool {
	for _, dir := range libDirs {
		if strings.Contains(strings.ToLower(filepath.Base(dir)), "cuda") {
			return true
		}
	}
	for _, file := range nativeProbeBackendFiles(libDirs) {
		if strings.Contains(strings.ToLower(filepath.Base(file)), "cuda") {
			return true
		}
	}
	return false
}

func nativeProbeHasROCm(libDirs []string) bool {
	for _, dir := range libDirs {
		base := strings.ToLower(filepath.Base(dir))
		if strings.Contains(base, "rocm") || strings.Contains(base, "hip") {
			return true
		}
	}
	for _, file := range nativeProbeBackendFiles(libDirs) {
		base := strings.ToLower(filepath.Base(file))
		if strings.Contains(base, "hip") {
			return true
		}
	}
	return false
}

func nativeProbeHasVulkan(libDirs []string) bool {
	for _, dir := range libDirs {
		if strings.Contains(strings.ToLower(filepath.Base(dir)), "vulkan") {
			return true
		}
	}
	for _, file := range nativeProbeBackendFiles(libDirs) {
		if strings.Contains(strings.ToLower(filepath.Base(file)), "vulkan") {
			return true
		}
	}
	return false
}

func addIntelVulkanDeviceInfo(devices []nativeProbeDevice, libDirs []string) {
	hasIntegratedVulkan := false
	for _, dev := range devices {
		if dev.Library == "Vulkan" && dev.Integrated {
			hasIntegratedVulkan = true
			break
		}
	}
	if !hasIntegratedVulkan || !nativeProbeHasVulkan(libDirs) {
		return
	}

	probed, err := probeIntelVulkanDevices(libDirs)
	if err != nil {
		slog.Debug("intel vulkan device info unavailable", "error", err)
		return
	}
	applyIntelVulkanDeviceInfo(devices, probed)
}

func parseNVIDIADriverMajor(version string) (int, error) {
	version = strings.TrimSpace(version)
	if version == "" {
		return 0, errors.New("empty NVIDIA driver version")
	}
	major, _, _ := strings.Cut(version, ".")
	driver, err := strconv.Atoi(major)
	if err != nil {
		return 0, fmt.Errorf("parse NVIDIA driver version %q: %w", version, err)
	}
	return driver, nil
}
