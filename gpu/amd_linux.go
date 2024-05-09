package gpu

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

// Discovery logic for AMD/ROCm GPUs

const (
	DriverVersionFile     = "/sys/module/amdgpu/version"
	AMDNodesSysfsDir      = "/sys/class/kfd/kfd/topology/nodes/"
	GPUPropertiesFileGlob = AMDNodesSysfsDir + "*/properties"

	// Prefix with the node dir
	GPUTotalMemoryFileGlob = "mem_banks/*/properties" // size_in_bytes line
	GPUUsedMemoryFileGlob  = "mem_banks/*/used_memory"
)

var (
	// Used to validate if the given ROCm lib is usable
	ROCmLibGlobs          = []string{"libhipblas.so.2*", "rocblas"} // TODO - probably include more coverage of files here...
	RocmStandardLocations = []string{"/opt/rocm/lib", "/usr/lib64"}
)

// Gather GPU information from the amdgpu driver if any supported GPUs are detected
func AMDGetGPUInfo() []GpuInfo {
	resp := []GpuInfo{}
	if !AMDDetected() {
		return resp
	}

	// Opportunistic logging of driver version to aid in troubleshooting
	driverMajor, driverMinor, err := AMDDriverVersion()
	if err != nil {
		// TODO - if we see users crash and burn with the upstreamed kernel this can be adjusted to hard-fail rocm support and fallback to CPU
		slog.Warn("ollama recommends running the https://www.amd.com/en/support/linux-drivers", "error", err)
	}

	// Determine if the user has already pre-selected which GPUs to look at, then ignore the others
	var visibleDevices []string
	hipVD := os.Getenv("HIP_VISIBLE_DEVICES")   // zero based index only
	rocrVD := os.Getenv("ROCR_VISIBLE_DEVICES") // zero based index or UUID, but consumer cards seem to not support UUID
	gpuDO := os.Getenv("GPU_DEVICE_ORDINAL")    // zero based index
	switch {
	// TODO is this priorty order right?
	case hipVD != "":
		visibleDevices = strings.Split(hipVD, ",")
	case rocrVD != "":
		visibleDevices = strings.Split(rocrVD, ",")
		// TODO - since we don't yet support UUIDs, consider detecting and reporting here
		// all our test systems show GPU-XX indicating UUID is not supported
	case gpuDO != "":
		visibleDevices = strings.Split(gpuDO, ",")
	}

	gfxOverride := os.Getenv("HSA_OVERRIDE_GFX_VERSION")
	var supported []string
	libDir := ""

	// The amdgpu driver always exposes the host CPU(s) first, but we have to skip them and subtract
	// from the other IDs to get alignment with the HIP libraries expectations (zero is the first GPU, not the CPU)
	matches, _ := filepath.Glob(GPUPropertiesFileGlob)
	cpuCount := 0
	for _, match := range matches {
		slog.Debug("evaluating amdgpu node " + match)
		fp, err := os.Open(match)
		if err != nil {
			slog.Debug("failed to open sysfs node", "file", match, "error", err)
			continue
		}
		defer fp.Close()
		nodeID, err := strconv.Atoi(filepath.Base(filepath.Dir(match)))
		if err != nil {
			slog.Debug("failed to parse node ID", "error", err)
			continue
		}

		scanner := bufio.NewScanner(fp)
		isCPU := false
		var major, minor, patch uint64
		var vendor, device uint64
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			// Note: we could also use "cpu_cores_count X" where X is greater than zero to detect CPUs
			if strings.HasPrefix(line, "gfx_target_version") {
				ver := strings.Fields(line)

				// Detect CPUs
				if len(ver) == 2 && ver[1] == "0" {
					slog.Debug("detected CPU " + match)
					isCPU = true
					break
				}

				if len(ver) != 2 || len(ver[1]) < 5 {
					slog.Warn("malformed "+match, "gfx_target_version", line)
					// If this winds up being a CPU, our offsets may be wrong
					continue
				}
				l := len(ver[1])
				var err1, err2, err3 error
				patch, err1 = strconv.ParseUint(ver[1][l-2:l], 10, 32)
				minor, err2 = strconv.ParseUint(ver[1][l-4:l-2], 10, 32)
				major, err3 = strconv.ParseUint(ver[1][:l-4], 10, 32)
				if err1 != nil || err2 != nil || err3 != nil {
					slog.Debug("malformed int " + line)
					continue
				}
			} else if strings.HasPrefix(line, "vendor_id") {
				ver := strings.Fields(line)
				if len(ver) != 2 {
					slog.Debug("malformed vendor_id", "vendor_id", line)
					continue
				}
				vendor, err = strconv.ParseUint(ver[1], 10, 32)
				if err != nil {
					slog.Debug("malformed vendor_id" + line)
				}
			} else if strings.HasPrefix(line, "device_id") {
				ver := strings.Fields(line)
				if len(ver) != 2 {
					slog.Debug("malformed device_id", "device_id", line)
					continue
				}
				device, err = strconv.ParseUint(ver[1], 10, 32)
				if err != nil {
					slog.Debug("malformed device_id" + line)
				}
			}

			// TODO - any other properties we want to extract and record?
			// vendor_id + device_id -> pci lookup for "Name"
			// Other metrics that may help us understand relative performance between multiple GPUs
		}

		if isCPU {
			cpuCount++
			continue
		}

		// CPUs are always first in the list
		gpuID := nodeID - cpuCount

		// Shouldn't happen, but just in case...
		if gpuID < 0 {
			slog.Error("unexpected amdgpu sysfs data resulted in negative GPU ID, please set OLLAMA_DEBUG=1 and report an issue")
			return []GpuInfo{}
		}

		if int(major) < RocmComputeMin {
			slog.Warn(fmt.Sprintf("amdgpu too old gfx%d%x%x", major, minor, patch), "gpu", gpuID)
			continue
		}

		// Look up the memory for the current node
		totalMemory := uint64(0)
		usedMemory := uint64(0)
		propGlob := filepath.Join(AMDNodesSysfsDir, strconv.Itoa(nodeID), GPUTotalMemoryFileGlob)
		propFiles, err := filepath.Glob(propGlob)
		if err != nil {
			slog.Warn("error looking up total GPU memory", "glob", propGlob, "error", err)
		}
		// 1 or more memory banks - sum the values of all of them
		for _, propFile := range propFiles {
			fp, err := os.Open(propFile)
			if err != nil {
				slog.Warn("failed to open sysfs node", "file", propFile, "erroir", err)
				continue
			}
			defer fp.Close()
			scanner := bufio.NewScanner(fp)
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if strings.HasPrefix(line, "size_in_bytes") {
					ver := strings.Fields(line)
					if len(ver) != 2 {
						slog.Warn("malformed " + line)
						continue
					}
					bankSizeInBytes, err := strconv.ParseUint(ver[1], 10, 64)
					if err != nil {
						slog.Warn("malformed int " + line)
						continue
					}
					totalMemory += bankSizeInBytes
				}
			}
		}
		if totalMemory == 0 {
			slog.Warn("amdgpu reports zero total memory", "gpu", gpuID)
			continue
		}
		usedGlob := filepath.Join(AMDNodesSysfsDir, strconv.Itoa(nodeID), GPUUsedMemoryFileGlob)
		usedFiles, err := filepath.Glob(usedGlob)
		if err != nil {
			slog.Warn("error looking up used GPU memory", "glob", usedGlob, "error", err)
			continue
		}
		for _, usedFile := range usedFiles {
			fp, err := os.Open(usedFile)
			if err != nil {
				slog.Warn("failed to open sysfs node", "file", usedFile, "error", err)
				continue
			}
			defer fp.Close()
			data, err := io.ReadAll(fp)
			if err != nil {
				slog.Warn("failed to read sysfs node", "file", usedFile, "error", err)
				continue
			}
			used, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			if err != nil {
				slog.Warn("malformed used memory", "data", string(data), "error", err)
				continue
			}
			usedMemory += used
		}

		// iGPU detection, remove this check once we can support an iGPU variant of the rocm library
		if totalMemory < IGPUMemLimit {
			slog.Info("unsupported Radeon iGPU detected skipping", "id", gpuID, "total", format.HumanBytes2(totalMemory))
			continue
		}
		var name string
		// TODO - PCI ID lookup
		if vendor > 0 && device > 0 {
			name = fmt.Sprintf("%04x:%04x", vendor, device)
		}

		slog.Debug("amdgpu memory", "gpu", gpuID, "total", format.HumanBytes2(totalMemory))
		slog.Debug("amdgpu memory", "gpu", gpuID, "available", format.HumanBytes2(totalMemory-usedMemory))
		gpuInfo := GpuInfo{
			Library: "rocm",
			memInfo: memInfo{
				TotalMemory: totalMemory,
				FreeMemory:  (totalMemory - usedMemory),
			},
			ID:            fmt.Sprintf("%d", gpuID),
			Name:          name,
			Compute:       fmt.Sprintf("gfx%d%x%x", major, minor, patch),
			MinimumMemory: rocmMinimumMemory,
			DriverMajor:   driverMajor,
			DriverMinor:   driverMinor,
		}

		// If the user wants to filter to a subset of devices, filter out if we aren't a match
		if len(visibleDevices) > 0 {
			include := false
			for _, visible := range visibleDevices {
				if visible == gpuInfo.ID {
					include = true
					break
				}
			}
			if !include {
				slog.Info("filtering out device per user request", "id", gpuInfo.ID, "visible_devices", visibleDevices)
				continue
			}
		}

		// Final validation is gfx compatibility - load the library if we haven't already loaded it
		// even if the user overrides, we still need to validate the library
		if libDir == "" {
			libDir, err = AMDValidateLibDir()
			if err != nil {
				slog.Warn("unable to verify rocm library, will use cpu", "error", err)
				return []GpuInfo{}
			}
		}
		gpuInfo.DependencyPath = libDir

		if gfxOverride == "" {
			// Only load supported list once
			if len(supported) == 0 {
				supported, err = GetSupportedGFX(libDir)
				if err != nil {
					slog.Warn("failed to lookup supported GFX types, falling back to CPU mode", "error", err)
					return []GpuInfo{}
				}
				slog.Debug("rocm supported GPUs", "types", supported)
			}
			gfx := gpuInfo.Compute
			if !slices.Contains[[]string, string](supported, gfx) {
				slog.Warn("amdgpu is not supported", "gpu", gpuInfo.ID, "gpu_type", gfx, "library", libDir, "supported_types", supported)
				// TODO - consider discrete markdown just for ROCM troubleshooting?
				slog.Warn("See https://github.com/ollama/ollama/blob/main/docs/gpu.md#overrides for HSA_OVERRIDE_GFX_VERSION usage")
				continue
			} else {
				slog.Info("amdgpu is supported", "gpu", gpuInfo.ID, "gpu_type", gfx)
			}
		} else {
			slog.Info("skipping rocm gfx compatibility check", "HSA_OVERRIDE_GFX_VERSION", gfxOverride)
		}

		// The GPU has passed all the verification steps and is supported
		resp = append(resp, gpuInfo)
	}
	if len(resp) == 0 {
		slog.Info("no compatible amdgpu devices detected")
	}
	return resp
}

// Quick check for AMD driver so we can skip amdgpu discovery if not present
func AMDDetected() bool {
	// Some driver versions (older?) don't have a version file, so just lookup the parent dir
	sysfsDir := filepath.Dir(DriverVersionFile)
	_, err := os.Stat(sysfsDir)
	if errors.Is(err, os.ErrNotExist) {
		slog.Debug("amdgpu driver not detected " + sysfsDir)
		return false
	} else if err != nil {
		slog.Debug("error looking up amd driver", "path", sysfsDir, "error", err)
		return false
	}
	return true
}

// Prefer to use host installed ROCm, as long as it meets our minimum requirements
// failing that, tell the user how to download it on their own
func AMDValidateLibDir() (string, error) {
	libDir, err := commonAMDValidateLibDir()
	if err == nil {
		return libDir, nil
	}

	// Well known ollama installer path
	installedRocmDir := "/usr/share/ollama/lib/rocm"
	if rocmLibUsable(installedRocmDir) {
		return installedRocmDir, nil
	}

	// If we still haven't found a usable rocm, the user will have to install it on their own
	slog.Warn("amdgpu detected, but no compatible rocm library found.  Either install rocm v6, or follow manual install instructions at https://github.com/ollama/ollama/blob/main/docs/linux.md#manual-install")
	return "", fmt.Errorf("no suitable rocm found, falling back to CPU")
}

func AMDDriverVersion() (driverMajor, driverMinor int, err error) {
	_, err = os.Stat(DriverVersionFile)
	if err != nil {
		return 0, 0, fmt.Errorf("amdgpu version file missing: %s %w", DriverVersionFile, err)
	}
	fp, err := os.Open(DriverVersionFile)
	if err != nil {
		return 0, 0, err
	}
	defer fp.Close()
	verString, err := io.ReadAll(fp)
	if err != nil {
		return 0, 0, err
	}

	pattern := `\A(\d+)\.(\d+).*`
	regex := regexp.MustCompile(pattern)
	match := regex.FindStringSubmatch(string(verString))
	if len(match) < 2 {
		return 0, 0, fmt.Errorf("malformed version string %s", string(verString))
	}
	driverMajor, err = strconv.Atoi(match[1])
	if err != nil {
		return 0, 0, err
	}
	driverMinor, err = strconv.Atoi(match[2])
	if err != nil {
		return 0, 0, err
	}
	return driverMajor, driverMinor, nil
}
