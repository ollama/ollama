package discover

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

// Discovery logic for AMD/ROCm GPUs

const (
	DriverVersionFile     = "/sys/module/amdgpu/version"
	AMDNodesSysfsDir      = "/sys/class/kfd/kfd/topology/nodes/"
	GPUPropertiesFileGlob = AMDNodesSysfsDir + "*/properties"

	// Prefix with the node dir
	GPUTotalMemoryFileGlob = "mem_banks/*/properties" // size_in_bytes line

	// Direct Rendering Manager sysfs location
	DRMDeviceDirGlob   = "/sys/class/drm/card*/device"
	DRMTotalMemoryFile = "mem_info_vram_total"
	DRMUsedMemoryFile  = "mem_info_vram_used"

	// In hex; properties file is in decimal
	DRMUniqueIDFile = "unique_id"
	DRMVendorFile   = "vendor"
	DRMDeviceFile   = "device"
)

var (
	// Used to validate if the given ROCm lib is usable
	ROCmLibGlobs          = []string{"libhipblas.so.2*", "rocblas"} // TODO - probably include more coverage of files here...
	RocmStandardLocations = []string{"/opt/rocm/lib", "/usr/lib64"}
)

// Gather GPU information from the amdgpu driver if any supported GPUs are detected
// Only called once during bootstrap
func AMDGetGPUInfo() ([]RocmGPUInfo, error) {
	resp := []RocmGPUInfo{}
	if !AMDDetected() {
		return resp, fmt.Errorf("AMD GPUs not detected")
	}

	// Opportunistic logging of driver version to aid in troubleshooting
	driverMajor, driverMinor, err := AMDDriverVersion()
	if err != nil {
		// TODO - if we see users crash and burn with the upstreamed kernel this can be adjusted to hard-fail rocm support and fallback to CPU
		slog.Warn("ollama recommends running the https://www.amd.com/en/support/linux-drivers", "error", err)
	}

	// Determine if the user has already pre-selected which GPUs to look at, then ignore the others
	var visibleDevices []string
	hipVD := envconfig.HipVisibleDevices()   // zero based index only
	rocrVD := envconfig.RocrVisibleDevices() // zero based index or UUID, but consumer cards seem to not support UUID
	gpuDO := envconfig.GpuDeviceOrdinal()    // zero based index
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

	gfxOverride := envconfig.HsaOverrideGfxVersion()
	var supported []string
	libDir := ""

	// The amdgpu driver always exposes the host CPU(s) first, but we have to skip them and subtract
	// from the other IDs to get alignment with the HIP libraries expectations (zero is the first GPU, not the CPU)
	matches, _ := filepath.Glob(GPUPropertiesFileGlob)
	sort.Slice(matches, func(i, j int) bool {
		// /sys/class/kfd/kfd/topology/nodes/<number>/properties
		a, err := strconv.ParseInt(filepath.Base(filepath.Dir(matches[i])), 10, 64)
		if err != nil {
			slog.Debug("parse err", "error", err, "match", matches[i])
			return false
		}
		b, err := strconv.ParseInt(filepath.Base(filepath.Dir(matches[j])), 10, 64)
		if err != nil {
			slog.Debug("parse err", "error", err, "match", matches[i])
			return false
		}
		return a < b
	})
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
		var vendor, device, uniqueID uint64
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
					slog.Debug("malformed", "vendor_id", line)
					continue
				}
				vendor, err = strconv.ParseUint(ver[1], 10, 64)
				if err != nil {
					slog.Debug("malformed", "vendor_id", line, "error", err)
				}
			} else if strings.HasPrefix(line, "device_id") {
				ver := strings.Fields(line)
				if len(ver) != 2 {
					slog.Debug("malformed", "device_id", line)
					continue
				}
				device, err = strconv.ParseUint(ver[1], 10, 64)
				if err != nil {
					slog.Debug("malformed", "device_id", line, "error", err)
				}
			} else if strings.HasPrefix(line, "unique_id") {
				ver := strings.Fields(line)
				if len(ver) != 2 {
					slog.Debug("malformed", "unique_id", line)
					continue
				}
				uniqueID, err = strconv.ParseUint(ver[1], 10, 64)
				if err != nil {
					slog.Debug("malformed", "unique_id", line, "error", err)
				}
			}
			// TODO - any other properties we want to extract and record?
			// vendor_id + device_id -> pci lookup for "Name"
			// Other metrics that may help us understand relative performance between multiple GPUs
		}

		// Note: while ./mem_banks/*/used_memory exists, it doesn't appear to take other VRAM consumers
		// into consideration, so we instead map the device over to the DRM driver sysfs nodes which
		// do reliably report VRAM usage.

		if isCPU {
			cpuCount++
			continue
		}

		// CPUs are always first in the list
		gpuID := nodeID - cpuCount

		// Shouldn't happen, but just in case...
		if gpuID < 0 {
			err := fmt.Errorf("unexpected amdgpu sysfs data resulted in negative GPU ID, please set OLLAMA_DEBUG=1 and report an issue")
			slog.Error(err.Error())
			return nil, err
		}

		// Look up the memory for the current node
		totalMemory := uint64(0)
		usedMemory := uint64(0)
		var usedFile string
		mapping := []struct {
			id       uint64
			filename string
		}{
			{vendor, DRMVendorFile},
			{device, DRMDeviceFile},
			{uniqueID, DRMUniqueIDFile}, // Not all devices will report this
		}
		slog.Debug("mapping amdgpu to drm sysfs nodes", "amdgpu", match, "vendor", vendor, "device", device, "unique_id", uniqueID)
		// Map over to DRM location to find the total/free memory
		drmMatches, _ := filepath.Glob(DRMDeviceDirGlob)
		for _, devDir := range drmMatches {
			matched := true
			for _, m := range mapping {
				if m.id == 0 {
					// Null ID means it didn't populate, so we can't use it to match
					continue
				}
				filename := filepath.Join(devDir, m.filename)
				buf, err := os.ReadFile(filename)
				if err != nil {
					slog.Debug("failed to read sysfs node", "file", filename, "error", err)
					matched = false
					break
				}
				// values here are in hex, strip off the lead 0x and parse so we can compare the numeric (decimal) values in amdgpu
				cmp, err := strconv.ParseUint(strings.TrimPrefix(strings.TrimSpace(string(buf)), "0x"), 16, 64)
				if err != nil {
					slog.Debug("failed to parse sysfs node", "file", filename, "error", err)
					matched = false
					break
				}
				if cmp != m.id {
					matched = false
					break
				}
			}
			if !matched {
				continue
			}

			// Found the matching DRM directory
			slog.Debug("matched", "amdgpu", match, "drm", devDir)
			totalFile := filepath.Join(devDir, DRMTotalMemoryFile)
			buf, err := os.ReadFile(totalFile)
			if err != nil {
				slog.Debug("failed to read sysfs node", "file", totalFile, "error", err)
				break
			}
			totalMemory, err = strconv.ParseUint(strings.TrimSpace(string(buf)), 10, 64)
			if err != nil {
				slog.Debug("failed to parse sysfs node", "file", totalFile, "error", err)
				break
			}

			usedFile = filepath.Join(devDir, DRMUsedMemoryFile)
			usedMemory, err = getFreeMemory(usedFile)
			if err != nil {
				slog.Debug("failed to update used memory", "error", err)
			}
			break
		}

		var name string
		// TODO - PCI ID lookup
		if vendor > 0 && device > 0 {
			name = fmt.Sprintf("%04x:%04x", vendor, device)
		}

		gpuInfo := RocmGPUInfo{
			GpuInfo: GpuInfo{
				Library: "rocm",
				memInfo: memInfo{
					TotalMemory: totalMemory,
					FreeMemory:  (totalMemory - usedMemory),
				},
				ID:            strconv.Itoa(gpuID),
				Name:          name,
				Compute:       fmt.Sprintf("gfx%d%x%x", major, minor, patch),
				MinimumMemory: rocmMinimumMemory,
				DriverMajor:   driverMajor,
				DriverMinor:   driverMinor,
			},
			usedFilepath: usedFile,
		}

		// iGPU detection, remove this check once we can support an iGPU variant of the rocm library
		if totalMemory < IGPUMemLimit {
			reason := "unsupported Radeon iGPU detected skipping"
			slog.Info(reason, "id", gpuID, "total", format.HumanBytes2(totalMemory))
			unsupportedGPUs = append(unsupportedGPUs, UnsupportedGPUInfo{
				GpuInfo: gpuInfo.GpuInfo,
				Reason:  reason,
			})
			continue
		}

		if int(major) < RocmComputeMin {
			reason := fmt.Sprintf("amdgpu too old gfx%d%x%x", major, minor, patch)
			slog.Warn(reason, "gpu", gpuID)
			unsupportedGPUs = append(unsupportedGPUs, UnsupportedGPUInfo{
				GpuInfo: gpuInfo.GpuInfo,
				Reason:  reason,
			})

			continue
		}

		slog.Debug("amdgpu memory", "gpu", gpuID, "total", format.HumanBytes2(totalMemory))
		slog.Debug("amdgpu memory", "gpu", gpuID, "available", format.HumanBytes2(totalMemory-usedMemory))

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
				reason := "filtering out device per user request"
				slog.Info(reason, "id", gpuInfo.ID, "visible_devices", visibleDevices)
				unsupportedGPUs = append(unsupportedGPUs, UnsupportedGPUInfo{
					GpuInfo: gpuInfo.GpuInfo,
					Reason:  reason,
				})

				continue
			}
		}

		// Final validation is gfx compatibility - load the library if we haven't already loaded it
		// even if the user overrides, we still need to validate the library
		if libDir == "" {
			libDir, err = AMDValidateLibDir()
			if err != nil {
				err = fmt.Errorf("unable to verify rocm library: %w", err)
				slog.Warn(err.Error())
				unsupportedGPUs = append(unsupportedGPUs, UnsupportedGPUInfo{
					GpuInfo: gpuInfo.GpuInfo,
					Reason:  err.Error(),
				})
				return nil, err
			}
		}
		gpuInfo.DependencyPath = libDir

		if gfxOverride == "" {
			// Only load supported list once
			if len(supported) == 0 {
				supported, err = GetSupportedGFX(libDir)
				if err != nil {
					err = fmt.Errorf("failed to lookup supported GFX types: %w", err)
					slog.Warn(err.Error())
					unsupportedGPUs = append(unsupportedGPUs, UnsupportedGPUInfo{
						GpuInfo: gpuInfo.GpuInfo,
						Reason:  err.Error(),
					})
					return nil, err
				}
				slog.Debug("rocm supported GPUs", "types", supported)
			}
			gfx := gpuInfo.Compute
			if !slices.Contains[[]string, string](supported, gfx) {
				reason := fmt.Sprintf("amdgpu is not supported (supported types:%s)", supported)
				slog.Warn(reason, "gpu_type", gfx, "gpu", gpuInfo.ID, "library", libDir)
				unsupportedGPUs = append(unsupportedGPUs, UnsupportedGPUInfo{
					GpuInfo: gpuInfo.GpuInfo,
					Reason:  reason,
				})

				// TODO - consider discrete markdown just for ROCM troubleshooting?
				slog.Warn("See https://github.com/ollama/ollama/blob/main/docs/gpu.md#overrides for HSA_OVERRIDE_GFX_VERSION usage")
				continue
			} else {
				slog.Info("amdgpu is supported", "gpu", gpuInfo.ID, "gpu_type", gfx)
			}
		} else {
			slog.Info("skipping rocm gfx compatibility check", "HSA_OVERRIDE_GFX_VERSION", gfxOverride)
		}

		// Check for env var workarounds
		if name == "1002:687f" { // Vega RX 56
			gpuInfo.EnvWorkarounds = append(gpuInfo.EnvWorkarounds, [2]string{"HSA_ENABLE_SDMA", "0"})
		}

		// The GPU has passed all the verification steps and is supported
		resp = append(resp, gpuInfo)
	}
	if len(resp) == 0 {
		err := fmt.Errorf("no compatible amdgpu devices detected")
		slog.Info(err.Error())
		return nil, err
	}
	if err := verifyKFDDriverAccess(); err != nil {
		err = fmt.Errorf("amdgpu devices detected but permission problems block access: %w", err)
		slog.Error(err.Error())
		return nil, err
	}
	return resp, nil
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
	return "", errors.New("no suitable rocm found, falling back to CPU")
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

func (gpus RocmGPUInfoList) RefreshFreeMemory() error {
	if len(gpus) == 0 {
		return nil
	}
	for i := range gpus {
		usedMemory, err := getFreeMemory(gpus[i].usedFilepath)
		if err != nil {
			return err
		}
		slog.Debug("updating rocm free memory", "gpu", gpus[i].ID, "name", gpus[i].Name, "before", format.HumanBytes2(gpus[i].FreeMemory), "now", format.HumanBytes2(gpus[i].TotalMemory-usedMemory))
		gpus[i].FreeMemory = gpus[i].TotalMemory - usedMemory
	}
	return nil
}

func getFreeMemory(usedFile string) (uint64, error) {
	buf, err := os.ReadFile(usedFile)
	if err != nil {
		return 0, fmt.Errorf("failed to read sysfs node %s %w", usedFile, err)
	}
	usedMemory, err := strconv.ParseUint(strings.TrimSpace(string(buf)), 10, 64)
	if err != nil {
		slog.Debug("failed to parse sysfs node", "file", usedFile, "error", err)
		return 0, fmt.Errorf("failed to parse sysfs node %s %w", usedFile, err)
	}
	return usedMemory, nil
}

func verifyKFDDriverAccess() error {
	// Verify we have permissions - either running as root, or we have group access to the driver
	fd, err := os.OpenFile("/dev/kfd", os.O_RDWR, 0o666)
	if err != nil {
		if errors.Is(err, fs.ErrPermission) {
			return fmt.Errorf("permissions not set up properly.  Either run ollama as root, or add you user account to the render group. %w", err)
		} else if errors.Is(err, fs.ErrNotExist) {
			// Container runtime failure?
			return fmt.Errorf("kfd driver not loaded.  If running in a container, remember to include '--device /dev/kfd --device /dev/dri'")
		}
		return fmt.Errorf("failed to check permission on /dev/kfd: %w", err)
	}
	fd.Close()
	return nil
}
