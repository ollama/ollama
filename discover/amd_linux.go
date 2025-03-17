package discover

import (
	"bufio"
	"bytes"
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


func AMDGetGPUInfo() []RocmGPUInfo {
	resp := AMDGetGPUInfoFromDriver()
	if len(resp) != 0 {
		return resp
	}

	return AMDGetGPUInfoFromRuntime()
}

// Gather GPU information from the amdgpu driver if any supported GPUs are detected
func AMDGetGPUInfoFromDriver() []RocmGPUInfo {
	resp := []RocmGPUInfo{}
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
			slog.Error("unexpected amdgpu sysfs data resulted in negative GPU ID, please set OLLAMA_DEBUG=1 and report an issue")
			return nil
		}

		//if int(major) < RocmComputeMin {
		//	slog.Warn(fmt.Sprintf("amdgpu too old gfx%d%x%x", major, minor, patch), "gpu", gpuID)
		//	continue
		//}

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
				return nil
			}
		}
		gpuInfo.DependencyPath = strings.Split(libDir, "")

		if gfxOverride == "" {
			// Only load supported list once
			if len(supported) == 0 {
				supported, err = GetSupportedGFX(libDir)
				if err != nil {
					slog.Warn("failed to lookup supported GFX types, falling back to CPU mode", "error", err)
					return nil
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

		// Check for env var workarounds
		if name == "1002:687f" { // Vega RX 56
			gpuInfo.EnvWorkarounds = append(gpuInfo.EnvWorkarounds, [2]string{"HSA_ENABLE_SDMA", "0"})
		}

		// The GPU has passed all the verification steps and is supported
		resp = append(resp, gpuInfo)
	}
	if len(resp) == 0 {
		slog.Info("no compatible amdgpu devices detected")
	}
	if err := verifyKFDDriverAccess(); err != nil {
		slog.Error("amdgpu devices detected but permission problems block access", "error", err)
		return nil
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

func AMDGetGPUInfoFromRuntime() []RocmGPUInfo {
	resp := []RocmGPUInfo{}
	hl, err := NewHipLib()
	if err != nil {
		slog.Debug(err.Error())
		return nil
	}
	defer hl.Release()

	driverMajor, driverMinor, err := hl.AMDDriverVersion()
	if err != nil {
		// For now this is benign, but we may eventually need to fail compatibility checks
		slog.Debug("error looking up amd driver version", "error", err)
	}

	// Note: the HIP library automatically handles subsetting to any HIP_VISIBLE_DEVICES the user specified
	count := hl.HipGetDeviceCount()
	if count == 0 {
		return nil
	}
	libDir, err := AMDValidateLibDir()
	if err != nil {
		slog.Warn("unable to verify rocm library, will use cpu", "error", err)
		return nil
	}

	var supported []string
	gfxOverride := envconfig.HsaOverrideGfxVersion()
	if gfxOverride == "" {
		supported, err = GetSupportedGFX(libDir)
		if err != nil {
			slog.Warn("failed to lookup supported GFX types, falling back to CPU mode", "error", err)
			return nil
		}
	} else {
		slog.Info("skipping rocm gfx compatibility check", "HSA_OVERRIDE_GFX_VERSION", gfxOverride)
	}

	slog.Debug("detected hip devices", "count", count)
	// TODO how to determine the underlying device ID when visible devices is causing this to subset?
	for i := range count {
		err = hl.HipSetDevice(i)
		if err != nil {
			slog.Warn("set device", "id", i, "error", err)
			continue
		}

		props, err := hl.HipGetDeviceProperties(i)
		if err != nil {
			slog.Warn("get properties", "id", i, "error", err)
			continue
		}
		n := bytes.IndexByte(props.Name[:], 0)
		name := string(props.Name[:n])
		// TODO is UUID actually populated on windows?
		// Can luid be used on windows for setting visible devices (and is it actually set?)
		n = bytes.IndexByte(props.GcnArchName[:], 0)
		gfx := string(props.GcnArchName[:n])
		slog.Debug("hip device", "id", i, "name", name, "gfx", gfx)
		// slog.Info(fmt.Sprintf("[%d] Integrated: %d", i, props.iGPU)) // DOESN'T REPORT CORRECTLY!  Always 0
		// ROCm in WSL does not support iGPU at the moment
		// if strings.EqualFold(name, iGPUName) {
		// 	slog.Info("unsupported Radeon iGPU detected skipping", "id", i, "name", name, "gfx", gfx)
		// 	continue
		// }
		if gfxOverride == "" {
			// Strip off Target Features when comparing
			if !slices.Contains[[]string, string](supported, strings.Split(gfx, ":")[0]) {
				slog.Warn("amdgpu is not supported", "gpu", i, "gpu_type", gfx, "library", libDir, "supported_types", supported)
				// TODO - consider discrete markdown just for ROCM troubleshooting?
				slog.Warn("See https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md for HSA_OVERRIDE_GFX_VERSION usage")
				continue
			} else {
				slog.Debug("amdgpu is supported", "gpu", i, "gpu_type", gfx)
			}
		}

		freeMemory, totalMemory, err := hl.HipMemGetInfo()
		if err != nil {
			slog.Warn("get mem info", "id", i, "error", err)
			continue
		}

		// iGPU detection, remove this check once we can support an iGPU variant of the rocm library
		if totalMemory < IGPUMemLimit {
			slog.Info("amdgpu appears to be an iGPU, skipping", "gpu", i, "total", format.HumanBytes2(totalMemory))
			continue
		}

		slog.Debug("amdgpu memory", "gpu", i, "total", format.HumanBytes2(totalMemory))
		slog.Debug("amdgpu memory", "gpu", i, "available", format.HumanBytes2(freeMemory))
		gpuInfo := RocmGPUInfo{
			GpuInfo: GpuInfo{
				Library: "rocm",
				memInfo: memInfo{
					TotalMemory: totalMemory,
					FreeMemory:  freeMemory,
				},
				// Free memory reporting on Windows is not reliable until we bump to ROCm v6.2
				UnreliableFreeMemory: true,

				ID:             strconv.Itoa(i), // TODO this is probably wrong if we specify visible devices
				DependencyPath: strings.Split(libDir, ""),
				MinimumMemory:  rocmMinimumMemory,
				Name:           name,
				Compute:        gfx,
				DriverMajor:    driverMajor,
				DriverMinor:    driverMinor,
			},
			index: i,
		}

		resp = append(resp, gpuInfo)
	}

	return resp
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

func rocmGetVisibleDevicesEnv(gpuInfo []GpuInfo) (string, string) {
	ids := []string{}
	for _, info := range gpuInfo {
		if info.Library != "rocm" {
			// TODO shouldn't happen if things are wired correctly...
			slog.Debug("rocmGetVisibleDevicesEnv skipping over non-rocm device", "library", info.Library)
			continue
		}
		ids = append(ids, info.ID)
	}
	// There are 3 potential env vars to use to select GPUs.
	// ROCR_VISIBLE_DEVICES supports UUID or numeric so is our preferred on linux
	// GPU_DEVICE_ORDINAL supports numeric IDs only
	// HIP_VISIBLE_DEVICES supports numeric IDs only
	return "ROCR_VISIBLE_DEVICES", strings.Join(ids, ",")
}
