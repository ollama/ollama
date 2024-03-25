package gpu

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// Discovery logic for AMD/ROCm GPUs

const (
	DriverVersionFile     = "/sys/module/amdgpu/version"
	AMDNodesSysfsDir      = "/sys/class/kfd/kfd/topology/nodes/"
	GPUPropertiesFileGlob = AMDNodesSysfsDir + "*/properties"

	// Prefix with the node dir
	GPUTotalMemoryFileGlob = "mem_banks/*/properties" // size_in_bytes line
	GPUUsedMemoryFileGlob  = "mem_banks/*/used_memory"
	RocmStandardLocation   = "/opt/rocm/lib"

	// TODO find a better way to detect iGPU instead of minimum memory
	IGPUMemLimit = 1024 * 1024 * 1024 // 512G is what they typically report, so anything less than 1G must be iGPU
)

var (
	// Used to validate if the given ROCm lib is usable
	ROCmLibGlobs = []string{"libhipblas.so.2*", "rocblas"} // TODO - probably include more coverage of files here...
)

// Gather GPU information from the amdgpu driver if any supported GPUs are detected
// HIP_VISIBLE_DEVICES will be set if we detect a mix of unsupported and supported devices
// and the user hasn't already set this variable
func AMDGetGPUInfo(resp *GpuInfo) {
	// TODO - DRY this out with windows
	if !AMDDetected() {
		return
	}
	skip := map[int]interface{}{}

	// Opportunistic logging of driver version to aid in troubleshooting
	ver, err := AMDDriverVersion()
	if err == nil {
		slog.Info("AMD Driver: " + ver)
	} else {
		// TODO - if we see users crash and burn with the upstreamed kernel this can be adjusted to hard-fail rocm support and fallback to CPU
		slog.Warn(fmt.Sprintf("ollama recommends running the https://www.amd.com/en/support/linux-drivers: %s", err))
	}

	// If the user has specified exactly which GPUs to use, look up their memory
	visibleDevices := os.Getenv("HIP_VISIBLE_DEVICES")
	if visibleDevices != "" {
		ids := []int{}
		for _, idStr := range strings.Split(visibleDevices, ",") {
			id, err := strconv.Atoi(idStr)
			if err != nil {
				slog.Warn(fmt.Sprintf("malformed HIP_VISIBLE_DEVICES=%s %s", visibleDevices, err))
			} else {
				ids = append(ids, id)
			}
		}
		amdProcMemLookup(resp, nil, ids)
		return
	}

	// Gather GFX version information from all detected cards
	gfx := AMDGFXVersions()
	verStrings := []string{}
	for i, v := range gfx {
		verStrings = append(verStrings, v.ToGFXString())
		if v.Major == 0 {
			// Silently skip CPUs
			skip[i] = struct{}{}
			continue
		}
		if v.Major < 9 {
			// TODO consider this a build-time setting if we can support 8xx family GPUs
			slog.Warn(fmt.Sprintf("amdgpu [%d] too old %s", i, v.ToGFXString()))
			skip[i] = struct{}{}
		}
	}
	slog.Info(fmt.Sprintf("detected amdgpu versions %v", verStrings))

	// Abort if all GPUs are skipped
	if len(skip) >= len(gfx) {
		slog.Info("all detected amdgpus are skipped, falling back to CPU")
		return
	}

	// If we got this far, then we have at least 1 GPU that's a ROCm candidate, so make sure we have a lib
	libDir, err := AMDValidateLibDir()
	if err != nil {
		slog.Warn(fmt.Sprintf("unable to verify rocm library, will use cpu: %s", err))
		return
	}

	gfxOverride := os.Getenv("HSA_OVERRIDE_GFX_VERSION")
	if gfxOverride == "" {
		supported, err := GetSupportedGFX(libDir)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to lookup supported GFX types, falling back to CPU mode: %s", err))
			return
		}
		slog.Debug(fmt.Sprintf("rocm supported GPU types %v", supported))

		for i, v := range gfx {
			if !slices.Contains[[]string, string](supported, v.ToGFXString()) {
				slog.Warn(fmt.Sprintf("amdgpu [%d] %s is not supported by %s %v", i, v.ToGFXString(), libDir, supported))
				// TODO - consider discrete markdown just for ROCM troubleshooting?
				slog.Warn("See https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md for HSA_OVERRIDE_GFX_VERSION usage")
				skip[i] = struct{}{}
			} else {
				slog.Info(fmt.Sprintf("amdgpu [%d] %s is supported", i, v.ToGFXString()))
			}
		}
	} else {
		slog.Debug("skipping rocm gfx compatibility check with HSA_OVERRIDE_GFX_VERSION=" + gfxOverride)
	}

	if len(skip) >= len(gfx) {
		slog.Info("all detected amdgpus are skipped, falling back to CPU")
		return
	}

	ids := make([]int, len(gfx))
	i := 0
	for k := range gfx {
		ids[i] = k
		i++
	}
	amdProcMemLookup(resp, skip, ids)
	if resp.memInfo.DeviceCount == 0 {
		return
	}
	if len(skip) > 0 {
		amdSetVisibleDevices(ids, skip)
	}
}

// Walk the sysfs nodes for the available GPUs and gather information from them
// skipping over any devices in the skip map
func amdProcMemLookup(resp *GpuInfo, skip map[int]interface{}, ids []int) {
	resp.memInfo.DeviceCount = 0
	resp.memInfo.TotalMemory = 0
	resp.memInfo.FreeMemory = 0
	slog.Debug("discovering VRAM for amdgpu devices")
	if len(ids) == 0 {
		entries, err := os.ReadDir(AMDNodesSysfsDir)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to read amdgpu sysfs %s - %s", AMDNodesSysfsDir, err))
			return
		}
		for _, node := range entries {
			if !node.IsDir() {
				continue
			}
			id, err := strconv.Atoi(node.Name())
			if err != nil {
				slog.Warn("malformed amdgpu sysfs node id " + node.Name())
				continue
			}
			ids = append(ids, id)
		}
	}
	slog.Debug(fmt.Sprintf("amdgpu devices %v", ids))

	for _, id := range ids {
		if _, skipped := skip[id]; skipped {
			continue
		}
		totalMemory := uint64(0)
		usedMemory := uint64(0)
		// Adjust for sysfs vs HIP ids
		propGlob := filepath.Join(AMDNodesSysfsDir, strconv.Itoa(id+1), GPUTotalMemoryFileGlob)
		propFiles, err := filepath.Glob(propGlob)
		if err != nil {
			slog.Warn(fmt.Sprintf("error looking up total GPU memory: %s %s", propGlob, err))
		}
		// 1 or more memory banks - sum the values of all of them
		for _, propFile := range propFiles {
			fp, err := os.Open(propFile)
			if err != nil {
				slog.Warn(fmt.Sprintf("failed to open sysfs node file %s: %s", propFile, err))
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
			slog.Warn(fmt.Sprintf("amdgpu [%d] reports zero total memory, skipping", id))
			skip[id] = struct{}{}
			continue
		}
		if totalMemory < IGPUMemLimit {
			slog.Info(fmt.Sprintf("amdgpu [%d] appears to be an iGPU with %dM reported total memory, skipping", id, totalMemory/1024/1024))
			skip[id] = struct{}{}
			continue
		}
		usedGlob := filepath.Join(AMDNodesSysfsDir, strconv.Itoa(id), GPUUsedMemoryFileGlob)
		usedFiles, err := filepath.Glob(usedGlob)
		if err != nil {
			slog.Warn(fmt.Sprintf("error looking up used GPU memory: %s %s", usedGlob, err))
			continue
		}
		for _, usedFile := range usedFiles {
			fp, err := os.Open(usedFile)
			if err != nil {
				slog.Warn(fmt.Sprintf("failed to open sysfs node file %s: %s", usedFile, err))
				continue
			}
			defer fp.Close()
			data, err := io.ReadAll(fp)
			if err != nil {
				slog.Warn(fmt.Sprintf("failed to read sysfs node file %s: %s", usedFile, err))
				continue
			}
			used, err := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			if err != nil {
				slog.Warn(fmt.Sprintf("malformed used memory %s: %s", string(data), err))
				continue
			}
			usedMemory += used
		}
		slog.Info(fmt.Sprintf("[%d] amdgpu totalMemory %dM", id, totalMemory/1024/1024))
		slog.Info(fmt.Sprintf("[%d] amdgpu freeMemory  %dM", id, (totalMemory-usedMemory)/1024/1024))
		resp.memInfo.DeviceCount++
		resp.memInfo.TotalMemory += totalMemory
		resp.memInfo.FreeMemory += (totalMemory - usedMemory)
	}
	if resp.memInfo.DeviceCount > 0 {
		resp.Library = "rocm"
	}
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
		slog.Debug(fmt.Sprintf("error looking up amd driver %s %s", sysfsDir, err))
		return false
	}
	return true
}

func setupLink(source, target string) error {
	if err := os.RemoveAll(target); err != nil {
		return fmt.Errorf("failed to remove old rocm directory %s %w", target, err)
	}
	if err := os.Symlink(source, target); err != nil {
		return fmt.Errorf("failed to create link %s => %s %w", source, target, err)
	}
	slog.Debug(fmt.Sprintf("host rocm linked %s => %s", source, target))
	return nil
}

// Ensure the AMD rocm lib dir is wired up
// Prefer to use host installed ROCm, as long as it meets our minimum requirements
// failing that, tell the user how to download it on their own
func AMDValidateLibDir() (string, error) {
	// We rely on the rpath compiled into our library to find rocm
	// so we establish a symlink to wherever we find it on the system
	// to <payloads>/rocm
	payloadsDir, err := PayloadsDir()
	if err != nil {
		return "", err
	}

	// If we already have a rocm dependency wired, nothing more to do
	rocmTargetDir := filepath.Clean(filepath.Join(payloadsDir, "..", "rocm"))
	if rocmLibUsable(rocmTargetDir) {
		return rocmTargetDir, nil
	}

	// next to the running binary
	exe, err := os.Executable()
	if err == nil {
		peerDir := filepath.Dir(exe)
		if rocmLibUsable(peerDir) {
			slog.Debug("detected ROCM next to ollama executable " + peerDir)
			return rocmTargetDir, setupLink(peerDir, rocmTargetDir)
		}
		peerDir = filepath.Join(filepath.Dir(exe), "rocm")
		if rocmLibUsable(peerDir) {
			slog.Debug("detected ROCM next to ollama executable " + peerDir)
			return rocmTargetDir, setupLink(peerDir, rocmTargetDir)
		}
	}

	// Well known ollama installer path
	installedRocmDir := "/usr/share/ollama/lib/rocm"
	if rocmLibUsable(installedRocmDir) {
		return rocmTargetDir, setupLink(installedRocmDir, rocmTargetDir)
	}

	// Prefer explicit HIP env var
	hipPath := os.Getenv("HIP_PATH")
	if hipPath != "" {
		hipLibDir := filepath.Join(hipPath, "lib")
		if rocmLibUsable(hipLibDir) {
			slog.Debug("detected ROCM via HIP_PATH=" + hipPath)
			return rocmTargetDir, setupLink(hipLibDir, rocmTargetDir)
		}
	}

	// Scan the library path for potential matches
	ldPaths := strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":")
	for _, ldPath := range ldPaths {
		d, err := filepath.Abs(ldPath)
		if err != nil {
			continue
		}
		if rocmLibUsable(d) {
			return rocmTargetDir, setupLink(d, rocmTargetDir)
		}
	}

	// Well known location(s)
	if rocmLibUsable("/opt/rocm/lib") {
		return rocmTargetDir, setupLink("/opt/rocm/lib", rocmTargetDir)
	}

	// If we still haven't found a usable rocm, the user will have to install it on their own
	slog.Warn("amdgpu detected, but no compatible rocm library found.  Either install rocm v6, or follow manual install instructions at https://github.com/ollama/ollama/blob/main/docs/linux.md#manual-install")
	return "", fmt.Errorf("no suitable rocm found, falling back to CPU")
}

func AMDDriverVersion() (string, error) {
	_, err := os.Stat(DriverVersionFile)
	if err != nil {
		return "", fmt.Errorf("amdgpu version file missing: %s %w", DriverVersionFile, err)
	}
	fp, err := os.Open(DriverVersionFile)
	if err != nil {
		return "", err
	}
	defer fp.Close()
	verString, err := io.ReadAll(fp)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(verString)), nil
}

func AMDGFXVersions() map[int]Version {
	// The amdgpu driver always exposes the host CPU as node 0, but we have to skip that and subtract one
	// from the other IDs to get alignment with the HIP libraries expectations (zero is the first GPU, not the CPU)
	res := map[int]Version{}
	matches, _ := filepath.Glob(GPUPropertiesFileGlob)
	for _, match := range matches {
		fp, err := os.Open(match)
		if err != nil {
			slog.Debug(fmt.Sprintf("failed to open sysfs node file %s: %s", match, err))
			continue
		}
		defer fp.Close()
		i, err := strconv.Atoi(filepath.Base(filepath.Dir(match)))
		if err != nil {
			slog.Debug(fmt.Sprintf("failed to parse node ID %s", err))
			continue
		}

		if i == 0 {
			// Skipping the CPU
			continue
		}
		// Align with HIP IDs (zero is first GPU, not CPU)
		i -= 1

		scanner := bufio.NewScanner(fp)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if strings.HasPrefix(line, "gfx_target_version") {
				ver := strings.Fields(line)
				if len(ver) != 2 || len(ver[1]) < 5 {
					if ver[1] != "0" {
						slog.Debug("malformed " + line)
					}
					res[i] = Version{
						Major: 0,
						Minor: 0,
						Patch: 0,
					}
					continue
				}
				l := len(ver[1])
				patch, err1 := strconv.ParseUint(ver[1][l-2:l], 10, 32)
				minor, err2 := strconv.ParseUint(ver[1][l-4:l-2], 10, 32)
				major, err3 := strconv.ParseUint(ver[1][:l-4], 10, 32)
				if err1 != nil || err2 != nil || err3 != nil {
					slog.Debug("malformed int " + line)
					continue
				}

				res[i] = Version{
					Major: uint(major),
					Minor: uint(minor),
					Patch: uint(patch),
				}
			}
		}
	}
	return res
}

func (v Version) ToGFXString() string {
	return fmt.Sprintf("gfx%d%d%d", v.Major, v.Minor, v.Patch)
}
