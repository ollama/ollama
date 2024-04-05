package gpu

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const (
	RocmStandardLocation = "C:\\Program Files\\AMD\\ROCm\\5.7\\bin" // TODO glob?

	// TODO  We're lookinng for this exact name to detect iGPUs since hipGetDeviceProperties never reports integrated==true
	iGPUName = "AMD Radeon(TM) Graphics"
)

var (
	// Used to validate if the given ROCm lib is usable
	ROCmLibGlobs = []string{"hipblas.dll", "rocblas"} // TODO - probably include more coverage of files here...
)

func AMDGetGPUInfo(resp *GpuInfo) {
	hl, err := NewHipLib()
	if err != nil {
		slog.Debug(err.Error())
		return
	}
	defer hl.Release()
	skip := map[int]interface{}{}
	ids := []int{}
	resp.memInfo.DeviceCount = 0
	resp.memInfo.TotalMemory = 0
	resp.memInfo.FreeMemory = 0

	ver, err := hl.AMDDriverVersion()
	if err == nil {
		slog.Info("AMD Driver: " + ver)
	} else {
		// For now this is benign, but we may eventually need to fail compatibility checks
		slog.Debug(fmt.Sprintf("error looking up amd driver version: %s", err))
	}

	// Note: the HIP library automatically handles HIP_VISIBLE_DEVICES
	count := hl.HipGetDeviceCount()
	if count == 0 {
		return
	}
	libDir, err := AMDValidateLibDir()
	if err != nil {
		slog.Warn(fmt.Sprintf("unable to verify rocm library, will use cpu: %s", err))
		return
	}

	var supported []string
	gfxOverride := os.Getenv("HSA_OVERRIDE_GFX_VERSION")
	if gfxOverride == "" {
		supported, err = GetSupportedGFX(libDir)
		if err != nil {
			slog.Warn(fmt.Sprintf("failed to lookup supported GFX types, falling back to CPU mode: %s", err))
			return
		}
	} else {
		slog.Debug("skipping rocm gfx compatibility check with HSA_OVERRIDE_GFX_VERSION=" + gfxOverride)
	}

	slog.Info(fmt.Sprintf("detected %d hip devices", count))
	for i := 0; i < count; i++ {
		ids = append(ids, i)
		err = hl.HipSetDevice(i)
		if err != nil {
			slog.Warn(fmt.Sprintf("[%d] %s", i, err))
			skip[i] = struct{}{}
			continue
		}

		props, err := hl.HipGetDeviceProperties(i)
		if err != nil {
			slog.Warn(fmt.Sprintf("[%d] %s", i, err))
			skip[i] = struct{}{}
			continue
		}
		n := bytes.IndexByte(props.Name[:], 0)
		name := string(props.Name[:n])
		slog.Info(fmt.Sprintf("[%d] Name: %s", i, name))
		n = bytes.IndexByte(props.GcnArchName[:], 0)
		gfx := string(props.GcnArchName[:n])
		slog.Info(fmt.Sprintf("[%d] GcnArchName: %s", i, gfx))
		//slog.Info(fmt.Sprintf("[%d] Integrated: %d", i, props.iGPU)) // DOESN'T REPORT CORRECTLY!  Always 0
		// TODO  Why isn't props.iGPU accurate!?
		if strings.EqualFold(name, iGPUName) {
			slog.Info(fmt.Sprintf("iGPU detected [%d] skipping", i))
			skip[i] = struct{}{}
			continue
		}
		if gfxOverride == "" {
			if !slices.Contains[[]string, string](supported, gfx) {
				slog.Warn(fmt.Sprintf("amdgpu [%d] %s is not supported by %s %v", i, gfx, libDir, supported))
				// TODO - consider discrete markdown just for ROCM troubleshooting?
				slog.Warn("See https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md for HSA_OVERRIDE_GFX_VERSION usage")
				skip[i] = struct{}{}
				continue
			} else {
				slog.Info(fmt.Sprintf("amdgpu [%d] %s is supported", i, gfx))
			}
		}

		totalMemory, freeMemory, err := hl.HipMemGetInfo()
		if err != nil {
			slog.Warn(fmt.Sprintf("[%d] %s", i, err))
			continue
		}

		// TODO according to docs, freeMem may lie on windows!
		slog.Info(fmt.Sprintf("[%d] Total Mem: %d", i, totalMemory))
		slog.Info(fmt.Sprintf("[%d] Free Mem:  %d", i, freeMemory))
		resp.memInfo.DeviceCount++
		resp.memInfo.TotalMemory += totalMemory
		resp.memInfo.FreeMemory += freeMemory
	}
	if resp.memInfo.DeviceCount > 0 {
		resp.Library = "rocm"
	}
	// Abort if all GPUs are skipped
	if len(skip) >= count {
		slog.Info("all detected amdgpus are skipped, falling back to CPU")
		return
	}
	if len(skip) > 0 {
		amdSetVisibleDevices(ids, skip)
	}
	UpdatePath(libDir)
}

func AMDValidateLibDir() (string, error) {
	// On windows non-admins typically can't create links
	// so instead of trying to rely on rpath and a link in
	// $LibDir/rocm, we instead rely on setting PATH to point
	// to the location of the ROCm library

	// Installer payload location if we're running the installed binary
	exe, err := os.Executable()
	if err == nil {
		rocmTargetDir := filepath.Join(filepath.Dir(exe), "rocm")
		if rocmLibUsable(rocmTargetDir) {
			slog.Debug("detected ROCM next to ollama executable " + rocmTargetDir)
			return rocmTargetDir, nil
		}
	}

	// Installer payload (if we're running from some other location)
	localAppData := os.Getenv("LOCALAPPDATA")
	appDir := filepath.Join(localAppData, "Programs", "Ollama")
	rocmTargetDir := filepath.Join(appDir, "rocm")
	if rocmLibUsable(rocmTargetDir) {
		slog.Debug("detected ollama installed ROCm at " + rocmTargetDir)
		return rocmTargetDir, nil
	}

	// Prefer explicit HIP env var
	hipPath := os.Getenv("HIP_PATH")
	if hipPath != "" {
		hipLibDir := filepath.Join(hipPath, "bin")
		if rocmLibUsable(hipLibDir) {
			slog.Debug("detected ROCM via HIP_PATH=" + hipPath)
			return hipLibDir, nil
		}
	}

	// Well known location(s)
	if rocmLibUsable(RocmStandardLocation) {
		return RocmStandardLocation, nil
	}

	// Should not happen on windows since we include it in the installer, but stand-alone binary might hit this
	slog.Warn("amdgpu detected, but no compatible rocm library found.  Please install ROCm")
	return "", fmt.Errorf("no suitable rocm found, falling back to CPU")
}
