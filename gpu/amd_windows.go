package gpu

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/format"
)

const (

	// TODO  We're lookinng for this exact name to detect iGPUs since hipGetDeviceProperties never reports integrated==true
	iGPUName = "AMD Radeon(TM) Graphics"
)

var (
	// Used to validate if the given ROCm lib is usable
	ROCmLibGlobs          = []string{"hipblas.dll", "rocblas"}                 // TODO - probably include more coverage of files here...
	RocmStandardLocations = []string{"C:\\Program Files\\AMD\\ROCm\\5.7\\bin"} // TODO glob?
)

func AMDGetGPUInfo() []RocmGPUInfo {
	resp := []RocmGPUInfo{}
	hl, err := NewHipLib()
	if err != nil {
		slog.Debug(err.Error())
		return nil
	}
	defer hl.Release()

	// TODO - this reports incorrect version information, so omitting for now
	// driverMajor, driverMinor, err := hl.AMDDriverVersion()
	// if err != nil {
	// 	// For now this is benign, but we may eventually need to fail compatibility checks
	// 	slog.Debug("error looking up amd driver version", "error", err)
	// }

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
	gfxOverride := os.Getenv("HSA_OVERRIDE_GFX_VERSION")
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
		//slog.Info(fmt.Sprintf("[%d] Integrated: %d", i, props.iGPU)) // DOESN'T REPORT CORRECTLY!  Always 0
		// TODO  Why isn't props.iGPU accurate!?
		if strings.EqualFold(name, iGPUName) {
			slog.Info("unsupported Radeon iGPU detected skipping", "id", i, "name", name, "gfx", gfx)
			continue
		}
		if gfxOverride == "" {
			if !slices.Contains[[]string, string](supported, gfx) {
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

		// TODO revisit this once ROCm v6 is available on windows.
		// v5.7 only reports VRAM used by this process, so it's completely wrong and unusable
		slog.Debug("amdgpu memory", "gpu", i, "total", format.HumanBytes2(totalMemory))
		slog.Debug("amdgpu memory", "gpu", i, "available", format.HumanBytes2(freeMemory))
		gpuInfo := RocmGPUInfo{
			GpuInfo: GpuInfo{
				Library: "rocm",
				memInfo: memInfo{
					TotalMemory: totalMemory,
					FreeMemory:  freeMemory,
				},
				ID:             strconv.Itoa(i), // TODO this is probably wrong if we specify visible devices
				DependencyPath: libDir,
				MinimumMemory:  rocmMinimumMemory,
				Name:           name,
				Compute:        gfx,

				// TODO - this information isn't accurate on windows, so don't report it until we find the right way to retrieve
				// DriverMajor:    driverMajor,
				// DriverMinor:    driverMinor,
			},
			index: i,
		}

		resp = append(resp, gpuInfo)
	}

	return resp
}

func AMDValidateLibDir() (string, error) {
	libDir, err := commonAMDValidateLibDir()
	if err == nil {
		return libDir, nil
	}

	// Installer payload (if we're running from some other location)
	localAppData := os.Getenv("LOCALAPPDATA")
	appDir := filepath.Join(localAppData, "Programs", "Ollama")
	rocmTargetDir := filepath.Join(appDir, "rocm")
	if rocmLibUsable(rocmTargetDir) {
		slog.Debug("detected ollama installed ROCm at " + rocmTargetDir)
		return rocmTargetDir, nil
	}

	// Should not happen on windows since we include it in the installer, but stand-alone binary might hit this
	slog.Warn("amdgpu detected, but no compatible rocm library found.  Please install ROCm")
	return "", fmt.Errorf("no suitable rocm found, falling back to CPU")
}

func (gpus RocmGPUInfoList) RefreshFreeMemory() error {
	if len(gpus) == 0 {
		return nil
	}
	hl, err := NewHipLib()
	if err != nil {
		slog.Debug(err.Error())
		return nil
	}
	defer hl.Release()

	for i := range gpus {
		err := hl.HipSetDevice(gpus[i].index)
		if err != nil {
			return err
		}
		freeMemory, _, err := hl.HipMemGetInfo()
		if err != nil {
			slog.Warn("get mem info", "id", i, "error", err)
			continue
		}
		slog.Debug("updating rocm free memory", "gpu", gpus[i].ID, "name", gpus[i].Name, "before", format.HumanBytes2(gpus[i].FreeMemory), "now", format.HumanBytes2(freeMemory))
		gpus[i].FreeMemory = freeMemory
	}
	return nil
}
