package discover

// GPU discovery via llama-server --list-devices

import (
	"context"
	"log/slog"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

var (
	deviceMu     sync.Mutex
	devices      []ml.DeviceInfo
	libDirs      map[string]struct{}
	bootstrapped bool
)

func GPUDevices(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
	deviceMu.Lock()
	defer deviceMu.Unlock()
	startDiscovery := time.Now()
	msg := "overall device VRAM discovery took"
	defer func() {
		slog.Debug(msg, "duration", time.Since(startDiscovery))
	}()

	if !bootstrapped {
		msg = "GPU bootstrap discovery took"
		libDirs = make(map[string]struct{})
		files, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
		if err != nil {
			slog.Debug("unable to lookup GPU backend directories", "error", err)
		}
		for _, file := range files {
			libDirs[filepath.Dir(file)] = struct{}{}
		}

		if len(libDirs) == 0 {
			libDirs[""] = struct{}{}
		}

		slog.Info("discovering available GPUs...")
		detectIncompatibleLibraries()
		detectOldAMDDriverWindows()
		overrideWarnings()

		requested := envconfig.LLMLibrary()
		jetpack := cudaJetpack()

		// Discover GPUs through each detected GPU backend library.
		// llama-server --list-devices enumerates devices for the loaded backend.
		for dir := range libDirs {
			bootstrapTimeout := 30 * time.Second
			if runtime.GOOS == "windows" {
				// Windows Defender AV scanning of DLLs can be slow on first load
				bootstrapTimeout = 90 * time.Second
			}
			var dirs []string
			if dir != "" {
				if requested != "" && !strings.HasPrefix(requested, "mlx_") && filepath.Base(dir) != requested {
					slog.Debug("skipping available library at user's request", "requested", requested, "libDir", dir)
					continue
				} else if jetpack != "" && filepath.Base(dir) != "cuda_"+jetpack {
					continue
				} else if jetpack == "" && strings.Contains(filepath.Base(dir), "cuda_jetpack") {
					slog.Debug("jetpack not detected (set JETSON_JETPACK or OLLAMA_LLM_LIBRARY to override), skipping", "libDir", dir)
					continue
				} else if !envconfig.EnableVulkan() && strings.Contains(filepath.Base(dir), "vulkan") {
					slog.Info("experimental Vulkan support disabled.  To enable, set OLLAMA_VULKAN=1")
					continue
				}
				dirs = []string{ml.LibOllamaPath, dir}
			} else {
				dirs = []string{ml.LibOllamaPath}
			}

			ctx1stPass, cancel := context.WithTimeout(ctx, bootstrapTimeout)
			defer cancel()
			devices = append(devices, bootstrapDevices(ctx1stPass, dirs, nil)...)
		}

		// Filter duplicate devices:
		// - Same backend (e.g., cuda_v12 and cuda_v13 both see same GPU): keep newer version
		// - Different backends (e.g., CUDA and Vulkan see same GPU): prefer CUDA/HIP
		for i := 0; i < len(devices); i++ {
			for j := i + 1; j < len(devices); j++ {
				switch devices[i].Compare(devices[j]) {
				case ml.SameBackendDevice:
					// Same library, different version — keep the better one
					if devices[i].IsBetter(devices[j]) {
						devices[i] = devices[j]
					}
					devices = append(devices[:j], devices[j+1:]...)
					j--
					continue
				case ml.DuplicateDevice:
					var droppedDevice ml.DeviceInfo
					if devices[i].PreferredLibrary(devices[j]) {
						droppedDevice = devices[j]
					} else {
						droppedDevice = devices[i]
						devices[i] = devices[j]
					}
					devices = append(devices[:j], devices[j+1:]...)
					j--

					typeStr := "discrete"
					if droppedDevice.Integrated {
						typeStr = "iGPU"
					}
					slog.Debug("dropping duplicate device",
						"id", droppedDevice.ID,
						"library", droppedDevice.Library,
						"name", droppedDevice.Name,
						"pci_id", droppedDevice.PCIID,
						"type", typeStr,
						"total", format.HumanBytes2(droppedDevice.TotalMemory),
					)
					continue
				}
			}
		}

		// Renumber device IDs after filtering
		postFilteredID := map[string]int{}
		for i := range devices {
			if _, ok := postFilteredID[devices[i].Library]; !ok {
				postFilteredID[devices[i].Library] = 0
			}
			if _, err := strconv.Atoi(devices[i].ID); err == nil {
				devices[i].FilterID = devices[i].ID
				devices[i].ID = strconv.Itoa(postFilteredID[devices[i].Library])
			}
			postFilteredID[devices[i].Library]++
		}

		// Record which lib dirs are actually in use for VRAM refresh
		libDirs = make(map[string]struct{})
		for _, dev := range devices {
			dir := dev.LibraryPath[len(dev.LibraryPath)-1]
			if dir != ml.LibOllamaPath {
				libDirs[dir] = struct{}{}
			}
		}
		if len(libDirs) == 0 {
			libDirs[""] = struct{}{}
		}

		bootstrapped = true
	} else {
		if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
			// Metal never updates free VRAM
			return append([]ml.DeviceInfo{}, devices...)
		}

		// Refresh free memory from running llama-server instances if possible,
		// otherwise re-run llama-server --list-devices.
		slog.Debug("refreshing free memory")
		updated := make([]bool, len(devices))
		for _, runner := range runners {
			if runner == nil {
				continue
			}
			deviceIDs := runner.GetActiveDeviceIDs()
			if len(deviceIDs) == 0 {
				continue
			}

			skip := true
			for _, dev := range deviceIDs {
				for i := range devices {
					if dev == devices[i].DeviceID && !updated[i] {
						skip = false
						break
					}
				}
				if !skip {
					break
				}
			}
			if skip {
				continue
			}

			rctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			for _, u := range runner.GetDeviceInfos(rctx) {
				for i := range devices {
					if u.DeviceID == devices[i].DeviceID {
						updated[i] = true
						devices[i].FreeMemory = u.FreeMemory
						break
					}
				}
			}
		}

		// Fall back to bootstrap discovery for any devices not refreshed
		allDone := true
		for _, done := range updated {
			if !done {
				allDone = false
				break
			}
		}
		if !allDone {
			slog.Debug("refreshing remaining GPUs via llama-server --list-devices")
			rctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			devFilter := ml.GetVisibleDevicesEnv(devices, false)
			for dir := range libDirs {
				for _, u := range bootstrapDevices(rctx, []string{ml.LibOllamaPath, dir}, devFilter) {
					for i := range devices {
						if u.DeviceID == devices[i].DeviceID && u.PCIID == devices[i].PCIID {
							updated[i] = true
							devices[i].FreeMemory = u.FreeMemory
							break
						}
					}
				}
			}
		}
	}

	return append([]ml.DeviceInfo{}, devices...)
}

func bootstrapDevices(ctx context.Context, ollamaLibDirs []string, extraEnvs map[string]string) []ml.DeviceInfo {
	return llamaServerBootstrapDevices(ctx, ollamaLibDirs, extraEnvs)
}

func overrideWarnings() {
	anyFound := false
	m := envconfig.AsMap()
	for _, k := range []string{
		"CUDA_VISIBLE_DEVICES",
		"HIP_VISIBLE_DEVICES",
		"ROCR_VISIBLE_DEVICES",
		"GGML_VK_VISIBLE_DEVICES",
		"GPU_DEVICE_ORDINAL",
		"HSA_OVERRIDE_GFX_VERSION",
	} {
		if e, found := m[k]; found && e.Value != "" {
			anyFound = true
			slog.Warn("user overrode visible devices", k, e.Value)
		}
	}
	if anyFound {
		slog.Warn("if GPUs are not correctly discovered, unset and try again")
	}
}

func detectIncompatibleLibraries() {
	if runtime.GOOS != "windows" {
		return
	}
	basePath, err := exec.LookPath("ggml-base.dll")
	if err != nil || basePath == "" {
		return
	}
	if !strings.HasPrefix(basePath, ml.LibOllamaPath) {
		slog.Warn("potentially incompatible library detected in PATH", "location", basePath)
	}
}

func detectOldAMDDriverWindows() {
	if runtime.GOOS != "windows" {
		return
	}
	_, errV6 := exec.LookPath("amdhip64_6.dll")
	_, errV7 := exec.LookPath("amdhip64_7.dll")
	if errV6 == nil && errV7 != nil {
		slog.Warn("AMD driver is too old. Update your AMD driver to enable GPU inference.")
	}
}
