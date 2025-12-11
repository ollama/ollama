package discover

// Runner based GPU discovery

import (
	"context"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

var (
	deviceMu     sync.Mutex
	devices      []ml.DeviceInfo
	libDirs      map[string]struct{}
	exe          string
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
		var err error
		exe, err = os.Executable()
		if err != nil {
			slog.Error("unable to lookup executable path", "error", err)
			return nil
		}
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		files, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
		if err != nil {
			slog.Debug("unable to lookup runner library directories", "error", err)
		}
		for _, file := range files {
			libDirs[filepath.Dir(file)] = struct{}{}
		}

		if len(libDirs) == 0 {
			libDirs[""] = struct{}{}
		}

		slog.Info("discovering available GPUs...")
		detectIncompatibleLibraries()

		// Warn if any user-overrides are set which could lead to incorrect GPU discovery
		overrideWarnings()

		requested := envconfig.LLMLibrary()
		jetpack := cudaJetpack()

		// For our initial discovery pass, we gather all the known GPUs through
		// all the libraries that were detected. This pass may include GPUs that
		// are enumerated, but not actually supported.
		// We run this in serial to avoid potentially initializing a GPU multiple
		// times concurrently leading to memory contention
		for dir := range libDirs {
			// Typically bootstrapping takes < 1s, but on some systems, with devices
			// in low power/idle mode, initialization can take multiple seconds.  We
			// set a longer timeout just for bootstrap discovery to reduce the chance
			// of giving up too quickly
			bootstrapTimeout := 30 * time.Second
			if runtime.GOOS == "windows" {
				// On Windows with Defender enabled, AV scanning of the DLLs
				// takes place sequentially and this can significantly increase
				// the time it takes too do the initial discovery pass.
				// Subsequent loads will be faster as the scan results are
				// cached
				bootstrapTimeout = 90 * time.Second
			}
			var dirs []string
			if dir != "" {
				if requested != "" && filepath.Base(dir) != requested {
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

			// For this pass, we retain duplicates in case any are incompatible with some libraries
			devices = append(devices, bootstrapDevices(ctx1stPass, dirs, nil)...)
		}

		// In the second pass, we more deeply initialize the GPUs to weed out devices that
		// aren't supported by a given library.  We run this phase in parallel to speed up discovery.
		// Only devices that need verification are included in this pass
		slog.Debug("evaluating which, if any, devices to filter out", "initial_count", len(devices))
		ctx2ndPass, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		var wg sync.WaitGroup
		needsDelete := make([]bool, len(devices))
		supportedMu := sync.Mutex{}
		supported := make(map[string]map[string]map[string]int) // [Library][libDir][ID] = pre-deletion devices index
		for i := range devices {
			libDir := devices[i].LibraryPath[len(devices[i].LibraryPath)-1]
			if !devices[i].NeedsInitValidation() {
				// No need to validate, add to the supported map
				supportedMu.Lock()
				if _, ok := supported[devices[i].Library]; !ok {
					supported[devices[i].Library] = make(map[string]map[string]int)
				}
				if _, ok := supported[devices[i].Library][libDir]; !ok {
					supported[devices[i].Library][libDir] = make(map[string]int)
				}
				supported[devices[i].Library][libDir][devices[i].ID] = i
				supportedMu.Unlock()
				continue
			}
			slog.Debug("verifying if device is supported", "library", libDir, "description", devices[i].Description, "compute", devices[i].Compute(), "id", devices[i].ID, "pci_id", devices[i].PCIID)
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				extraEnvs := ml.GetVisibleDevicesEnv(devices[i:i+1], true)
				devices[i].AddInitValidation(extraEnvs)
				if len(bootstrapDevices(ctx2ndPass, devices[i].LibraryPath, extraEnvs)) == 0 {
					slog.Debug("filtering device which didn't fully initialize",
						"id", devices[i].ID,
						"libdir", devices[i].LibraryPath[len(devices[i].LibraryPath)-1],
						"pci_id", devices[i].PCIID,
						"library", devices[i].Library,
					)
					needsDelete[i] = true
				} else {
					supportedMu.Lock()
					if _, ok := supported[devices[i].Library]; !ok {
						supported[devices[i].Library] = make(map[string]map[string]int)
					}
					if _, ok := supported[devices[i].Library][libDir]; !ok {
						supported[devices[i].Library][libDir] = make(map[string]int)
					}
					supported[devices[i].Library][libDir][devices[i].ID] = i
					supportedMu.Unlock()
				}
			}(i)
		}
		wg.Wait()
		logutil.Trace("supported GPU library combinations before filtering", "supported", supported)

		// Mark for deletion any overlaps - favoring the library version that can cover all GPUs if possible
		filterOverlapByLibrary(supported, needsDelete)

		// Any Libraries that utilize numeric IDs need adjusting based on any possible filtering taking place
		postFilteredID := map[string]int{}
		for i := 0; i < len(needsDelete); i++ {
			if needsDelete[i] {
				logutil.Trace("removing unsupported or overlapping GPU combination", "libDir", devices[i].LibraryPath[len(devices[i].LibraryPath)-1], "description", devices[i].Description, "compute", devices[i].Compute(), "pci_id", devices[i].PCIID)
				devices = append(devices[:i], devices[i+1:]...)
				needsDelete = append(needsDelete[:i], needsDelete[i+1:]...)
				i--
			} else {
				if _, ok := postFilteredID[devices[i].Library]; !ok {
					postFilteredID[devices[i].Library] = 0
				}
				if _, err := strconv.Atoi(devices[i].ID); err == nil {
					// Replace the numeric ID with the post-filtered IDs
					slog.Debug("adjusting filtering IDs", "FilterID", devices[i].ID, "new_ID", strconv.Itoa(postFilteredID[devices[i].Library]))
					devices[i].FilterID = devices[i].ID
					devices[i].ID = strconv.Itoa(postFilteredID[devices[i].Library])
				}
				postFilteredID[devices[i].Library]++
			}
		}

		// Now filter out any overlap with different libraries (favor CUDA/HIP over others)
		for i := 0; i < len(devices); i++ {
			for j := i + 1; j < len(devices); j++ {
				// For this pass, we only drop exact duplicates
				switch devices[i].Compare(devices[j]) {
				case ml.SameBackendDevice:
					// Same library and device, skip it
					devices = append(devices[:j], devices[j+1:]...)
					j--
					continue
				case ml.DuplicateDevice:
					// Different library, choose based on priority
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
						"compute", droppedDevice.Compute(),
						"name", droppedDevice.Name,
						"description", droppedDevice.Description,
						"libdirs", strings.Join(droppedDevice.LibraryPath, ","),
						"driver", droppedDevice.Driver(),
						"pci_id", droppedDevice.PCIID,
						"type", typeStr,
						"total", format.HumanBytes2(droppedDevice.TotalMemory),
						"available", format.HumanBytes2(droppedDevice.FreeMemory),
					)
					continue
				}
			}
		}

		// Reset the libDirs to what we actually wind up using for future refreshes
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
			// metal never updates free VRAM
			return append([]ml.DeviceInfo{}, devices...)
		}

		slog.Debug("refreshing free memory")
		updated := make([]bool, len(devices))
		allDone := func() bool {
			allDone := true
			for _, done := range updated {
				if !done {
					allDone = false
					break
				}
			}
			return allDone
		}

		// First try to use existing runners to refresh VRAM since they're already
		// active on GPU(s)
		for _, runner := range runners {
			if runner == nil {
				continue
			}
			deviceIDs := runner.GetActiveDeviceIDs()
			if len(deviceIDs) == 0 {
				// Skip this runner since it doesn't have active GPU devices
				continue
			}

			// Check to see if this runner is active on any devices that need a refresh
			skip := true
		devCheck:
			for _, dev := range deviceIDs {
				for i := range devices {
					if dev == devices[i].DeviceID {
						if !updated[i] {
							skip = false
							break devCheck
						}
					}
				}
			}
			if skip {
				continue
			}

			// Typical refresh on existing runner is ~500ms but allow longer if the system
			// is under stress before giving up and using stale data.
			ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			start := time.Now()
			updatedDevices := runner.GetDeviceInfos(ctx)
			slog.Debug("existing runner discovery took", "duration", time.Since(start))
			for _, u := range updatedDevices {
				for i := range devices {
					if u.DeviceID == devices[i].DeviceID {
						updated[i] = true
						devices[i].FreeMemory = u.FreeMemory
						break
					}
				}
			}
			// Short circuit if we've updated all the devices
			if allDone() {
				break
			}
		}
		if !allDone() {
			slog.Debug("unable to refresh all GPUs with existing runners, performing bootstrap discovery")

			// Bootstrapping may take longer in some cases (AMD windows), but we
			// would rather use stale free data to get the model running sooner
			ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()

			// Apply any dev filters to avoid re-discovering unsupported devices, and get IDs correct
			// We avoid CUDA filters here to keep ROCm from failing to discover GPUs in a mixed environment
			devFilter := ml.GetVisibleDevicesEnv(devices, false)

			for dir := range libDirs {
				updatedDevices := bootstrapDevices(ctx, []string{ml.LibOllamaPath, dir}, devFilter)
				for _, u := range updatedDevices {
					for i := range devices {
						if u.DeviceID == devices[i].DeviceID && u.PCIID == devices[i].PCIID {
							updated[i] = true
							devices[i].FreeMemory = u.FreeMemory
							break
						}
					}
					// TODO - consider evaluating if new devices have appeared (e.g. hotplug)
				}
				if allDone() {
					break
				}
			}
			if !allDone() {
				slog.Warn("unable to refresh free memory, using old values")
			}
		}
	}

	return append([]ml.DeviceInfo{}, devices...)
}

func filterOverlapByLibrary(supported map[string]map[string]map[string]int, needsDelete []bool) {
	// For multi-GPU systems, use the newest version that supports all the GPUs
	for _, byLibDirs := range supported {
		libDirs := make([]string, 0, len(byLibDirs))
		for libDir := range byLibDirs {
			libDirs = append(libDirs, libDir)
		}
		sort.Sort(sort.Reverse(sort.StringSlice(libDirs)))
		anyMissing := false
		var newest string
		for _, newest = range libDirs {
			for _, libDir := range libDirs {
				if libDir == newest {
					continue
				}
				if len(byLibDirs[newest]) != len(byLibDirs[libDir]) {
					anyMissing = true
					break
				}
				for dev := range byLibDirs[newest] {
					if _, found := byLibDirs[libDir][dev]; !found {
						anyMissing = true
						break
					}
				}
			}
			if !anyMissing {
				break
			}
		}
		// Now we can mark overlaps for deletion
		for _, libDir := range libDirs {
			if libDir == newest {
				continue
			}
			for dev, i := range byLibDirs[libDir] {
				if _, found := byLibDirs[newest][dev]; found {
					slog.Debug("filtering device with overlapping libraries",
						"id", dev,
						"library", libDir,
						"delete_index", i,
						"kept_library", newest,
					)
					needsDelete[i] = true
				}
			}
		}
	}
}

type bootstrapRunner struct {
	port int
	cmd  *exec.Cmd
}

func (r *bootstrapRunner) GetPort() int {
	return r.port
}

func (r *bootstrapRunner) HasExited() bool {
	if r.cmd != nil && r.cmd.ProcessState != nil {
		return true
	}
	return false
}

func bootstrapDevices(ctx context.Context, ollamaLibDirs []string, extraEnvs map[string]string) []ml.DeviceInfo {
	var out io.Writer
	if envconfig.LogLevel() == logutil.LevelTrace {
		out = os.Stderr
	}
	start := time.Now()
	defer func() {
		slog.Debug("bootstrap discovery took", "duration", time.Since(start), "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs)
	}()

	logutil.Trace("starting runner for device discovery", "libDirs", ollamaLibDirs, "extraEnvs", extraEnvs)
	cmd, port, err := llm.StartRunner(
		true, // ollama engine
		"",   // no model
		ollamaLibDirs,
		out,
		extraEnvs,
	)
	if err != nil {
		slog.Debug("failed to start runner to discovery GPUs", "error", err)
		return nil
	}

	go func() {
		cmd.Wait() // exit status ignored
	}()

	defer cmd.Process.Kill()
	devices, err := ml.GetDevicesFromRunner(ctx, &bootstrapRunner{port: port, cmd: cmd})
	if err != nil {
		if cmd.ProcessState != nil && cmd.ProcessState.ExitCode() >= 0 {
			// Expected during bootstrapping while we filter out unsupported AMD GPUs
			logutil.Trace("runner exited", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs, "code", cmd.ProcessState.ExitCode())
		} else {
			slog.Info("failure during GPU discovery", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs, "error", err)
		}
	}
	logutil.Trace("runner enumerated devices", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "devices", devices)

	return devices
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
