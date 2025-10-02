package discover

// Runner based GPU discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
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
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

var (
	deviceMu     sync.Mutex
	devices      []ml.DeviceInfo
	libDirs      map[string]struct{}
	rocmDir      string
	exe          string
	bootstrapped bool
)

func GPUDevices(ctx context.Context, runners []FilteredRunnerDiscovery) []ml.DeviceInfo {
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
		files, err := filepath.Glob(filepath.Join(LibOllamaPath, "*", "*ggml-*"))
		if err != nil {
			slog.Debug("unable to lookup runner library directories", "error", err)
		}
		for _, file := range files {
			libDirs[filepath.Dir(file)] = struct{}{}
		}

		// Our current packaging model places ggml-hip in the main directory
		// but keeps rocm in an isolated directory.  We have to add it to
		// the [LD_LIBRARY_]PATH so ggml-hip will load properly
		rocmDir = filepath.Join(LibOllamaPath, "rocm")
		if _, err := os.Stat(rocmDir); err != nil {
			rocmDir = ""
		}

		if len(libDirs) == 0 {
			libDirs[""] = struct{}{}
		}

		slog.Info("discovering available GPUs...")

		// For our initial discovery pass, we gather all the known GPUs through
		// all the libraries that were detected. This pass may include GPUs that
		// are enumerated, but not actually supported.
		// We run this in serial to avoid potentially initializing a GPU multiple
		// times concurrently leading to memory contention
		for dir := range libDirs {
			var dirs []string
			if dir == "" {
				dirs = []string{LibOllamaPath}
			} else {
				dirs = []string{LibOllamaPath, dir}
			}
			// Typically bootstrapping takes < 1s, but on some systems, with devices
			// in low power/idle mode, initialization can take multiple seconds.  We
			// set a long timeout just for bootstrap discovery to reduce the chance
			// of giving up too quickly
			ctx1stPass, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()

			// For this pass, we retain duplicates in case any are incompatible with some libraries
			devices = append(devices, bootstrapDevices(ctx1stPass, dirs, nil)...)
		}

		// In the second pass, we more deeply initialize the GPUs to weed out devices that
		// aren't supported by a given library.  We run this phase in parallel to speed up discovery.
		slog.Debug("filtering out unsupported or overlapping GPU library combinations", "count", len(devices))
		ctx2ndPass, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		var wg sync.WaitGroup
		needsDelete := make([]bool, len(devices))
		supportedMu := sync.Mutex{}
		supported := make(map[string]map[string]map[string]int) // [Library][libDir][ID] = pre-deletion devices index
		for i := range devices {
			libDir := devices[i].LibraryPath[len(devices[i].LibraryPath)-1]
			if devices[i].Library == "Metal" {
				continue
			}
			slog.Debug("verifying GPU is supported", "library", libDir, "description", devices[i].Description, "compute", devices[i].Compute(), "pci_id", devices[i].PCIID)
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				var envVar string
				if devices[i].Library == "ROCm" {
					if runtime.GOOS != "linux" {
						envVar = "HIP_VISIBLE_DEVICES"
					} else {
						envVar = "ROCR_VISIBLE_DEVICES"
					}
				} else {
					envVar = "CUDA_VISIBLE_DEVICES"
				}

				extraEnvs := []string{
					"GGML_CUDA_INIT=1",           // force deep initialization to trigger crash on unsupported GPUs
					envVar + "=" + devices[i].ID, // Filter to just this one GPU
				}
				if len(bootstrapDevices(ctx2ndPass, devices[i].LibraryPath, extraEnvs)) == 0 {
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
		logutil.Trace("supported GPU library combinations", "supported", supported)

		// Mark for deletion any overlaps - favoring the library version that can cover all GPUs if possible
		filterOverlapByLibrary(supported, needsDelete)

		// TODO if we ever support multiple ROCm library versions this algorithm will need to be adjusted to keep the rocmID numeric value correct
		rocmID := 0
		for i := 0; i < len(needsDelete); i++ {
			if needsDelete[i] {
				logutil.Trace("removing unsupported or overlapping GPU combination", "libDir", devices[i].LibraryPath[len(devices[i].LibraryPath)-1], "description", devices[i].Description, "compute", devices[i].Compute(), "pci_id", devices[i].PCIID)
				devices = append(devices[:i], devices[i+1:]...)
				needsDelete = append(needsDelete[:i], needsDelete[i+1:]...)
				i--
			} else if devices[i].Library == "ROCm" {
				if _, err := strconv.Atoi(devices[i].ID); err == nil {
					// Replace the numeric ID with the post-filtered IDs
					devices[i].FilteredID = devices[i].ID
					devices[i].ID = strconv.Itoa(rocmID)
				}
				rocmID++
			}
		}

		// Now filter out any overlap with different libraries (favor CUDA/ROCm over others)
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
					if devices[i].Library == "CUDA" || devices[i].Library == "ROCm" {
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
			if dir != LibOllamaPath {
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
			return devices
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

			for dir := range libDirs {
				updatedDevices := bootstrapDevices(ctx, []string{LibOllamaPath, dir}, nil)
				for _, u := range updatedDevices {
					for i := range devices {
						if u.DeviceID == devices[i].DeviceID {
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

	return devices
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

func bootstrapDevices(ctx context.Context, ollamaLibDirs []string, extraEnvs []string) []ml.DeviceInfo {
	// TODO DRY out with llm/server.go
	slog.Debug("spawing runner with", "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs)
	start := time.Now()
	defer func() {
		slog.Debug("bootstrap discovery took", "duration", time.Since(start), "OLLAMA_LIBRARY_PATH", ollamaLibDirs, "extra_envs", extraEnvs)
	}()
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		var l *net.TCPListener
		if l, err = net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		slog.Debug("ResolveTCPAddr failed, using random port")
		port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
	}
	params := []string{"runner", "--ollama-engine", "--port", strconv.Itoa(port)}
	var pathEnv string
	switch runtime.GOOS {
	case "windows":
		pathEnv = "PATH"
	case "darwin":
		pathEnv = "DYLD_LIBRARY_PATH"
	default:
		pathEnv = "LD_LIBRARY_PATH"
	}
	libraryPaths := append([]string{LibOllamaPath}, ollamaLibDirs...)
	if rocmDir != "" {
		libraryPaths = append(libraryPaths, rocmDir)
	}
	// Note: we always put our dependency paths first
	// since these are the exact version we compiled/linked against
	if libraryPath, ok := os.LookupEnv(pathEnv); ok {
		libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
	}

	cmd := exec.Command(exe, params...)
	cmd.Env = os.Environ()
	if envconfig.LogLevel() == logutil.LevelTrace {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	// cmd.SysProcAttr = llm.LlamaServerSysProcAttr // circular dependency - bring back once refactored
	cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(ollamaLibDirs, string(filepath.ListSeparator)))
	pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))
	pathNeeded := true
	extraDone := make([]bool, len(extraEnvs))
	for i := range cmd.Env {
		cmp := strings.SplitN(cmd.Env[i], "=", 2)
		if strings.EqualFold(cmp[0], pathEnv) {
			cmd.Env[i] = pathEnv + "=" + pathEnvVal
			pathNeeded = false
		} else {
			for j := range extraEnvs {
				if extraDone[j] {
					continue
				}
				extra := strings.SplitN(extraEnvs[j], "=", 2)
				if cmp[0] == extra[0] {
					cmd.Env[i] = extraEnvs[j]
					extraDone[j] = true
				}
			}
		}
	}
	if pathNeeded {
		cmd.Env = append(cmd.Env, pathEnv+"="+pathEnvVal)
	}
	for i := range extraDone {
		if !extraDone[i] {
			cmd.Env = append(cmd.Env, extraEnvs[i])
		}
	}
	logutil.Trace("starting runner for device discovery", "env", cmd.Env, "cmd", cmd)
	if err := cmd.Start(); err != nil {
		slog.Warn("unable to start discovery subprocess", "cmd", cmd, "error", err)
		return nil
	}
	go func() {
		cmd.Wait() // exit status ignored
	}()

	defer cmd.Process.Kill()
	devices, err := GetDevicesFromRunner(ctx, &bootstrapRunner{port: port, cmd: cmd})
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

func GetDevicesFromRunner(ctx context.Context, runner BaseRunner) ([]ml.DeviceInfo, error) {
	var moreDevices []ml.DeviceInfo
	port := runner.GetPort()
	tick := time.Tick(10 * time.Millisecond)
	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("failed to finish discovery before timeout")
		case <-tick:
			r, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/info", port), nil)
			if err != nil {
				return nil, fmt.Errorf("failed to create request: %w", err)
			}
			r.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(r)
			if err != nil {
				// slog.Warn("failed to send request", "error", err)
				if runner.HasExited() {
					return nil, fmt.Errorf("runner crashed")
				}
				continue
			}
			defer resp.Body.Close()

			if resp.StatusCode == http.StatusNotFound {
				// old runner, fall back to bootstrapping model
				return nil, fmt.Errorf("llamarunner free vram reporting not supported")
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				slog.Warn("failed to read response", "error", err)
				continue
			}
			if resp.StatusCode != 200 {
				logutil.Trace("runner failed to discover free VRAM", "status", resp.StatusCode, "response", body)
				return nil, fmt.Errorf("runner error: %s", string(body))
			}

			if err := json.Unmarshal(body, &moreDevices); err != nil {
				slog.Warn("unmarshal encode response", "error", err)
				continue
			}
			return moreDevices, nil
		}
	}
}
