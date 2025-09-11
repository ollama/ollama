package discover

// Runner based GPU discovery

import (
	"bytes"
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
	defer func() {
		slog.Debug("overall device VRAM discovery took", "duration", time.Since(startDiscovery))
	}()

	if !bootstrapped {
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

		libDirs[""] = struct{}{}

		// Typically bootstrapping takes < 1s, but on some systems, with devices
		// in low power/idle mode, initialization can take multiple seconds.  We
		// set a long timeout just for bootstrap discovery to reduce the chance
		// of giving up too quickly
		ctx1stPass, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		slog.Info("discovering available GPUs...")

		for dir := range libDirs {
			// TODO maybe run these in parallel?
			var dirs []string
			if dir == "" {
				dirs = []string{LibOllamaPath}
			} else {
				dirs = []string{LibOllamaPath, dir}
			}
			newDevices := bootstrapDevices(ctx1stPass, dirs, nil)
			for _, dev := range newDevices {
				new := true
				for i := range devices {
					// For this pass, we only drop exact duplicates
					switch dev.Compare(devices[i]) {
					case ml.SameBackendDevice:
						// Same library and device, see if we have a preferred libDir
						if devices[i].IsBetter(dev) {
							devices[i] = dev
						}
						new = false
					}
				}
				if new {
					devices = append(devices, dev)
				}
			}
		}

		// Do a second pass to filter out unsupported AMD GPUs which crash rocBLAS
		envVar := "ROCR_VISIBLE_DEVICES"
		if runtime.GOOS != "linux" {
			envVar = "HIP_VISIBLE_DEVICES"
		}
		slog.Debug("initial GPU list", "count", len(devices))
		ctx2ndPass, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
		var wg sync.WaitGroup
		needsDelete := make([]bool, len(devices))
		for i := range devices {
			if devices[i].Library == "HIP" {
				slog.Info("verifying AMD GPU is supported", "description", devices[i].Description, "compute", devices[i].Compute(), "pci_id", devices[i].PCIID)
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					extraEnvs := []string{
						"GGML_ROCBLAS_INIT=1",        // Initialize rocBLAS to trigger crash on unsupported GPUs
						envVar + "=" + devices[i].ID, // Filter to just this GPU
					}
					if len(bootstrapDevices(ctx2ndPass, devices[i].LibraryPath, extraEnvs)) == 0 {
						needsDelete[i] = true
					}
				}(i)
			}
		}
		wg.Wait()
		rocmID := 0
		for i := 0; i < len(needsDelete); i++ {
			if needsDelete[i] {
				slog.Warn("disabling unsupported GPU", "description", devices[i].Description, "compute", devices[i].Compute(), "pci_id", devices[i].PCIID)
				devices = append(devices[:i], devices[i+1:]...)
				needsDelete = append(needsDelete[:i], needsDelete[i+1:]...)
				i--
			} else if devices[i].Library == "HIP" {
				if _, err := strconv.Atoi(devices[i].ID); err == nil {
					// Replace the numeric ID with the post-filtered IDs
					devices[i].FilteredID = devices[i].ID
					devices[i].ID = strconv.Itoa(rocmID)
				}
				rocmID++
			}
		}

		// Now filter out any overlap with different libraries
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
					if devices[i].Library == "CUDA" || devices[i].Library == "HIP" {
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
					if dev.ID == devices[i].ID && dev.Library == devices[i].Library {
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
					if u.Library == devices[i].Library && u.ID == devices[i].ID {
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
			slog.Debug("unable to refresh all GPUs, doing full bootstrap discovery")

			// Bootstrapping may take longer in some cases (AMD windows), but we
			// would rather use stale free data to get the model running sooner
			ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()

			for dir := range libDirs {
				// TODO maybe run these in parallel?
				updatedDevices := bootstrapDevices(ctx, []string{LibOllamaPath, dir}, nil)
				for _, u := range updatedDevices {
					for i := range devices {
						if u.Library == devices[i].Library && u.ID == devices[i].ID {
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
	slog.Debug("spawing runner with", "lib_dir", ollamaLibDirs, "extra_envs", extraEnvs)
	start := time.Now()
	defer func() {
		slog.Debug("bootstrap discovery took", "duration", time.Since(start), "lib_dir", ollamaLibDirs, "extra_envs", extraEnvs)
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
	cmd.Stdout = os.Stdout
	errBuf := &bytes.Buffer{}
	if envconfig.LogLevel() == slog.Level(-8) {
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stderr = errBuf
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
					extraDone[i] = true
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
	slog.Log(context.TODO(), logutil.LevelTrace, "starting runner for device discovery", "env", cmd.Env, "cmd", cmd)
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
			slog.Log(context.TODO(), logutil.LevelTrace, "runner exited", "lib_dir", ollamaLibDirs, "extra_envs", extraEnvs, "code", cmd.ProcessState.ExitCode())
		} else {
			slog.Info("failure during GPU discovery", "lib_dir", ollamaLibDirs, "extra_envs", extraEnvs, "error", err)
		}
	}
	slog.Debug("got devices", "lib_dir", ollamaLibDirs, "devices", devices)
	return devices
}

func GetDevicesFromRunner(ctx context.Context, runner BaseRunner) ([]ml.DeviceInfo, error) {
	var moreDevices []ml.DeviceInfo
	port := runner.GetPort()
	tick := time.Tick(500 * time.Millisecond)
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
			if resp.StatusCode != 200 {
				slog.Debug("runner failed to discover free VRAM", "status", resp.StatusCode)
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				slog.Warn("failed to read response", "error", err)
				continue
			}

			if err := json.Unmarshal(body, &moreDevices); err != nil {
				slog.Warn("unmarshal encode response", "error", err)
				continue
			}
			return moreDevices, nil
		}
	}
}
