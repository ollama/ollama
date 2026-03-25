package discover

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// llamaServerDiscoverDevices spawns llama-server briefly (without a model) to
// discover GPU devices and their capabilities. The server prints device info
// and system_info (including compiled CUDA architectures) on startup before
// any model load, then we kill it.
//
// Captured from combined stderr output:
//
//	Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes, VRAM: 16379 MiB
//	Device 0: AMD Radeon RX 6700 XT, gfx1031 (0x1031), VMM: no, Wave Size: 32, VRAM: 12272 MiB
//
// Captured from stdout device list:
//
//	CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
//	Metal: Apple M3 Max (98304 MiB, 98303 MiB free)
//
// Captured from system_info line:
//
//	system_info: ... | CUDA : ARCHS = 750,800,860,890,... |
func llamaServerDiscoverDevices(ctx context.Context, libDirs []string, extraEnvs map[string]string) []ml.DeviceInfo {
	devices, _, _, _ := llamaServerDiscoverDevicesWithStatus(ctx, libDirs, extraEnvs)
	return devices
}

func llamaServerDiscoverDevicesWithStatus(ctx context.Context, libDirs []string, extraEnvs map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, int, error) {
	status := llm.NewStatusWriter(llamaServerDiscoveryOutput(ctx))
	llamaServer, err := llm.FindLlamaServer()
	if err != nil {
		slog.Debug("llama-server not available for device discovery", "error", err)
		return nil, status, -1, err
	}

	start := time.Now()
	defer func() {
		slog.Debug("llama-server device discovery took", "duration", time.Since(start), "libDirs", libDirs)
	}()

	// Use a random port to avoid conflicts. The server will start listening
	// but we kill it as soon as we have the system_info output.
	port := 49152 + time.Now().UnixNano()%16383
	cmd := exec.CommandContext(ctx, llamaServer,
		"--port", strconv.FormatInt(port, 10),
		"--host", "127.0.0.1",
		"--no-webui",
		"--offline",
	)
	cmd.Env = os.Environ()

	setupLibraryEnv(cmd, libDirs, extraEnvs)

	logutil.Trace("running llama-server for discovery", "cmd", cmd.Path, "libDirs", libDirs)

	// Capture stderr (device info + system_info) via pipe so we can
	// read it line-by-line and kill the server as soon as we have what we need.
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		slog.Debug("llama-server discovery: failed to create stderr pipe", "error", err)
		return nil, status, -1, err
	}
	// Capture stdout (device list) into a buffer
	var stdoutBuf strings.Builder
	cmd.Stdout = io.MultiWriter(&stdoutBuf, status)

	if err := cmd.Start(); err != nil {
		slog.Debug("llama-server discovery: failed to start", "error", err)
		return nil, status, -1, err
	}

	// Read stderr until we see system_info or timeout
	var stderrLines []string
	gotSystemInfo := false
	done := make(chan struct{})
	go func() {
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			stderrLines = append(stderrLines, line)
			_, _ = status.Write([]byte(line + "\n"))
			if strings.Contains(line, "system_info:") {
				gotSystemInfo = true
				break
			}
		}
		close(done)
	}()

	select {
	case <-done:
	case <-ctx.Done():
	}

	// Kill the server — we have what we need (or timed out)
	stoppedForDiscovery := false
	if cmd.Process != nil {
		stoppedForDiscovery = cmd.Process.Kill() == nil
	}
	waitErr := cmd.Wait()
	exitCode := -1
	if cmd.ProcessState != nil {
		exitCode = cmd.ProcessState.ExitCode()
	}
	if waitErr != nil {
		if stoppedForDiscovery {
			slog.Debug("llama-server discovery: stopped subprocess after collecting GPU info", "libDirs", libDirs)
		} else {
			slog.Debug("llama-server discovery: server startup exited", "error", waitErr, "libDirs", libDirs)
		}
	}
	<-done

	if !gotSystemInfo {
		slog.Warn("llama-server discovery: system_info line not found in output — "+
			"CUDA architecture filtering will be disabled. If GPU inference fails, "+
			"this may indicate an incompatible llama-server version.",
			"libDirs", libDirs, "lines_captured", len(stderrLines))
	}

	// Also run --list-devices to get the stdout device list with free memory
	// (the brief server startup doesn't print that)
	cmd2 := exec.CommandContext(ctx, llamaServer, "--list-devices", "--offline")
	cmd2.Env = cmd.Env // reuse same environment
	listOutput, err := cmd2.CombinedOutput()
	_, _ = status.Write(listOutput)
	if err != nil {
		slog.Debug("llama-server --list-devices failed", "error", err)
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
		return nil, status, exitCode, err
	}

	combined := string(listOutput) + "\n" + strings.Join(stderrLines, "\n")
	return parseLlamaServerDevices(combined, libDirs), status, exitCode, nil
}

func llamaServerDiscoveryOutput(ctx context.Context) io.Writer {
	if slog.Default().Enabled(ctx, logutil.LevelTrace) {
		return os.Stderr
	}
	return io.Discard
}

// setupLibraryEnv configures the command environment for GPU backend discovery:
// sets library paths and GGML_BACKEND_PATH for the GPU backend .so/.dll.
func setupLibraryEnv(cmd *exec.Cmd, libDirs []string, extraEnvs map[string]string) {
	var pathEnv string
	switch runtime.GOOS {
	case "windows":
		pathEnv = "PATH"
	case "darwin":
		pathEnv = "DYLD_LIBRARY_PATH"
	default:
		pathEnv = "LD_LIBRARY_PATH"
	}

	libraryPaths := append([]string{}, libDirs...)
	if libraryPath, ok := os.LookupEnv(pathEnv); ok {
		libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
	}
	pathVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

	pathSet := false
	for i := range cmd.Env {
		if key, _, ok := strings.Cut(cmd.Env[i], "="); ok && strings.EqualFold(key, pathEnv) {
			cmd.Env[i] = pathEnv + "=" + pathVal
			pathSet = true
		}
	}
	if !pathSet {
		cmd.Env = append(cmd.Env, pathEnv+"="+pathVal)
	}

	for k, v := range extraEnvs {
		found := false
		for i := range cmd.Env {
			if key, _, ok := strings.Cut(cmd.Env[i], "="); ok && strings.EqualFold(key, k) {
				cmd.Env[i] = k + "=" + v
				found = true
			}
		}
		if !found {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}

	// Find GPU backend .so/.dll and set GGML_BACKEND_PATH
	backendPatterns := []string{
		"libggml-cuda*", "ggml-cuda*.dll",
		"libggml-hip*", "ggml-hip*.dll",
		"libggml-vulkan*", "ggml-vulkan*.dll",
	}
	for _, dir := range libDirs {
		for _, pattern := range backendPatterns {
			matches, _ := filepath.Glob(filepath.Join(dir, pattern))
			if len(matches) > 0 {
				cmd.Env = append(cmd.Env, "GGML_BACKEND_PATH="+matches[0])
				return
			}
		}
	}
}

// deviceLineRegex matches stdout lines like:
//
//	CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
//	Metal: Apple M3 Max (98304 MiB, 98303 MiB free)
var deviceLineRegex = regexp.MustCompile(
	`^\s+(.+?):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB\s+free\)`,
)

// gfxTargetRegex matches ROCm stderr lines like:
//
//	Device 0: AMD Radeon RX 6700 XT, gfx1031 (0x1031), VMM: no, Wave Size: 32, VRAM: 12272 MiB
//	Device 1: AMD Radeon Pro VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 16368 MiB
var gfxTargetRegex = regexp.MustCompile(
	`Device\s+(\d+):.*,\s+(gfx[0-9a-f]+)[\s:(]`,
)

// cudaCCRegex matches CUDA stderr lines like:
//
//	Device 0: NVIDIA GeForce GTX 1060 6GB, compute capability 6.1, VMM: yes, VRAM: 6063 MiB
var cudaCCRegex = regexp.MustCompile(
	`Device\s+(\d+):.*compute capability\s+(\d+)\.(\d+)`,
)

// cudaArchsRegex matches the CUDA architecture list from system_info like:
//
//	CUDA : ARCHS = 750,800,860,890,900,1000,1030,1100,1200,1210
var cudaArchsRegex = regexp.MustCompile(
	`CUDA\s*:\s*ARCHS\s*=\s*([\d,]+)`,
)

// parseLlamaServerDevices parses the combined output of llama-server discovery.
// It extracts device info, ROCm gfx targets, CUDA compute capabilities, and
// CUDA compiled architecture lists.
func parseLlamaServerDevices(output string, libDirs []string) []ml.DeviceInfo {
	// Extract per-device metadata from stderr
	gfxByIndex := make(map[int]string)
	ccByIndex := make(map[int]string) // "610", "890", etc.
	var cudaArchs []string            // compiled architectures for this variant

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if matches := gfxTargetRegex.FindStringSubmatch(line); matches != nil {
			idx, _ := strconv.Atoi(matches[1])
			gfxByIndex[idx] = matches[2]
		}
		if matches := cudaCCRegex.FindStringSubmatch(line); matches != nil {
			idx, _ := strconv.Atoi(matches[1])
			major, _ := strconv.Atoi(matches[2])
			minor, _ := strconv.Atoi(matches[3])
			ccByIndex[idx] = fmt.Sprintf("%d%d0", major, minor)
		}
		if matches := cudaArchsRegex.FindStringSubmatch(line); matches != nil {
			cudaArchs = strings.Split(matches[1], ",")
		}
	}

	// Validate CUDA devices against compiled architectures
	cudaArchSet := make(map[string]bool, len(cudaArchs))
	for _, arch := range cudaArchs {
		cudaArchSet[strings.TrimSpace(arch)] = true
	}

	// Parse stdout device lines
	var devices []ml.DeviceInfo
	deviceIndex := 0
	scanner = bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		matches := deviceLineRegex.FindStringSubmatch(scanner.Text())
		if matches == nil {
			continue
		}

		name := matches[1]
		description := matches[2]
		totalMiB, _ := strconv.ParseUint(matches[3], 10, 64)
		freeMiB, _ := strconv.ParseUint(matches[4], 10, 64)
		library := inferLibrary(name, description)

		// Skip pseudo-devices like BLAS/Accelerate that report zero memory.
		// These are CPU math libraries, not real GPUs — they shouldn't appear
		// as inference compute devices or inflate the scheduler's GPU count.
		if totalMiB == 0 {
			slog.Debug("skipping pseudo-device with zero memory", "name", name, "description", description)
			deviceIndex++
			continue
		}

		// For CUDA devices, check if this variant supports the device's CC
		if library == "CUDA" {
			cc := ccByIndex[deviceIndex]
			if cc != "" && len(cudaArchSet) > 0 {
				if !cudaArchSet[cc] {
					slog.Info("skipping CUDA device — compute capability not in compiled architectures",
						"device", description, "cc", cc, "archs", cudaArchs,
						"libDirs", libDirs)
					deviceIndex++
					continue
				}
			} else if cc == "" {
				slog.Warn("llama-server discovery: could not determine compute capability for CUDA device — "+
					"architecture filtering disabled for this device. If inference crashes, "+
					"check that the CUDA backend supports this GPU.",
					"device", description, "libDirs", libDirs)
			} else if len(cudaArchSet) == 0 {
				slog.Warn("llama-server discovery: could not determine compiled CUDA architectures — "+
					"architecture filtering disabled. If inference crashes on older GPUs, "+
					"check llama-server system_info output for ARCHS.",
					"device", description, "libDirs", libDirs)
			}
		}

		dev := ml.DeviceInfo{
			DeviceID: ml.DeviceID{
				ID:      strconv.Itoa(deviceIndex),
				Library: library,
			},
			Name:        name,
			Description: description,
			TotalMemory: totalMiB * 1024 * 1024,
			FreeMemory:  freeMiB * 1024 * 1024,
			LibraryPath: libDirs,
			GFXTarget:   gfxByIndex[deviceIndex],
		}

		devices = append(devices, dev)
		deviceIndex++
	}

	return devices
}

// inferLibrary determines the GPU library type from the llama-server device name and description.
func inferLibrary(name, description string) string {
	combined := strings.ToLower(name + " " + description)
	switch {
	case strings.Contains(combined, "cuda"):
		return "CUDA"
	case strings.Contains(combined, "rocm") || strings.Contains(combined, "hip"):
		return "ROCm"
	case strings.Contains(combined, "metal") || strings.Contains(combined, "apple"):
		return "Metal"
	case strings.Contains(combined, "vulkan"):
		return "Vulkan"
	default:
		return description
	}
}

// llamaServerBootstrapDevices discovers GPU devices using llama-server.
// CUDA devices are validated against compiled architectures (from system_info).
// ROCm devices are validated against bundled rocblas kernels.
func llamaServerBootstrapDevices(ctx context.Context, ollamaLibDirs []string, extraEnvs map[string]string) []ml.DeviceInfo {
	devices, _, _, _ := llamaServerBootstrapDevicesWithStatus(ctx, ollamaLibDirs, extraEnvs)
	return devices
}

func llamaServerBootstrapDevicesWithStatus(ctx context.Context, ollamaLibDirs []string, extraEnvs map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, int, error) {
	devices, status, exitCode, err := llamaServerDiscoverDevicesWithStatus(ctx, ollamaLibDirs, extraEnvs)
	if err != nil {
		return devices, status, exitCode, err
	}
	if runtime.GOOS != "linux" {
		return devices, status, exitCode, nil
	}

	hasROCm := false
	for _, d := range devices {
		if d.Library == "ROCm" {
			hasROCm = true
			break
		}
	}
	if !hasROCm {
		return devices, status, exitCode, nil
	}

	return filterUnsupportedROCmDevices(devices, ollamaLibDirs), status, exitCode, nil
}

// rocblasGFXTargets scans the rocblas library directory for supported gfx targets
// by looking for TensileLibrary_lazy_gfxNNNN.dat files.
func rocblasGFXTargets(libDirs []string) map[string]bool {
	targets := make(map[string]bool)
	for _, dir := range libDirs {
		files, _ := filepath.Glob(filepath.Join(dir, "rocblas", "library", "TensileLibrary_lazy_gfx*.dat"))
		for _, f := range files {
			base := filepath.Base(f)
			if t, ok := strings.CutPrefix(base, "TensileLibrary_lazy_"); ok {
				if t, ok = strings.CutSuffix(t, ".dat"); ok {
					targets[t] = true
				}
			}
		}
	}
	return targets
}

// filterUnsupportedROCmDevices removes ROCm devices whose gfx target doesn't have
// matching rocblas kernels bundled.
func filterUnsupportedROCmDevices(devices []ml.DeviceInfo, libDirs []string) []ml.DeviceInfo {
	supported := rocblasGFXTargets(libDirs)
	if len(supported) == 0 {
		return devices
	}

	var filtered []ml.DeviceInfo
	for _, dev := range devices {
		if dev.Library != "ROCm" {
			filtered = append(filtered, dev)
			continue
		}
		gfx := dev.GFXTarget
		if gfx == "" {
			filtered = append(filtered, dev)
			continue
		}
		if supported[gfx] {
			filtered = append(filtered, dev)
		} else {
			slog.Warn("dropping ROCm device — no rocblas support for gfx target",
				"device", dev.Name, "gfx_target", gfx, "supported", supported,
				"hint", "set HSA_OVERRIDE_GFX_VERSION to map to a supported target")
		}
	}
	return filtered
}

// Ensure stderrPipe is fully consumed to avoid blocking
var _ io.Reader
