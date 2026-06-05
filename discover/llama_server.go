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

// llamaServerDiscoveryWaitDelay bounds how long Wait can hang after we stop
// the short-lived discovery subprocess.
const llamaServerDiscoveryWaitDelay = 5 * time.Second

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

func llamaServerDiscoverDevices(ctx context.Context, libDirs []string, extraEnvs map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
	status := llm.NewStatusWriter(llamaServerDiscoveryOutput(ctx))
	llamaServer, err := llm.FindLlamaServer()
	if err != nil {
		slog.Debug("llama-server not available for device discovery", "error", err)
		return nil, status, err
	}

	start := time.Now()
	defer func() {
		slog.Debug("llama-server device discovery took", "duration", time.Since(start), "libDirs", libDirs)
	}()

	// Use a random port to avoid conflicts. The server may start listening
	// before it emits system_info, but we stop it as soon as we have the GPU
	// discovery output we need.
	port := 49152 + time.Now().UnixNano()%16383
	cmd := exec.CommandContext(ctx, llamaServer,
		"--port", strconv.FormatInt(port, 10),
		"--host", "127.0.0.1",
		"--no-webui",
		"--offline",
		"--verbose",
	)
	cmd.WaitDelay = llamaServerDiscoveryWaitDelay
	cmd.Env = os.Environ()

	llm.SetupLlamaServerCommandEnv(cmd, llamaServer, libDirs, extraEnvs)

	logutil.Trace("running llama-server for discovery", "cmd", cmd.Path, "libDirs", libDirs)

	// Capture stderr (device info + system_info) via pipe so we can
	// read it line-by-line and kill the server as soon as we have what we need.
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		slog.Debug("llama-server discovery: failed to create stderr pipe", "error", err)
		return nil, status, err
	}
	// Forward stdout through the same status writer so trace logging captures
	// all llama-server discovery output.
	cmd.Stdout = status

	if err := cmd.Start(); err != nil {
		slog.Debug("llama-server discovery: failed to start", "error", err)
		return nil, status, err
	}

	// Read stderr until we see system_info or timeout
	var stderrLines []string
	gotSystemInfo := false
	done := make(chan struct{})
	go func() {
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			_, _ = status.Write([]byte(line + "\n"))
			stderrLines = append(stderrLines, line)
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

	// Kill the server - we have what we need, or timed out.
	stoppedForDiscovery := false
	if cmd.Process != nil {
		stoppedForDiscovery = cmd.Process.Kill() == nil
	}
	waitErr := cmd.Wait()
	if waitErr != nil {
		exit := llm.ExitStatusFromError(waitErr)
		if stoppedForDiscovery {
			slog.Debug("llama-server discovery: stopped subprocess after collecting GPU info", "exit", exit, "libDirs", libDirs)
		}
		if !stoppedForDiscovery {
			slog.Debug("llama-server discovery: server startup exited", "error", waitErr, "exit", exit, "libDirs", libDirs)
		}
	}
	<-done

	if ctx.Err() != nil {
		slog.Warn("llama-server discovery: timed out waiting for server startup", "error", ctx.Err(), "libDirs", libDirs, "lines_captured", len(stderrLines))
		return nil, status, ctx.Err()
	}

	if !gotSystemInfo {
		slog.Warn("llama-server discovery: system_info line not found in output - "+
			"CUDA architecture filtering will be disabled. If GPU inference fails, "+
			"this may indicate an incompatible llama-server version.",
			"libDirs", libDirs, "lines_captured", len(stderrLines))
	}

	// Also run --list-devices to get the stdout device list with free memory
	// (the brief server startup doesn't print that)
	cmd2 := exec.CommandContext(ctx, llamaServer, "--list-devices", "--offline", "--verbose")
	cmd2.WaitDelay = llamaServerDiscoveryWaitDelay
	cmd2.Env = cmd.Env // reuse same environment
	listOutput, err := cmd2.CombinedOutput()
	_, _ = status.Write(listOutput)
	if err != nil {
		exit := llm.ExitStatusFromError(err)
		slog.Debug("llama-server --list-devices failed", "error", err, "exit", exit)
		if exit.Known() {
			return nil, status, fmt.Errorf("llama-server --list-devices failed: %s", exit)
		}
		return nil, status, fmt.Errorf("llama-server --list-devices failed: %w", err)
	}

	nativeDevices, nativeStderr, nativeErr := discoverNativeDevices(ctx, llamaServer, libDirs, extraEnvs)
	_, _ = status.Write([]byte(nativeStderr))
	if nativeErr != nil {
		logNativeProbeFailure(nativeErr, nativeStderr, libDirs)
	}

	combined := string(listOutput) + "\n" + strings.Join(stderrLines, "\n") + "\n" + nativeStderr
	return parseLlamaServerDevicesWithNative(combined, libDirs, nativeDevices), status, nil
}

func llamaServerDiscoveryOutput(ctx context.Context) io.Writer {
	if slog.Default().Enabled(ctx, logutil.LevelTrace) {
		return os.Stderr
	}
	return io.Discard
}

// deviceLineRegex matches stdout lines like:
//
//	CUDA0: NVIDIA GeForce RTX 4060 Ti (16379 MiB, 14900 MiB free)
//	Metal: Apple M3 Max (98304 MiB, 98303 MiB free)
var deviceLineRegex = regexp.MustCompile(
	`^\s+(.+?):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB\s+free\)`,
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

var (
	cudaRuntimeSORegex  = regexp.MustCompile(`^libcudart\.so\.(\d+)(?:\.(\d+))?`)
	cudaRuntimeDLLRegex = regexp.MustCompile(`^cudart64_(\d{2})(\d)\.dll$`)
	cudaRuntimeDirRegex = regexp.MustCompile(`^cuda_v(\d+)$`)
)

// parseLlamaServerDevices parses the combined output of llama-server discovery.
// It extracts device info, ROCm gfx targets, CUDA compute capabilities, and
// CUDA compiled architecture lists.
func parseLlamaServerDevices(output string, libDirs []string) []ml.DeviceInfo {
	return parseLlamaServerDevicesWithNative(output, libDirs, nil)
}

func parseLlamaServerDevicesWithNative(output string, libDirs []string, nativeDevices []nativeProbeDevice) []ml.DeviceInfo {
	// Extract per-device metadata from stderr
	gfxByIndex := parseROCmGFXTargets(output)
	rocmGFXOverride := hsaOverrideGFXTarget()
	integratedByIndex := parseVulkanUMA(output)
	ccByIndex := make(map[int]cudaComputeCapability)
	var cudaArchs []string // compiled architectures for this variant
	nativeByIndex := nativeProbeByLibraryIndex(nativeDevices)
	for idx, dev := range nativeByIndex["ROCm"] {
		if rocmGFXOverride != "" {
			gfxByIndex[idx] = rocmGFXOverride
		} else if dev.GFXTarget != "" {
			gfxByIndex[idx] = dev.GFXTarget
		}
	}

	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()
		if matches := cudaCCRegex.FindStringSubmatch(line); matches != nil {
			idx, _ := strconv.Atoi(matches[1])
			major, _ := strconv.Atoi(matches[2])
			minor, _ := strconv.Atoi(matches[3])
			ccByIndex[idx] = cudaComputeCapability{
				major: major,
				minor: minor,
				arch:  fmt.Sprintf("%d%d0", major, minor),
			}
		}
		if matches := cudaArchsRegex.FindStringSubmatch(line); matches != nil {
			cudaArchs = strings.Split(matches[1], ",")
		}
	}
	if cudaDevices := nativeByIndex["CUDA"]; len(cudaDevices) > 0 {
		for idx, dev := range cudaDevices {
			if dev.ComputeMajor <= 0 {
				continue
			}
			ccByIndex[idx] = cudaComputeCapability{
				major: dev.ComputeMajor,
				minor: dev.ComputeMinor,
				arch:  fmt.Sprintf("%d%d0", dev.ComputeMajor, dev.ComputeMinor),
			}
		}
	}

	// Validate CUDA devices against compiled architectures
	cudaArchSet := make(map[string]bool, len(cudaArchs))
	for _, arch := range cudaArchs {
		cudaArchSet[strings.TrimSpace(arch)] = true
	}
	cudaRuntimeMajor, cudaRuntimeMinor, hasCUDARuntime := cudaRuntimeVersion(libDirs)

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
			cc, ok := ccByIndex[deviceIndex]
			if ok && len(cudaArchSet) > 0 {
				if !cudaArchSet[cc.arch] {
					slog.Info("skipping CUDA device — compute capability not in compiled architectures",
						"device", description, "cc", cc.arch, "archs", cudaArchs,
						"libDirs", libDirs)
					deviceIndex++
					continue
				}
			} else if !ok {
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

		nativeDevice, hasNativeDevice := nativeByIndex[library][deviceIndex]
		totalBytes := totalMiB * 1024 * 1024
		if hasNativeDevice && !nativeProbeMatchesLlamaServerDevice(library, description, totalBytes, nativeDevice) {
			hasNativeDevice = false
		}
		computeMajor, computeMinor := computeVersion(library, deviceIndex, gfxByIndex, ccByIndex)
		dev := ml.DeviceInfo{
			DeviceID: ml.DeviceID{
				ID:      strconv.Itoa(deviceIndex),
				Library: library,
			},
			Name:         name,
			Description:  description,
			TotalMemory:  totalBytes,
			FreeMemory:   freeMiB * 1024 * 1024,
			ComputeMajor: computeMajor,
			ComputeMinor: computeMinor,
			LibraryPath:  libDirs,
			GFXTarget:    gfxByIndex[deviceIndex],
			Integrated:   isIntegratedLlamaServerDevice(library, deviceIndex, integratedByIndex),
		}
		if hasNativeDevice {
			if nativeDevice.DeviceID != "" {
				dev.PCIID = nativeDevice.DeviceID
			}
			if nativeDevice.IntegratedKnown {
				dev.Integrated = nativeDevice.Integrated
			} else {
				dev.Integrated = dev.Integrated || nativeDevice.Integrated
			}
			if dev.ComputeMajor == 0 && nativeDevice.ComputeMajor > 0 {
				dev.ComputeMajor = nativeDevice.ComputeMajor
				dev.ComputeMinor = nativeDevice.ComputeMinor
			}
			if nativeDevice.CUDADriverMajor > 0 {
				dev.DriverMajor = nativeDevice.CUDADriverMajor
				dev.DriverMinor = nativeDevice.CUDADriverMinor
			}
			if nativeDevice.NVIDIADriverMajor > 0 {
				dev.NVIDIADriverMajor = nativeDevice.NVIDIADriverMajor
			}
			setROCmGFXTarget(&dev, nativeDevice.GFXTarget)
		}
		setROCmGFXTarget(&dev, rocmGFXOverride)
		if library == "CUDA" && dev.DriverMajor == 0 && hasCUDARuntime {
			dev.DriverMajor = cudaRuntimeMajor
			dev.DriverMinor = cudaRuntimeMinor
		}

		devices = append(devices, dev)
		deviceIndex++
	}

	return refineLlamaServerDevices(devices, libDirs)
}

func nativeProbeMatchesLlamaServerDevice(library, description string, totalBytes uint64, nativeDevice nativeProbeDevice) bool {
	if library != "Vulkan" {
		return true
	}

	nativeDescription := nativeDevice.Description
	if nativeDescription == "" {
		nativeDescription = nativeDevice.Name
	}
	if nativeDescription == "" || !ml.SimilarDeviceDescription(description, nativeDescription) {
		slog.Debug("skipping Vulkan native metadata with mismatched device name",
			"llama_server_name", description,
			"native_name", nativeDescription)
		return false
	}
	if nativeDevice.TotalMemory != 0 && !ml.SimilarDeviceMemory(totalBytes, nativeDevice.TotalMemory) {
		slog.Debug("skipping Vulkan native metadata with mismatched memory",
			"llama_server_name", description,
			"llama_server_total", totalBytes,
			"native_total", nativeDevice.TotalMemory)
		return false
	}

	return true
}

func cudaRuntimeVersion(libDirs []string) (int, int, bool) {
	bestMajor, bestMinor := -1, -1
	update := func(major, minor int) {
		if major > bestMajor || (major == bestMajor && minor > bestMinor) {
			bestMajor, bestMinor = major, minor
		}
	}

	for _, dir := range libDirs {
		for _, entry := range readDirNames(dir) {
			if matches := cudaRuntimeSORegex.FindStringSubmatch(entry); matches != nil {
				major, _ := strconv.Atoi(matches[1])
				minor := 0
				if matches[2] != "" {
					minor, _ = strconv.Atoi(matches[2])
				}
				update(major, minor)
			}
			if matches := cudaRuntimeDLLRegex.FindStringSubmatch(entry); matches != nil {
				major, _ := strconv.Atoi(matches[1])
				minor, _ := strconv.Atoi(matches[2])
				update(major, minor)
			}
		}

		if matches := cudaRuntimeDirRegex.FindStringSubmatch(filepath.Base(dir)); matches != nil {
			major, _ := strconv.Atoi(matches[1])
			update(major, 0)
		}
	}

	if bestMajor < 0 {
		return 0, 0, false
	}
	return bestMajor, bestMinor, true
}

func readDirNames(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names
}

type cudaComputeCapability struct {
	major int
	minor int
	arch  string
}

func computeVersion(library string, deviceIndex int, gfxByIndex map[int]string, ccByIndex map[int]cudaComputeCapability) (int, int) {
	switch library {
	case "CUDA":
		if cc, ok := ccByIndex[deviceIndex]; ok {
			return cc.major, cc.minor
		}
	case "ROCm":
		return parseGFXTarget(gfxByIndex[deviceIndex])
	}
	return 0, 0
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

func isIntegratedLlamaServerDevice(library string, deviceIndex int, integratedByIndex map[int]bool) bool {
	if library == "Vulkan" && integratedByIndex[deviceIndex] {
		return true
	}

	// llama-server discovery does not expose a stable backend device-type field,
	// so we only infer "integrated" here for cases where the contract is strong:
	// explicit Vulkan UMA metadata, or the single Apple Silicon Metal device.
	//
	// Other backends stay unclassified unless discovery provides a stronger
	// signal. That keeps scheduling conservative instead of guessing from
	// device names or backend-specific heuristics.
	return library == "Metal" && runtime.GOOS == "darwin" && runtime.GOARCH == "arm64"
}

func llamaServerBootstrapDevicesWithStatus(ctx context.Context, ollamaLibDirs []string, extraEnvs map[string]string) ([]ml.DeviceInfo, *llm.StatusWriter, error) {
	devices, status, err := llamaServerDiscoverDevices(ctx, ollamaLibDirs, extraEnvs)
	if err != nil {
		return devices, status, err
	}

	hasROCm := false
	for _, d := range devices {
		if d.Library == "ROCm" {
			hasROCm = true
			break
		}
	}
	if !hasROCm {
		return devices, status, nil
	}

	return filterUnsupportedROCmDevices(devices, ollamaLibDirs), status, nil
}

// Ensure stderrPipe is fully consumed to avoid blocking
var _ io.Reader
