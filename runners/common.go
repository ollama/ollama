package runners

import (
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"golang.org/x/sys/cpu"

	"github.com/ollama/ollama/envconfig"
)

var (
	runnersDir = ""
	once       = sync.Once{}
)

type CPUCapability uint32

// Override at build time when building base GPU runners
// var GPURunnerCPUCapability = CPUCapabilityAVX

const (
	CPUCapabilityNone CPUCapability = iota
	CPUCapabilityAVX
	CPUCapabilityAVX2
	// TODO AVX512
)

func (c CPUCapability) String() string {
	switch c {
	case CPUCapabilityAVX:
		return "avx"
	case CPUCapabilityAVX2:
		return "avx2"
	default:
		return "no vector extensions"
	}
}

func GetCPUCapability() CPUCapability {
	if cpu.X86.HasAVX2 {
		return CPUCapabilityAVX2
	}
	if cpu.X86.HasAVX {
		return CPUCapabilityAVX
	}
	// else LCD
	return CPUCapabilityNone
}

// Return the location where runners were located
// empty string indicates only builtin is present
func Locate() string {
	once.Do(locateRunnersOnce)
	return runnersDir
}

// searches for runners in a prioritized set of locations
// 1. local build, with executable at the top of the tree
// 2. lib directory relative to executable
func locateRunnersOnce() {
	exe, err := os.Executable()
	if err != nil {
		slog.Debug("runner locate", "error", err)
	}

	paths := []string{
		filepath.Join(filepath.Dir(exe), "llama", "build", runtime.GOOS+"-"+runtime.GOARCH, "runners"),
		filepath.Join(filepath.Dir(exe), envconfig.LibRelativeToExe(), "lib", "ollama", "runners"),
		filepath.Join(filepath.Dir(exe), "lib", "ollama", "runners"),
	}
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			runnersDir = path
			slog.Debug("runners located", "dir", runnersDir)
			return
		}
	}
	// Fall back to built-in
	slog.Debug("no dynamic runners detected, using only built-in")
	runnersDir = ""
}

// Return the well-known name of the builtin runner for the given platform
func BuiltinName() string {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return "metal"
	}
	return "cpu"
}

// directory names are the name of the runner and may contain an optional
// variant prefixed with '_' as the separator. For example, "cuda_v11" and
// "cuda_v12" or "cpu" and "cpu_avx2". Any library without a variant is the
// lowest common denominator
func GetAvailableServers() map[string]string {
	once.Do(locateRunnersOnce)

	servers := make(map[string]string)
	exe, err := os.Executable()
	if err == nil {
		servers[BuiltinName()] = exe
	}

	if runnersDir == "" {
		return servers
	}

	// glob runnersDir for files that start with ollama_
	pattern := filepath.Join(runnersDir, "*", "ollama_*")

	files, err := filepath.Glob(pattern)
	if err != nil {
		slog.Debug("could not glob", "pattern", pattern, "error", err)
		return nil
	}

	for _, file := range files {
		slog.Debug("availableServers : found", "file", file)
		runnerName := filepath.Base(filepath.Dir(file))
		// Special case for our GPU runners - if compiled with standard AVX flag
		// detect incompatible system
		// Custom builds will omit this and its up to the user to ensure compatibility
		parsed := strings.Split(runnerName, "_")
		if len(parsed) == 3 && parsed[2] == "avx" && !cpu.X86.HasAVX {
			slog.Info("GPU runner incompatible with host system, CPU does not have AVX", "runner", runnerName)
			continue
		}
		servers[runnerName] = file
	}

	return servers
}

// serversForGpu returns a list of compatible servers give the provided GPU library/variant
func ServersForGpu(requested string) []string {
	// glob workDir for files that start with ollama_
	availableServers := GetAvailableServers()

	// Short circuit if the only option is built-in
	if _, ok := availableServers[BuiltinName()]; ok && len(availableServers) == 1 {
		return []string{BuiltinName()}
	}

	bestCPUVariant := GetCPUCapability()
	requestedLib := strings.Split(requested, "_")[0]
	servers := []string{}

	// exact match first
	for a := range availableServers {
		short := a
		parsed := strings.Split(a, "_")
		if len(parsed) == 3 {
			// Strip off optional _avx for comparison
			short = parsed[0] + "_" + parsed[1]
		}
		if a == requested || short == requested {
			servers = []string{a}
		}
	}

	// If no exact match, then try without variant
	if len(servers) == 0 {
		alt := []string{}
		for a := range availableServers {
			if requestedLib == strings.Split(a, "_")[0] && a != requested {
				alt = append(alt, a)
			}
		}
		slices.Sort(alt)
		servers = append(servers, alt...)
	}

	// Finally append the best CPU option if found, then builtin
	if bestCPUVariant != CPUCapabilityNone {
		for cmp := range availableServers {
			if cmp == "cpu_"+bestCPUVariant.String() {
				servers = append(servers, cmp)
				break
			}
		}
	}
	servers = append(servers, BuiltinName())
	return servers
}

// Return the optimal server for this CPU architecture
func ServerForCpu() string {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return BuiltinName()
	}
	variant := GetCPUCapability()
	availableServers := GetAvailableServers()
	if variant != CPUCapabilityNone {
		for cmp := range availableServers {
			if cmp == "cpu_"+variant.String() {
				return cmp
			}
		}
	}
	return BuiltinName()
}
